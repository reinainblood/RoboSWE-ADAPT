"""Repeated real-renderer recording lifecycle control for long-lived sessions."""

import argparse
import gc
import json
from pathlib import Path

from .session import RobotSession


def rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux RSS counter unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cycles", type=int, default=16)
    args = parser.parse_args()
    if args.cycles < 6:
        parser.error("Use at least six cycles to measure growth after warmup")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    session = RobotSession(
        {
            "robot": "PandaOmron",
            "task": "robocasa/Kitchen",
            "seed": 0,
            "split": None,
            "env_kwargs": {
                "layout_ids": 2,
                "style_ids": 1,
                "init_robot_base_ref": "microwave_main_group",
                "camera_widths": 128,
                "camera_heights": 128,
            },
        }
    )
    samples = []
    try:
        session.recording_root = directory / "recordings"
        initial = session.fingerprint()
        for index in range(args.cycles):
            key = f"cycle-{index:02d}"
            session.start_recording({"recording_id": key, "width": 640, "height": 480})
            session.stop_recording({"recording_id": key})
            gc.collect()
            recording = session.recordings[key]
            assert (
                recording.overview is None
                and not recording.writers
                and not recording.last_frames
            )
            assert session.fingerprint() == initial and session.steps == 0
            samples.append({"cycle": index, "rss_bytes": rss_bytes()})
        # Allocator warmup may retain arenas. A growing renderer/encoder per task
        # is a different failure; test its growth after the first three cycles.
        growth = (
            max(x["rss_bytes"] for x in samples[3:])
            - min(x["rss_bytes"] for x in samples[3:])
            if len(samples) > 3
            else 0
        )
        report = {
            "kind": "recording_lifecycle_control_not_task_trial",
            "cycles": args.cycles,
            "samples": samples,
            "post_warmup_rss_growth_bytes": growth,
            "allowed_growth_bytes": 256 * 1024 * 1024,
            "simulator_unchanged": True,
            "passed": growth < 256 * 1024 * 1024,
        }
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        assert report["passed"], report
        print(
            json.dumps(
                {
                    "passed": True,
                    "cycles": args.cycles,
                    "growth_mib": growth / 2**20,
                    "last_rss_gib": samples[-1]["rss_bytes"] / 2**30,
                }
            )
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
