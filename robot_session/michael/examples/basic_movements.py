"""Basic real-time SimBody movements from pandas tables, with a recorded video.

    .venv/bin/python -m robot_session.michael.examples.basic_movements \
        --output artifacts/michael-body-demo

Physics runs continuously; this script mostly sleeps while motions execute.
It is a commissioning example, not a task or a hardware procedure.
"""

import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd

from robot_session.michael.body import SimBody, touch

ELBOW, WRIST = "right_arm.joint4", "right_arm.joint7"


def wait(handle, label, receipts):
    result = handle.result(timeout=60)
    receipts.append(
        {
            "label": label,
            "status": result.status.value,
            "reason": result.reason,
            "trigger": result.trigger,
            "rows_commanded": f"{result.rows_commanded}/{result.rows_total}",
            "attained": result.attained,
            "max_abs_target_error": float(result.target_error.abs().max()),
        }
    )
    print(f"{label}: {result.status.value} ({result.reason})", flush=True)
    return result


def there_and_back(q, deltas, seconds):
    """Smooth-ish out-and-back for several position channels."""
    t = np.linspace(seconds / 8, seconds, 8)
    shape = np.sin(np.pi * t / seconds) ** 2
    return pd.DataFrame({"t": t, **{c: q[c] + d * shape for c, d in deltas.items()}})


def main(output: Path, fps: float):
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    with SimBody(clock="realtime") as body:
        recording = body.record("overview", fps=fps, width=640, height=480)
        start = body.state()
        time.sleep(1.0)  # idle: physics keeps running, holding the pose
        idle = body.state()
        q = idle.joints.commanded

        arm = body.run(there_and_back(q, {ELBOW: 0.35, WRIST: 0.5}, 3.0))
        flex = {
            f"left_hand.{f}.mcp_flex": 0.5
            for f in ("index_finger", "middle_finger", "ring_finger", "pinky")
        }
        hand = body.run(there_and_back(q, flex, 3.0))  # disjoint resource, concurrent
        time.sleep(1.0)  # e.g. model inference while both move
        wait(arm, "right elbow+wrist out and back", receipts)
        wait(hand, "left fingers flex and open", receipts)

        lift = body.run(there_and_back(q, {"lift.height": 0.04}, 3.0))
        wait(lift, "lift up 4 cm and back", receipts)

        pose0 = body.state().base_pose
        base = body.run(
            pd.DataFrame(
                {
                    "t": [1.0, 2.5, 3.5, 4.5, 6.0, 7.0],
                    "base.vx": [0.15, 0.15, 0.0, 0.0, 0.0, 0.0],
                    "base.wz": [0.0, 0.0, 0.0, 0.4, 0.4, 0.0],
                }
            )
        )
        wait(base, "base forward then turn", receipts)
        time.sleep(0.3)
        pose1 = body.state().base_pose

        body.sim_place_probe("right_hand", "index_finger", gap_m=0.008)
        q = body.state().joints.commanded
        fingers = ("right_hand.index_finger.mcp_flex", "right_hand.index_finger.pip")
        close = pd.DataFrame(
            {
                "t": [2.0],
                fingers[0]: [q[fingers[0]] + 0.4],
                fingers[1]: [q[fingers[1]] + 0.3],
            }
        )
        felt = wait(
            body.run(close, until=touch("right_hand.index_finger", 0.1)),
            "right index closes until simulated touch",
            receipts,
        )
        time.sleep(0.5)
        body.sim_remove_probe()
        reopen = pd.DataFrame({"t": [1.0], **{c: [q[c]] for c in fingers}})
        wait(body.run(reopen), "right index reopens", receipts)
        time.sleep(0.5)

        frames = recording.stop()
        end = body.state()
        body.telemetry().to_parquet(output / "telemetry-last-10s.parquet")
        body.events().to_csv(output / "events.csv", index=False)
        felt.log().to_csv(output / "touch-motion-log.csv")

    with av.open(str(output / "basic-movements.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=round(fps))
        stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
        for frame in frames:
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(frame.rgb, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    h = end.health
    report = {
        "task_attempted": False,
        "idle_check": {
            "sim_s": idle.sim_time - start.sim_time,
            "wall_s": idle.wall_time - start.wall_time,
        },
        "whole_run": {
            "sim_s": end.sim_time - start.sim_time,
            "wall_s": end.wall_time - start.wall_time,
            "late_ticks": h.late_ticks,
            "dropped_s": h.dropped_s,
            "max_tick_s": h.max_tick_s,
            "mean_tick_s": h.mean_tick_s,
            "physics_warnings": h.physics_warnings,
            "faulted": h.faulted,
        },
        "base_pose_change": (pose1 - pose0).to_dict(),
        "video_frames": len(frames),
        "frame_sim_times": [frames[0].sim_time, frames[-1].sim_time] if frames else [],
        "motions": receipts,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    print(json.dumps(report["whole_run"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/michael-body-demo")
    )
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    main(args.output, args.fps)
