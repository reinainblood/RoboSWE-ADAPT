"""Real-simulator code/exception/replay control. Deliberately solves no task."""

import argparse
import json
import subprocess
from pathlib import Path

from roboswe.execution.sandbox import CodeSandbox, docker_path
from roboswe.library.attempts import AttemptRunner, Journal, replay_journal
from roboswe.library.simulator import SimulatorProcess
from roboswe.library.spec import library_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--simulator-image",
        help="Immutable Docker simulator image; omit for explicit local development",
    )
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    image = subprocess.check_output(
        [
            docker_path(),
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            "roboswe-code:local",
        ],
        text=True,
    ).strip()
    spec = library_set()
    config = {
        "robot": spec["robot"],
        "task": "robocasa/Kitchen",
        "split": None,
        "seed": 0,
        "env_kwargs": {
            "init_robot_base_ref": spec["initial_robot_base_ref"],
            "layout_ids": spec["layout_id"],
            "style_ids": spec["style_id"],
            "camera_widths": 64,
            "camera_heights": 64,
        },
    }
    source = "import robot\no=robot.observe()\na=[0.0]*o['action_schema']['shape'][0]\nfor _ in range(3): robot.step(a)\nraise RuntimeError('intentional partial attempt')\n"

    def decision(operation, path="", content=""):
        return {
            "operation": operation,
            "path": path,
            "content": content,
            "message": "infrastructure control",
        }

    simulator_args = {
        "image": args.simulator_image,
        "local_development": args.simulator_image is None,
    }
    live = SimulatorProcess(config, spec, directory / "original", **simulator_args)
    journal = Journal(directory / "submissions.jsonl")
    try:
        live.record_start("original", width=320, height=240)
        with CodeSandbox(image) as sandbox:
            runner = AttemptRunner(live, sandbox, journal)
            runner.start_task(0)
            runner.apply(decision("write", "probe.py", source))
            failed = runner.apply(decision("execute", "probe.py"))
            assert failed["returncode"] != 0 and live.observe()["control_step"] == 3, (
                failed
            )
            runner.apply(
                decision(
                    "write",
                    "probe.py",
                    "import robot\nrobot.step([0.0]*robot.observe()['action_schema']['shape'][0])\n",
                )
            )
            result = runner.apply(decision("execute", "probe.py"))
            assert result["returncode"] == 0 and live.observe()["control_step"] == 4, (
                result
            )
            verdict = runner.apply(decision("finish"))
            assert not verdict["observed_success"]
    finally:
        journal.close()
        live.close()
    fresh = SimulatorProcess(
        config,
        spec,
        directory / "verification",
        **simulator_args,
        role="verifier",
    )
    try:
        fresh.record_start("replay", width=320, height=240)
        with CodeSandbox(image) as sandbox:
            report = replay_journal(
                directory / "submissions.jsonl",
                fresh,
                sandbox,
                directory / "replay.jsonl",
            )
        assert report["reproduced"] and fresh.observe()["control_step"] == 4, report
        assert not any(t["verified_success"] for t in report["trials"])
        report["control"] = (
            "Zero actions, intentional failure after 3 steps, revision, 1 more step. Not a solver or successful benchmark trial."
        )
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Independent code replay passed: {directory}", flush=True)
    finally:
        fresh.close()


if __name__ == "__main__":
    main()
