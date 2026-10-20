"""Exercise matched-state cold controls with neutral code, not a solver model."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from roboswe.campaign import run_sequence
from roboswe.execution.sandbox import docker_path
from roboswe.library.simulator import SimulatorProcess
from roboswe.library.spec import library_set


def decision(operation, path="", content=""):
    return {
        "operation": operation,
        "path": path,
        "content": content,
        "message": "Neutral protocol control, not a task-solving attempt",
    }


class ScriptedParticipant:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.metadata = {
            "model": "scripted-neutral-control",
            "reasoning_effort": "medium",
        }

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def ask(self, *_, **kwargs):
        return next(self.decisions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--simulator-image")
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    code_image = subprocess.check_output(
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
    spec.update(id="neutral-cold-control-not-benchmark", task_timeout_seconds=30)
    spec["tasks"] = [dict(spec["tasks"][0], index=i, round=i + 1) for i in range(2)]
    spec.pop("sha256")
    spec["sha256"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    config = {
        "robot": "PandaOmron",
        "task": "robocasa/Kitchen",
        "split": None,
        "seed": 0,
        "env_kwargs": {
            "init_robot_base_ref": spec["initial_robot_base_ref"],
            "layout_ids": 2,
            "style_ids": 1,
            "camera_widths": 64,
            "camera_heights": 64,
        },
    }
    plan = {
        "mode": "library",
        "jobs_dir": str(directory / "jobs"),
        "agent": {"model": None, "reasoning_effort": "medium"},
        "environment": {"camera_width": 320, "camera_height": 240},
        "execution": {
            "image": args.simulator_image,
            "code_image": code_image,
            "timeout_seconds": 30,
            "max_control_steps": 2,
            "max_attempt_steps": 2,
            "attempt_timeout_seconds": 20,
            "lifetime_seconds": 300,
            "learning_controls": True,
            "video_judge": False,
        },
    }
    warm = (
        "import robot\nrobot.step([0.0]*robot.observe()['action_schema']['shape'][0])\n"
    )
    cold = (
        "from pathlib import Path\nassert not Path('learned.py').exists(), 'Warm library leaked into cold condition'\n"
        + warm
    )
    participants = iter(
        [
            [
                decision("write", "learned.py", warm),
                decision("execute", "learned.py"),
                decision("finish"),
                decision("execute", "learned.py"),
                decision("finish"),
            ],
            [
                decision("write", "cold.py", cold),
                decision("execute", "cold.py"),
                decision("finish"),
            ],
        ]
    )

    def model_factory(**_):
        return ScriptedParticipant(next(participants))

    def simulator_factory(config, spec, path, **kwargs):
        if args.simulator_image is None:
            kwargs.pop("image")
            kwargs["local_development"] = True
        return SimulatorProcess(config, spec, path, **kwargs)

    report = run_sequence(
        plan,
        spec,
        config,
        directory,
        model_factory=model_factory,
        simulator_factory=simulator_factory,
    )
    assert report["reproduced"]
    comparison = report["metrics"]["controlled_comparison"]
    assert comparison["status"] == "controlled_pilot_estimate", comparison
    assert comparison["library_score"] == 0
    controls = json.loads(
        next(directory.glob("sequence-*/cold-controls/controls.json")).read_text()
    )
    assert controls[0]["control_steps"] == 1 and controls[0]["execution_errors"] == 0, (
        controls
    )
    assert controls[0]["start_fingerprint"] == report["trials"][1]["start_fingerprint"]
    report["control"] = (
        "Neutral code; no task success. Warm file persisted, cold file-access assertion passed, physical starting states matched, cold code reran independently."
    )
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Cold control and independent replay passed: {directory}")


if __name__ == "__main__":
    main()
