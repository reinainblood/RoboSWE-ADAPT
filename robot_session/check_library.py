"""Explicit real-simulator predicate controls, not an agent success campaign.

Direct joint writes are allowed only in these test controls. Generated agent code
must never get this access to the simulator or use these states as solutions.
"""

import argparse
import json
from pathlib import Path

from roboswe.library.spec import GOALS, library_set
from roboswe.library.verifiers import bind_fixtures, evaluate, read_positions

from .session import RobotSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
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
    s = RobotSession(config)
    results = []
    try:
        bindings = bind_fixtures(s.raw)
        initial = read_positions(s.raw, bindings)
        for name, goal in GOALS.items():
            b = bindings[goal.fixture]
            fixture = s.raw.fixtures[b.name]
            samples = (0.0, 0.2, 0.5, 0.96, 1.0)
            for fraction in samples:
                for joint in b.joints:
                    q = (
                        joint.high - (joint.high - joint.low) * fraction
                        if joint.low < 0
                        else joint.low + (joint.high - joint.low) * fraction
                    )
                    s.raw.sim.data.set_joint_qpos(joint.name, q)
                s.raw.sim.forward()
                positions = read_positions(s.raw, bindings)
                actual = evaluate(name, b, positions)
                if goal.fixture == "drawer":
                    values = fixture.get_door_state(s.raw).values()
                    upstream = (
                        all(v >= 0.95 for v in values)
                        if goal.predicate == "open"
                        else all(v <= 0.05 for v in values)
                    )
                elif goal.fixture == "sink":
                    water = fixture.get_handle_state(s.raw)["water_on"]
                    upstream = water if goal.predicate == "water_on" else not water
                else:
                    upstream = (
                        fixture.is_open(s.raw)
                        if goal.predicate == "open"
                        else fixture.is_closed(s.raw)
                    )
                assert actual["success"] == bool(upstream), (
                    name,
                    fraction,
                    actual,
                    upstream,
                )
                results.append(
                    {
                        "task": name,
                        "normalized_test_position": fraction,
                        "ours": actual["success"],
                        "upstream": bool(upstream),
                    }
                )
        result = {
            "passed": True,
            "kind": "synthetic_state_predicate_controls_not_agent_trials",
            "spec_sha256": spec["sha256"],
            "config": config,
            "initial_fingerprint": s.initial,
            "bindings": {k: b.to_dict() for k, b in bindings.items()},
            "initial_positions": initial,
            "controls": results,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Passed {len(results)} real-model predicate comparisons: {path}")
    finally:
        s.close()


if __name__ == "__main__":
    main()
