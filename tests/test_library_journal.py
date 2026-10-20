import json
from typing import ClassVar

import pytest

from roboswe.library.attempts import Journal, read_journal
from roboswe.library.runtime import public_observation


def test_journal_detects_code_edits_and_reordering(tmp_path):
    path = tmp_path / "submissions.jsonl"
    journal = Journal(path)
    journal.append("initial", {"fingerprint": "example"})
    journal.append("operation_request", {"source": "robot.step([0])"})
    journal.close()
    assert len(read_journal(path)) == 2
    original = path.read_text()
    path.write_text(original.replace("robot.step([0])", "robot.step([1])"))
    with pytest.raises(ValueError, match="changed or reordered"):
        read_journal(path)
    path.write_text("\n".join(reversed(original.splitlines())) + "\n")
    with pytest.raises(ValueError, match="changed or reordered"):
        read_journal(path)


def test_sensor_whitelist_excludes_privileged_environment_state():
    class Array:
        def tolist(self):
            return [1, 2, 3]

    class Environment:
        camera_keys = ()

        def action_schema(self):
            return {"shape": [12]}

    class Session:
        env = Environment()
        steps = 7
        phase = "active"
        observation: ClassVar[dict] = {
            name: Array()
            for name in [
                "robot0_eef_pos",
                "robot0_left_eef_pos",
                "robot0_right_gripper_qpos",
                "robot0_left_object_to_eef_pos",
                "robot0_base_pos",
                "object-state",
                "robot0_object_to_eef_pos",
                "oven_joint",
                "annotation.human.task_description",
                "robot0_camera_segmentation",
                "task_success",
            ]
        }

    observed = public_observation(Session())
    assert set(observed["sensors"]) == {
        "robot0_eef_pos",
        "robot0_base_pos",
        "robot0_left_eef_pos",
        "robot0_right_gripper_qpos",
    }
    assert "oven_joint" not in json.dumps(observed)
    assert observed["control_step"] == 7
