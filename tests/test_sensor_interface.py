import base64
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

from roboswe.library.runtime import public_observation
from robot_session.environment import RobotEnvironment
from robot_session.interface import controller_parameters, metric_depth


def sim():
    return SimpleNamespace(
        model=SimpleNamespace(
            stat=SimpleNamespace(extent=2),
            vis=SimpleNamespace(map=SimpleNamespace(znear=0.05, zfar=5)),
        )
    )


def test_depth_has_metric_units_and_rejects_invalid_sensor_values():
    np.testing.assert_allclose(
        metric_depth(sim(), np.array([[0.0, 1.0]])), [[0.1, 10]], rtol=1e-5
    )
    for value in (float("nan"), -1, 1.1):
        with pytest.raises(ValueError):
            metric_depth(sim(), np.array([[value]]))


def test_benchmark_depth_whitelist_only_exports_enabled_robot_cameras():
    env = SimpleNamespace(
        camera_keys=["robot0_eye_image"],
        action_schema=dict,
        camera_calibration=lambda *_: {},
    )
    session = SimpleNamespace(
        env=env,
        raw=SimpleNamespace(sim=sim()),
        steps=0,
        phase="active",
        observation={
            "robot0_eye_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_depth": np.zeros((2, 2, 1)),
            "privileged_overview_depth": np.ones((2, 2, 1)),
            "object-state": np.ones(3),
        },
    )
    observation = public_observation(session)
    with np.load(
        BytesIO(base64.b64decode(observation["depth_npz_base64"])), allow_pickle=False
    ) as arrays:
        assert arrays.files == ["robot0_eye_depth"]
        np.testing.assert_allclose(arrays["robot0_eye_depth"], 0.1)
    assert not observation["sensors"]
    assert "Metadata only" in observation["depth_data_contract"]["depth_arrays"]
    assert "SAME observation" in observation["depth_data_contract"]["calibration"]


def test_step_receipt_distinguishes_physics_tick_from_goal_attainment():
    from roboswe.library.runtime import controller_reply

    observation = {"control_step": 12, "sensors": {"position": [0, 0, 0]}, "task_success": False}
    stepped = controller_reply(observation, "step")
    observed = controller_reply(observation, "observe")
    assert stepped["controller_receipt"]["completed_control_steps"] == 1
    assert observed["controller_receipt"]["completed_control_steps"] == 0
    assert observed["controller_receipt"]["physics_advanced"] is False
    assert stepped["task_success"] is False
    assert stepped["sensors"] == observation["sensors"]
    assert "controller_receipt" not in observation


def test_image_orientation_is_not_flipped_twice_for_opencv_observations():
    env = object.__new__(RobotEnvironment)
    env.env = SimpleNamespace(get_ep_meta=dict)
    raw = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    env._raw_images_bottom_up = False
    np.testing.assert_array_equal(
        env._observation({"cam_image": raw})["cam_image"], raw
    )
    env._raw_images_bottom_up = True
    np.testing.assert_array_equal(
        env._observation({"cam_image": raw})["cam_image"], raw[::-1]
    )


def test_controller_contract_does_not_publish_controller_goals_or_task_geometry():
    part = SimpleNamespace(
        input_type="delta",
        input_ref_frame="base",
        output_max=np.array([0.05] * 3),
        goal_pos=np.array([1, 2, 3]),
        fixture_location=[4, 5, 6],
    )
    result = controller_parameters(SimpleNamespace(part_controllers={"right": part}))[
        "right"
    ]
    assert result["input_ref_frame"] == "base" and result["output_max"] == [0.05] * 3
    assert "goal_pos" not in result and "fixture_location" not in result


def test_opt_in_observation_cache_invalidates_and_is_copy_safe():
    calls=[]
    def schema():
        calls.append(1);return {'shape':[1]}
    session=SimpleNamespace(env=SimpleNamespace(camera_keys=[],action_schema=schema),
        observation={'robot0_eef_pos':np.array([1.,2.,3.])},steps=0,phase='active',_cache_public_observation=True)
    first=public_observation(session)
    first['sensors']['robot0_eef_pos'][0]=999
    second=public_observation(session)
    assert second['sensors']['robot0_eef_pos']==[1.,2.,3.]
    assert len(calls)==1
    session.steps=1
    assert public_observation(session)['control_step']==1
    session.observation={'robot0_eef_pos':np.array([4.,5.,6.])}
    assert public_observation(session)['sensors']['robot0_eef_pos']==[4.,5.,6.]
    session.phase='faulted'
    assert public_observation(session)['simulator_phase']=='faulted'
    assert len(calls)==4


def test_nested_rgb_documentation_is_opt_in_and_does_not_change_values():
    session=SimpleNamespace(env=SimpleNamespace(camera_keys=[],action_schema=dict),
        observation={'robot0_eef_pos':np.array([1.,2.,3.])},steps=0,phase='active')
    original=public_observation(session)
    assert 'image_data_contract' not in original
    session._nested_sensor_documentation=True
    updated=public_observation(session)
    contract=updated.pop('image_data_contract')
    assert updated==original
    assert "['images_png_base64'][camera_key]" in contract['rgb_payload']
    assert 'JSON list' in contract['sensor_vectors']
