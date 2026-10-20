"""Session infrastructure tests; no task verifier or benchmark assertions."""

import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError

import gymnasium as gym
import numpy as np
import pytest

from robot_session import backends
from robot_session.client import SessionClient
from robot_session.session import RobotSession, normalize_config, serve


class Environment:
    def __init__(self, *args, **kwargs):
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float64)
        self.camera_keys = []
        self.unwrapped = SimpleNamespace(env=self)
        self.sim = SimpleNamespace(
            model=SimpleNamespace(get_xml=lambda: "<mujoco/>"),
            get_state=lambda: np.zeros(2),
        )
        self.calls = 0
        self.closed = False
        self.fail = False

    def action_schema(self):
        return {"format": "native_vector", "shape": [1]}

    def reset(self, seed):
        return {"state.position": np.zeros(1)}, {}

    def step(self, action):
        self.calls += 1
        if self.fail:
            raise RuntimeError("physics failure")
        return {"state.position": np.ones(1) * self.calls}, 0, False, False, {}

    def close(self):
        self.closed = True


@pytest.fixture
def session():
    result = RobotSession({"robot": "PandaOmron"}, factory=Environment)
    yield result
    result.close()


def test_http_lifetime_and_hold(session):
    # Reserve a local port for this bounded integration test.
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    stopping = [False]
    thread = threading.Thread(target=serve, args=(session, "127.0.0.1", port, stopping))
    thread.start()
    client = SessionClient(f"http://127.0.0.1:{port}")
    try:
        import time

        for _ in range(50):
            try:
                initial = client.status()
                break
            except OSError:
                time.sleep(0.02)
        else:
            pytest.fail("server did not start")
        client.observe()
        assert session.env.calls == 0
        with pytest.raises(HTTPError) as error:
            client.step([float("nan")])
        assert error.value.code == 400
        assert session.env.calls == 0
        obs, *_ = client.step([0.25])
        np.testing.assert_equal(obs["state.position"], [1])
        assert client.status()["session_id"] == initial["session_id"]
        assert not session.env.closed
        assert client.finish()["phase"] == "held"
        assert client.finish()["phase"] == "held"
        with pytest.raises(HTTPError) as error:
            client.step([0])
        assert error.value.code == 409
        assert client.status()["steps"] == 1
        assert not session.env.closed
        client.observe()
    finally:
        stopping[0] = True
        thread.join(3)
        assert not thread.is_alive()


def test_failing_physics_freezes_mutation(session):
    session.env.fail = True
    with pytest.raises(RuntimeError):
        session.step([0])
    assert session.phase == "faulted"
    with pytest.raises(ValueError):
        session.step([0])
    assert session.env.calls == 1


def test_mismatched_entrypoint_closes_environment():
    created = []

    def factory(*args, **kwargs):
        env = Environment()
        created.append(env)
        return env

    with pytest.raises(ValueError, match="Initial scene mismatch"):
        RobotSession({"robot": "PandaOmron"}, expected_initial="wrong", factory=factory)
    assert created[0].closed


def test_fresh_replay_preserves_image_and_entrypoint(monkeypatch, tmp_path):
    original = backends.SessionHandle(
        "docker",
        "old-container",
        "sha256:immutable",
        {"seed": 7},
        "agent",
        "initial",
        cpu=8,
        memory_mb=20000,
    )
    original.save(tmp_path / "original.json")
    original = backends.SessionHandle.load(tmp_path / "original.json")
    calls = []
    monkeypatch.setattr(backends, "start", lambda *a, **k: calls.append((a, k)))
    backends.replay(original, receipt=tmp_path / "replay.json")
    args, kwargs = calls[0]
    assert args == ("docker", "sha256:immutable", {"seed": 7})
    assert kwargs["role"] == "verifier"
    assert kwargs["expected_initial"] == "initial"
    assert kwargs["cpu"] == 8
    assert "old-container" not in json.dumps(
        (args, {k: str(v) for k, v in kwargs.items()})
    )


@pytest.mark.parametrize(
    "config", [{"seed": None}, {"seed": -1}, {"env_kwargs": {"seed": 1}}, {"typo": 3}]
)
def test_replay_config_rejects_ambiguous_inputs(config):
    with pytest.raises(ValueError):
        normalize_config({"robot": "PandaOmron", **config})


def test_docker_checkpoint_retains_container_and_resumes_same_daemon(
    monkeypatch, tmp_path
):
    import subprocess
    from dataclasses import asdict

    status = {
        "session_id": "session-1",
        "initial_fingerprint": "initial",
        "steps": 3,
        "phase": "active",
    }
    handle = backends.SessionHandle(
        "docker", "container-1", "sha256:base", {}, "agent", "initial"
    )
    monkeypatch.setattr(backends.SessionHandle, "status", lambda self: status)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **k: json.dumps(
            {"ExperimentalBuild": True, "OSType": "linux", "ID": "daemon-1"}
        ),
    )
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    checkpoint = tmp_path / "checkpoint.json"
    record = handle.checkpoint(checkpoint)
    assert record["state"] == "ready"
    assert calls[0][:4] == ["docker", "checkpoint", "create", "container-1"]
    assert handle.checkpointed
    restored = backends.resume(checkpoint, receipt=tmp_path / "restored.json")
    assert asdict(restored) == asdict(handle)
    assert calls[1] == [
        "docker",
        "start",
        "--checkpoint",
        record["checkpoint_id"],
        "container-1",
    ]
    handle.stop()
    assert calls[-1][:2] == ["docker", "stop"]
    assert not any("rm" in command for command in calls)
    with pytest.raises(ValueError, match="completed"):
        backends.resume(checkpoint, receipt=tmp_path / "again.json")


def test_docker_checkpoint_fails_without_capability(monkeypatch, tmp_path):
    import subprocess

    handle = backends.SessionHandle("docker", "container", "image", {}, "agent")
    monkeypatch.setattr(backends.SessionHandle, "status", lambda self: {})
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **k: '{"OSType":"linux","ExperimentalBuild":false}',
    )
    with pytest.raises(RuntimeError, match="CRIU"):
        handle.checkpoint(tmp_path / "checkpoint.json")
    assert not (tmp_path / "checkpoint.json").exists()


def test_modal_checkpoint_uses_memory_snapshot_and_checks_restore(
    monkeypatch, tmp_path
):
    import sys

    status = {
        "session_id": "logical-session",
        "initial_fingerprint": "initial",
        "steps": 5,
        "phase": "held",
    }
    handle = backends.SessionHandle(
        "modal", "sb-original", "im-base", {}, "agent", "initial", "FDR"
    )
    monkeypatch.setattr(backends.SessionHandle, "status", lambda self: status)
    monkeypatch.setattr(
        backends.SessionHandle,
        "sandbox",
        lambda self: SimpleNamespace(
            _experimental_snapshot=lambda: SimpleNamespace(object_id="snap-memory")
        ),
    )
    record = handle.checkpoint(tmp_path / "checkpoint.json")
    assert record["checkpoint_id"] == "snap-memory"
    assert record["session"]["image"] == "im-base"
    fake_modal = SimpleNamespace(
        SandboxSnapshot=SimpleNamespace(from_id=lambda x: x),
        Sandbox=SimpleNamespace(
            _experimental_from_snapshot=lambda snapshot: SimpleNamespace(
                object_id="sb-resumed"
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    restored = backends.resume(
        tmp_path / "checkpoint.json", receipt=tmp_path / "resumed.json"
    )
    assert restored.container_id == "sb-resumed"
    assert restored.image == "im-base"
    status["steps"] = 0
    with pytest.raises(RuntimeError, match="steps differs"):
        backends.resume(
            tmp_path / "checkpoint.json", receipt=tmp_path / "bad-resume.json"
        )
    assert (tmp_path / "bad-resume.json").exists()


@pytest.mark.parametrize("robot", [None, "", [], 12])
def test_robot_name_is_required(robot):
    with pytest.raises(ValueError, match="explicit robot"):
        normalize_config({"robot": robot})


@pytest.mark.parametrize("key", ["robot", "robots", "controller_configs"])
def test_robot_cannot_be_overridden_in_env_kwargs(key):
    with pytest.raises(ValueError, match="Reserved"):
        normalize_config({"robot": "PandaOmron", "env_kwargs": {key: "GR1"}})


def test_selected_robot_reaches_factory_and_identity():
    requested = []

    def factory(task, **kwargs):
        requested.append(kwargs["robot"])
        return Environment()

    first = RobotSession({"robot": "PandaOmron"}, factory=factory)
    second = RobotSession({"robot": "GR1FloatingBody"}, factory=factory)
    try:
        assert requested == ["PandaOmron", "GR1FloatingBody"]
        assert second.status()["robot"] == "GR1FloatingBody"
        assert first.initial != second.initial
    finally:
        first.close()
        second.close()


def test_wrong_robot_action_size_is_rejected_before_physics(session):
    with pytest.raises(ValueError, match="shape"):
        session.step([0, 0])
    with pytest.raises(ValueError, match="numeric action vector"):
        session.step({"action.gripper_close": [0]})
    with pytest.raises(ValueError, match="finite"):
        session.step([float("inf")])
    assert session.env.calls == 0


def test_cameras_are_discovered_and_explicit_invalid_cameras_fail():
    from robot_session.environment import RobotEnvironment

    adapter = object.__new__(RobotEnvironment)
    adapter.robot_name = "example"
    adapter.env = SimpleNamespace()
    adapter._requested_cameras = None
    adapter._camera_options = {
        "camera_widths": 64,
        "camera_heights": 64,
        "camera_depths": False,
    }
    xml = '<mujoco><worldbody><camera name="robot0_eye_in_left_hand"/><camera name="robot0_eye_in_right_hand"/></worldbody></mujoco>'
    assert adapter._select_cameras(xml) == xml
    assert adapter.env.camera_names == [
        "robot0_eye_in_left_hand",
        "robot0_eye_in_right_hand",
    ]
    assert adapter.env.camera_widths == [64, 64]
    adapter._requested_cameras = ["robot0_eye_in_hand"]
    with pytest.raises(ValueError, match="Invalid cameras"):
        adapter._select_cameras(xml)


def test_whole_body_schema_uses_composite_inputs_not_joint_dimensions(monkeypatch):
    # This tests adapter schema projection, not a real controller or simulator.
    import sys
    from types import ModuleType
    module = ModuleType("robosuite.controllers.composite.composite_controller")
    module.HybridMobileBase = type("HybridMobileBase", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    from robot_session.environment import RobotEnvironment

    adapter = object.__new__(RobotEnvironment)
    controller = SimpleNamespace(
        action_limits=(np.full(14, -np.inf), np.full(14, np.inf)),
        get_action_info_dict=lambda: {
            "Action Dimension": (14,),
            "right_pose": (0, 6),
            "left_pose": (6, 12),
            "grippers": (12, 14),
        },
    )
    adapter.env = SimpleNamespace(
        robots=[SimpleNamespace(composite_controller=controller, robot_model=object())]
    )
    adapter.action_space = gym.spaces.Box(-np.inf, np.inf, (14,), dtype=np.float64)
    schema = adapter.action_schema()
    assert schema["shape"] == [14]
    assert schema["controllers"][0]["parts"]["left_pose"] == [6, 12]
    assert schema["low"] == [None] * 14
    json.dumps(schema, allow_nan=False)


def test_internal_step_can_skip_unused_serialization(session,monkeypatch):
    import robot_session.session as module
    session._encode_step_observation=False
    def forbidden(*args,**kwargs):raise AssertionError('Unused observation was encoded')
    monkeypatch.setattr(module,'npz_bytes',forbidden)
    result=session.step([0.25])
    assert 'observation_npz' not in result
    assert session.steps==session.env.calls==1
    np.testing.assert_equal(session.observation['state.position'],[1])
    assert session.phase=='active'
