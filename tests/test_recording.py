import json
from types import SimpleNamespace

import av
import numpy as np
import pytest

from robot_session.recording import Recording
from robot_session.session import RobotSession


class FakeOverview:
    def __init__(self, raw, width, height):
        self.raw = raw
        self.shape = (height, width, 3)

    def frame(self):
        return np.full(
            self.shape, min(255, int(self.raw.sim.data.time * 500)), dtype=np.uint8
        )

    def close(self):
        pass


def session():
    return SimpleNamespace(
        id="recording-test",
        steps=0,
        raw=SimpleNamespace(
            control_freq=20, sim=SimpleNamespace(data=SimpleNamespace(time=0.0))
        ),
        env=SimpleNamespace(camera_keys=["wrist_image"]),
        observation={"wrist_image": np.zeros((64, 64, 3), dtype=np.uint8)},
    )


def test_both_clocks_and_idle_do_not_advance_physics(tmp_path):
    s = session()
    clock = [0.0]
    r = Recording(
        s,
        "attempt",
        tmp_path,
        64,
        64,
        clock=lambda: clock[0],
        overview_factory=FakeOverview,
    )
    for tick in range(1, 4):
        clock[0] = tick * 0.1
        s.steps = tick
        s.raw.sim.data.time = tick * 0.05
        s.observation["wrist_image"][:] = tick * 50
        r.capture()
    clock[0] = 2.3
    r.idle()
    assert s.steps == 3 and s.raw.sim.data.time == pytest.approx(0.15)
    clock[0] = 3.0
    manifest = r.stop()
    assert r.overview is None and not r.writers and not r.last_frames
    assert r.stop() == manifest
    assert manifest["start_step"] == 0 and manifest["end_step"] == 3
    for view in manifest["views"]:
        with av.open(r.directory / view["simulation"]) as c:
            frames = list(c.decode(video=0))
            assert len(frames) == 4
            assert [f.time for f in frames] == pytest.approx(
                [0, 0.05, 0.1, 0.15], abs=1e-4
            )
        with av.open(r.directory / view["wall"]) as c:
            frames = list(c.decode(video=0))
            assert [f.time for f in frames] == pytest.approx(
                [0, 0.1, 0.2, 0.3, 2.3, 3.0], abs=1e-4
            )
    events = [
        json.loads(line)
        for line in (r.directory / "video-timeline.jsonl").read_text().splitlines()
    ]
    assert events[-2]["event"] == "waiting" and events[-2]["step"] == 3
    assert events[-1]["event"] == "stop"
    assert len(events) == 6


def test_step_failure_keeps_partial_recording(tmp_path):
    s = session()
    r = Recording(s, "failed", tmp_path, 64, 64, overview_factory=FakeOverview)
    s.recording = r
    # Test the same recorder error boundary used by the session process.
    r.overview.frame = lambda: (_ for _ in ()).throw(RuntimeError("render broke"))
    RobotSession.record_event(s, "step")
    assert not r.active and r.manifest["status"] == "recording_error"
    assert "render broke" in r.manifest["errors"][0]
    with av.open(r.directory / "video-overview-simulation.mp4") as c:
        assert len(list(c.decode(video=0))) == 1


def test_recording_settings_and_existing_ids(tmp_path):
    with pytest.raises(ValueError):
        Recording(session(), "../escape", tmp_path)
    with pytest.raises(ValueError):
        Recording(session(), "odd", tmp_path, 65, 64)
    r = Recording(session(), "same", tmp_path, 64, 64, overview_factory=FakeOverview)
    r.stop()
    with pytest.raises(FileExistsError):
        Recording(session(), "same", tmp_path, 64, 64, overview_factory=FakeOverview)
