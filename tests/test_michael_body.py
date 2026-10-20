"""Behavioural tests for Michael's Body/SimBody on the real MuJoCo model."""

import os
import pytest

if os.environ.get("ROBOSWE_TEST_MUJOCO") != "1":
    pytest.skip("Set ROBOSWE_TEST_MUJOCO=1 after preparing the real MuJoCo body environment", allow_module_level=True)

import threading
import time

import numpy as np
import pandas as pd

from robot_session.michael.body import (
    Body,
    BodyClosed,
    BodyError,
    BodyFaulted,
    Reached,
    ResourceBusy,
    Signal,
    SimBody,
    Status,
    TableError,
    touch,
)
from robot_session.michael.runtime import make_environment

ELBOW = "right_arm.joint4"
INDEX = ("right_hand.index_finger.mcp_flex", "right_hand.index_finger.pip")


@pytest.fixture(scope="module")
def shared():
    body = SimBody(clock="manual")
    body.advance(0.3)
    yield body
    body.close()


@pytest.fixture
def body(shared):
    yield shared
    shared.cancel_all(wait=False)
    shared.advance(1.0)
    assert shared.state().health.active_motions == ()


def commanded(body):
    return body.state().joints.commanded


def ramp(channel, start, delta, seconds=1.0):
    return pd.DataFrame({"t": [seconds], channel: [start + delta]})


def test_catalogue_names_every_native_axis_with_units(shared):
    assert isinstance(shared, Body)
    channels = shared.channels()
    assert len(channels) == 58
    counts = channels.groupby("resource").size().to_dict()
    assert counts == {
        "right_arm": 7,
        "left_arm": 7,
        "right_hand": 20,
        "left_hand": 20,
        "base": 3,
        "lift": 1,
    }
    assert set(
        channels.loc[channels.resource.str.endswith(("_arm", "_hand")), "unit"]
    ) == {"rad"}
    assert channels.loc["lift.height", ["mode", "unit"]].tolist() == ["position", "m"]
    assert channels.loc[["base.vx", "base.vy", "base.wz"], "unit"].tolist() == [
        "m/s",
        "m/s",
        "rad/s",
    ]
    assert set(channels.loc[channels.resource == "base", "mode"]) == {"velocity"}
    assert "right_hand.thumb.cmc_flex" in channels.index
    assert "right_hand.index_finger.touch_N" in shared.signals()


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (pd.DataFrame({"t": [1.0, 2.0], ELBOW: [1.3, np.nan]}), "Missing"),
        (pd.DataFrame({"t": [1.0], "right_arm.joint99": [0.0]}), "Unknown"),
        (pd.DataFrame({"t": [1.0, 1.0], ELBOW: [1.3, 1.3]}), "increasing"),
        (pd.DataFrame({"t": [0.0, 1.0], ELBOW: [1.3, 1.3]}), "t > 0"),
        (pd.DataFrame({"t": [5.0], ELBOW: [9.0]}), "outside"),
        (pd.DataFrame({"t": [1.0, 1.1], ELBOW: [1.3, 2.0]}), "envelope"),
        (pd.DataFrame({"t": [1.0], "base.vx": [0.1]}), "last row"),
        (pd.DataFrame({"t": [1.0, 2.0], "base.vy": [0.1, 0.0]}), "differential"),
        (pd.DataFrame({"t": [0.1, 1.0], "base.vx": [0.2, 0.0]}), "accelerates"),
        (pd.DataFrame({"t": [1.0], ELBOW: [True]}), "numeric"),
        (pd.DataFrame({ELBOW: [1.3]}), "'t'"),
        (pd.DataFrame({"t": [1.0], ELBOW: [1.3 + 0.25j]}), "real numeric"),
        (pd.DataFrame({"t": [1.0 + 0j], ELBOW: [1.3]}), "real numeric"),
    ],
)
def test_invalid_tables_are_rejected_before_any_motion(body, table, message):
    before = body.state()
    with pytest.raises(TableError, match=message):
        body.run(table)
    assert body.state().health.active_motions == ()
    assert body.state().tick == before.tick


def test_first_segment_rate_is_checked_against_the_live_command(body):
    q = commanded(body)[ELBOW]
    # Within bounds and with no internal segments, but 0.5 rad in 0.1 s from now.
    with pytest.raises(TableError, match="envelope"):
        body.run(ramp(ELBOW, q, 0.5, seconds=0.1))
    assert body.state().health.active_motions == ()


def test_omitted_resources_and_channels_hold_while_one_joint_moves(body):
    before = body.state().joints
    handle = body.run(ramp(ELBOW, before.commanded[ELBOW], 0.3, seconds=0.6))
    assert handle.resources == {"right_arm"}
    body.advance(1.0)
    result = handle.result(0)
    after = body.state().joints
    assert result.status is Status.COMPLETED
    assert after.measured[ELBOW] - before.measured[ELBOW] > 0.25
    others = after.index != ELBOW
    np.testing.assert_array_equal(after.commanded[others], before.commanded[others])
    held = after.resource.isin(["left_arm", "left_hand", "lift"])
    assert np.abs(after.measured[held] - before.measured[held]).max() < 0.01


def test_conflicting_resources_are_rejected_and_disjoint_resources_run_together(body):
    q = commanded(body)
    arm = body.run(ramp(ELBOW, q[ELBOW], -0.2))
    with pytest.raises(ResourceBusy):
        body.run(ramp("right_arm.joint1", q["right_arm.joint1"], 0.1))
    hand = body.run(
        ramp(
            "left_hand.index_finger.mcp_flex", q["left_hand.index_finger.mcp_flex"], 0.2
        )
    )
    assert set(body.state().health.active_motions) == {arm.id, hand.id}
    body.advance(1.1)
    assert arm.result(0).status is Status.COMPLETED
    assert hand.result(0).status is Status.COMPLETED
    # Released leases can be taken again.
    body.run(ramp(ELBOW, q[ELBOW] - 0.2, 0.2)).cancel()


def test_completed_is_not_attainment_and_until_reports_measured_outcomes(body):
    q = commanded(body)[ELBOW]
    plain = body.run(ramp(ELBOW, q, 0.2, seconds=0.5))
    body.advance(0.6)
    plain = plain.result(0)
    assert plain.status is Status.COMPLETED
    assert plain.rows_commanded == plain.rows_total == 1
    assert abs(plain.target_error[ELBOW]) > 0  # a measurement, reported not judged

    met = body.run(ramp(ELBOW, q + 0.2, -0.2, seconds=0.5), until=Reached(rad=0.08))
    body.advance(2.0)
    met = met.result(0)
    assert met.status is Status.CONDITION_MET
    assert met.attained is True
    assert met.trigger["condition"].startswith("reached")

    # The servo's gravity error exceeds 1e-4 rad, so this is never met.
    missed = body.run(
        ramp(ELBOW, q, 0.2, seconds=0.5), until=Reached(rad=1e-4), timeout_s=1.0
    )
    body.advance(1.2)
    missed = missed.result(0)
    assert missed.status is Status.TIMED_OUT
    assert missed.attained is False
    assert missed.rows_commanded == 1
    body.run(ramp(ELBOW, q + 0.2, -0.2, seconds=0.5))
    body.advance(0.6)


def test_cancel_stops_progress_and_holds_the_last_command(body):
    q = commanded(body)[ELBOW]
    handle = body.run(pd.DataFrame({"t": [1.0, 2.0], ELBOW: [q + 0.3, q + 0.6]}))
    body.advance(0.5)
    result = handle.cancel()
    assert result.status is Status.CANCELLED
    assert result.rows_commanded == 0
    assert 0.1 < result.final_command[ELBOW] - q < 0.2
    body.advance(1.0)
    state = body.state().joints.loc[ELBOW]
    assert state.commanded == result.final_command[ELBOW]
    assert pd.isna(state.owner)
    assert handle.cancel().status is Status.CANCELLED  # idempotent
    body.run(ramp(ELBOW, state.commanded, q - state.commanded, seconds=0.5))
    body.advance(0.6)


def test_snapshots_and_receipts_are_isolated_copies(body):
    q = commanded(body)[ELBOW]
    handle = body.run(ramp(ELBOW, q, 0.1, seconds=0.3))
    body.advance(0.4)
    result = handle.result(0)
    state = body.state()
    joints = state.joints
    joints.loc[ELBOW, "commanded"] = 99.0
    joints["measured"] = -5.0
    log = result.log()
    log.iloc[:, 2:] = 0.0
    final = result.final_command
    final[:] = 42.0
    assert state.joints.loc[ELBOW, "commanded"] == pytest.approx(q + 0.1)
    assert result.final_command[ELBOW] == pytest.approx(q + 0.1)
    assert result.log()[("commanded", ELBOW)].iloc[-1] == pytest.approx(q + 0.1)
    body.advance(0.1)
    assert body.state().joints.loc[ELBOW, "commanded"] == pytest.approx(q + 0.1)
    with pytest.raises(ValueError):
        state._commanded[0] = 1.0
    body.run(ramp(ELBOW, q + 0.1, -0.1, seconds=0.3))
    body.advance(0.4)


def test_base_velocity_expires_and_cancel_ramps_to_rest(body):
    pose = body.state().base_pose
    handle = body.run(pd.DataFrame({"t": [1.0, 2.0, 3.0], "base.vx": [0.2, 0.2, 0.0]}))
    body.advance(3.3)
    assert handle.result(0).status is Status.COMPLETED
    state = body.state()
    assert state.base_pose.x_m - pose.x_m == pytest.approx(0.4, abs=0.02)
    assert np.abs(state.base.commanded).max() == 0
    assert np.abs(state.base.measured).max() < 1e-3

    handle = body.run(pd.DataFrame({"t": [0.5, 5.0, 5.5], "base.vx": [0.2, 0.2, 0.0]}))
    body.advance(1.5)
    assert handle.cancel(wait=False) is None
    assert not handle.done()  # manual clock: the ramp needs simulated time
    body.advance(0.6)
    result = handle.result(0)
    # Cancel decelerates at the envelope (0.5 m/s^2): about 0.4 s from 0.2 m/s.
    assert result.status is Status.CANCELLED
    assert 0.35 <= result.end_time - result.stop_time <= 0.45
    body.advance(0.3)
    assert np.abs(body.state().base.measured).max() < 1e-3


def test_base_forward_follows_heading_after_turning(body):
    before = body.state().base_pose
    turn = body.run(pd.DataFrame({"t": [1.0, 3.14, 4.14], "base.wz": [0.5, 0.5, 0.0]}))
    body.advance(2.0)
    turning = body.state().base
    assert turning.measured["base.wz"] == pytest.approx(0.5, abs=0.01)
    assert abs(turning.measured["base.vy"]) < 1e-3  # measured at the yaw axis
    body.advance(2.4)
    assert turn.result(0).status is Status.COMPLETED
    start = body.state().base_pose
    assert np.hypot(start.x_m - before.x_m, start.y_m - before.y_m) < 1e-3
    assert start.yaw_rad == pytest.approx(np.pi / 2, abs=0.05)
    body.run(pd.DataFrame({"t": [1.0, 2.0, 3.0], "base.vx": [0.2, 0.2, 0.0]}))
    body.advance(3.3)
    moved = (body.state().base_pose - start)[["x_m", "y_m"]].to_numpy()
    heading = np.array([np.cos(start.yaw_rad), np.sin(start.yaw_rad)])
    assert moved @ heading == pytest.approx(0.4, abs=0.02)
    assert abs(heading[0] * moved[1] - heading[1] * moved[0]) < 0.01


def _close_index(body, **kwargs):
    q = commanded(body)
    table = pd.DataFrame(
        {"t": [2.0], INDEX[0]: [q[INDEX[0]] + 0.4], INDEX[1]: [q[INDEX[1]] + 0.3]}
    )
    return q, body.run(table, **kwargs)


def _reopen(body, q):
    body.sim_remove_probe()
    body.run(pd.DataFrame({"t": [1.5], **{c: [q[c]] for c in INDEX}}))
    body.advance(1.6)


def test_finger_stops_on_simulated_fingertip_contact(body):
    body.sim_place_probe("right_hand", "index_finger", gap_m=0.008)
    body.advance(0.1)
    q, handle = _close_index(body, until=touch("right_hand.index_finger", 0.1))
    body.advance(2.2)
    result = handle.result(0)
    assert result.status is Status.CONDITION_MET
    assert result.trigger["signal"] == "right_hand.index_finger.touch_N"
    assert result.trigger["value"] > 0.1
    assert result.table_time_commanded < 1.5  # stopped mid-table, not at the end
    log = result.log()
    assert log[("signal", "right_hand.index_finger.touch_N")].iloc[0] == 0
    body.advance(0.5)
    after = body.state().joints.commanded
    assert (
        after[INDEX[0]] == result.final_command[INDEX[0]]
    )  # no further pushing target
    _reopen(body, q)


def test_touch_guard_trips_without_a_model_call(body):
    body.sim_place_probe("right_hand", "index_finger", gap_m=0.008)
    body.advance(0.1)
    q, handle = _close_index(
        body, guard=[touch("right_hand", 0.1)], stop_hold="measured"
    )
    body.advance(2.2)
    result = handle.result(0)
    assert result.status is Status.GUARDED
    assert result.stop_hold == "measured"
    log = result.log()
    # The measured hold was sent in one more tick before the handle resolved:
    # final_command is what physics received, equal to the position at the trip.
    assert log[("table", "phase")].iloc[-1] == "stopping"
    assert result.final_command[INDEX[0]] == log[("commanded", INDEX[0])].iloc[-1]
    assert result.final_command[INDEX[0]] == log[("measured", INDEX[0])].iloc[-2]
    _reopen(body, q)


def test_telemetry_and_events_stay_bounded_while_idle():
    with SimBody(clock="manual", telemetry_s=0.5) as body:
        body.advance(3.0)
        telemetry = body.telemetry()
        assert len(telemetry) == 50
        assert telemetry.index.is_monotonic_increasing
        assert telemetry.index[-1] == pytest.approx(body.state().sim_time)
        assert body.state().health.physics_warnings == 0
        assert len(body.events()) < 10


class _FlakyEnv:
    """Real env whose step can fail before physics runs, or after it advanced."""

    def __init__(self, env):
        self._env, self.fail, self.after = env, None, {}

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        if self.fail == "before":
            raise RuntimeError("injected simulator failure")
        out = self._env.step(action)
        if self.fail == "after":
            data, model = self._env.sim.data._data, self._env.sim.model._model
            joint = model.joint("robot0_openarm_right_joint7")
            self.after = {
                "time": float(data.time),
                "q": float(data.qpos[joint.qposadr[0]]),
            }
            raise RuntimeError("injected failure after physics advanced")
        return out


def test_worker_fault_terminates_handles_and_refuses_new_work():
    holder = {}

    def factory(hz):
        holder["env"] = _FlakyEnv(make_environment(control_freq=hz))
        return holder["env"]

    body = SimBody(clock="manual", env_factory=factory)
    q = commanded(body)[ELBOW]
    handle = body.run(ramp(ELBOW, q, 0.3))
    before = body.advance(0.2)
    holder["env"].fail = "before"
    with pytest.raises(BodyFaulted):
        body.advance(0.2)
    result = handle.result(timeout=5)
    assert result.status is Status.FAULTED
    assert "injected" in result.reason
    state = body.state()
    assert state.health.faulted
    assert state.sim_time == state.physics_time == before.sim_time
    assert result.measured_time == before.sim_time
    assert result.final_command[ELBOW] == before.joints.commanded[ELBOW]
    assert result.attempted_command[ELBOW] > result.final_command[ELBOW]
    with pytest.raises(BodyFaulted):
        body.run(ramp(ELBOW, q, 0.1))
    body.close()


def test_close_cancels_motions_and_rejects_later_submissions():
    body = SimBody(clock="manual")
    q = commanded(body)[ELBOW]
    handle = body.run(ramp(ELBOW, q, 0.3))
    body.advance(0.2)
    body.close()
    result = handle.result(timeout=0)
    assert result.status is Status.CANCELLED
    assert "closed" in result.reason
    assert handle.cancel().status is Status.CANCELLED
    with pytest.raises(BodyClosed):
        body.run(ramp(ELBOW, q, 0.1))
    with pytest.raises(BodyClosed):
        body.capture()
    assert body.state().health.running is False
    body.close()  # idempotent


def test_realtime_clock_advances_while_caller_is_idle_and_moves_joints():
    with SimBody(clock="realtime") as body:
        start = body.state()
        q = start.joints.commanded[ELBOW]
        handle = body.run(ramp(ELBOW, q, 0.3, seconds=0.8))
        time.sleep(1.5)  # e.g. waiting on model inference; nothing else is called
        end = body.state()
        assert handle.done()
        assert handle.result().status is Status.COMPLETED
        elapsed_wall = end.wall_time - start.wall_time
        assert end.sim_time - start.sim_time == pytest.approx(elapsed_wall, rel=0.1)
        assert end.joints.measured[ELBOW] - start.joints.measured[ELBOW] > 0.25
        assert end.health.dropped_s == 0
        frame = body.capture("right_hand", 320, 240)
        assert frame.rgb.shape == (240, 320, 3) and not frame.rgb.flags.writeable
        assert frame.sim_time >= end.sim_time


def test_concurrent_submitters_get_exactly_one_lease():
    with SimBody(clock="manual") as body:
        q = commanded(body)[ELBOW]
        outcomes = []

        def submit():
            try:
                outcomes.append(body.run(ramp(ELBOW, q, 0.2)))
            except ResourceBusy as exc:
                outcomes.append(exc)

        threads = [threading.Thread(target=submit) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sum(not isinstance(o, Exception) for o in outcomes) == 1


def test_close_during_base_ramp_keeps_the_decided_stop():
    body = SimBody(clock="manual")
    table = pd.DataFrame({"t": [0.5, 4.0, 4.5], "base.vx": [0.2, 0.2, 0.0]})
    handle = body.run(table, guard=Signal("base.vx", ">", 0.15))
    body.advance(0.45)  # guard fired; ramp-down not finished yet
    assert not handle.done()
    body.close()
    result = handle.result(0)
    assert result.status is Status.GUARDED
    assert result.trigger["signal"] == "base.vx"
    assert "stop not completed" in result.reason


def test_recording_samples_frames_at_fixed_simulated_spacing():
    with SimBody(clock="manual") as body:
        assert body.views()[:3] == ("overview", "right_hand", "left_hand")
        recording = body.record("overview", fps=10, width=160, height=120)
        body.advance(1.0)
        frames = recording.stop()
        body.advance(0.5)
        assert len(recording.frames()) == len(frames) == 11  # t0 .. t0 + 1.0
        times = np.array([f.sim_time for f in frames])
        np.testing.assert_allclose(np.diff(times), 0.1, atol=1e-9)
        assert frames[0].rgb.shape == (120, 160, 3)
        capped = body.record("overview", fps=50, width=160, height=120, max_frames=3)
        body.advance(0.5)
        assert len(capped.frames()) == 3 and capped.truncated


def test_reached_waits_for_the_final_row_of_an_out_and_back_path(body):
    wrist = "right_arm.joint7"
    q = commanded(body)[wrist]
    table = pd.DataFrame({"t": [0.6, 1.2], wrist: [q + 0.25, q]})
    handle = body.run(table, until=Reached(rad=0.06, for_s=0.02))
    body.advance(1.6)
    result = handle.result(0)
    assert result.status is Status.CONDITION_MET
    assert result.rows_commanded == result.rows_total == 2
    assert result.table_time_commanded == pytest.approx(1.2)
    sent = result.log()[("commanded", wrist)]
    assert sent.max() == pytest.approx(q + 0.25)  # the "out" knot was executed
    assert result.trigger["sim_time"] >= result.start_time + 1.2 - 1e-9


def test_position_bounds_are_strict_and_exact_bound_runs_without_fault(body):
    channel = "right_hand.index_finger.mcp_abd"
    high = body.channels().loc[channel, "high"]
    q = commanded(body)[channel]
    tick = body.state().tick
    with pytest.raises(TableError, match="outside"):
        body.run(pd.DataFrame({"t": [1.0], channel: [high + 5e-10]}))
    assert body.state().tick == tick and body.state().health.active_motions == ()
    handle = body.run(pd.DataFrame({"t": [1.0], channel: [high]}))
    body.advance(1.1)
    assert handle.result(0).status is Status.COMPLETED
    assert body.state().joints.commanded[channel] == high
    assert not body.state().health.faulted
    body.run(pd.DataFrame({"t": [1.0], channel: [q]}))
    body.advance(1.1)


def test_overlapping_manual_advance_is_rejected_and_close_releases_waiter():
    body = SimBody(clock="manual")
    start = body.state().sim_time
    outcome = {}

    def long_advance():
        try:
            outcome["state"] = body.advance(8.0)
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            outcome["error"] = exc

    worker = threading.Thread(target=long_advance)
    worker.start()
    deadline = time.monotonic() + 10
    while body.state().sim_time - start < 0.05 and time.monotonic() < deadline:
        time.sleep(0.005)
    with pytest.raises(BodyError, match="already in progress"):
        body.advance(0.1)
    progressed = body.state().sim_time
    time.sleep(0.05)
    assert body.state().sim_time > progressed  # the first advance kept running
    body.close()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert isinstance(outcome.get("error"), BodyClosed)


def test_render_traffic_cannot_starve_physics_or_motion(monkeypatch):
    import robot_session.michael.body.sim as sim_module

    class SlowCamera:  # deterministic 20 ms render, longer than a 10 ms tick
        def __init__(self, env, width, height):
            self.shape = (height, width, 3)

        def set_view(self, view):
            pass

        def render(self):
            time.sleep(0.02)
            return np.zeros(self.shape, np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(sim_module, "Camera", SlowCamera)
    with SimBody(clock="realtime") as body:
        body.capture("overview", 16, 12)
        q = commanded(body)[ELBOW]
        frames, done = [], threading.Event()

        def client():
            for _ in range(10):
                frames.append(body.capture("overview", 16, 12))

        clients = [threading.Thread(target=client) for _ in range(8)]
        start = body.state()
        handle = body.run(ramp(ELBOW, q, 0.3, seconds=1.0))
        for thread in clients:
            thread.start()
        result = handle.result(timeout=10)
        captured_at_motion_end = len(frames)
        middle = body.state()
        for thread in clients:
            thread.join(timeout=30)
        done.set()
        end = body.state()
        assert result.status is Status.COMPLETED
        assert result.end_time - result.start_time == pytest.approx(1.0, abs=0.02)
        assert captured_at_motion_end < 80  # requests were still outstanding
        wall = middle.wall_time - start.wall_time
        assert middle.sim_time - start.sim_time >= 0.8 * wall
        assert len(frames) == 80
        times = sorted(f.sim_time for f in frames)
        assert times[-1] - times[0] > 1.0  # frames sample a moving simulation
        assert end.sim_time - start.sim_time >= 0.8 * (end.wall_time - start.wall_time)
        assert end.joints.measured[ELBOW] - start.joints.measured[ELBOW] > 0.25


def test_direct_native_base_command_follows_heading_after_turning():
    env = make_environment(control_freq=100)
    try:
        robot = env.robots[0]
        model, data = env.sim.model._model, env.sim.data._data
        base = model.body("mobilebase0_base").id
        hold = robot.create_action_vector({})

        def heading():
            r = data.xmat[base]
            return np.arctan2(r[3], r[0])

        turn = hold.copy()
        turn[56] = 0.5 / 1.5  # native yaw: 0.5 rad/s
        for _ in range(314):
            env.step(turn)
        for _ in range(50):
            env.step(hold)
        yaw = heading()
        assert yaw == pytest.approx(np.pi / 2, abs=0.05)
        start = data.xanchor[model.joint("mobilebase0_joint_mobile_yaw").id][:2].copy()
        forward = hold.copy()
        forward[54] = 0.2  # native forward: 0.2 m/s in the current base frame
        for _ in range(100):
            env.step(forward)
        for _ in range(50):
            env.step(hold)
        moved = data.xanchor[model.joint("mobilebase0_joint_mobile_yaw").id][:2] - start
        direction = np.array([np.cos(yaw), np.sin(yaw)])
        assert moved @ direction > 0.18
        assert abs(direction[0] * moved[1] - direction[1] * moved[0]) < 0.01
    finally:
        env.close()


def test_fault_after_physics_advanced_reports_last_confirmed_sample():
    holder = {}

    def factory(hz):
        holder["env"] = _FlakyEnv(make_environment(control_freq=hz))
        return holder["env"]

    wrist = "right_arm.joint7"
    body = SimBody(clock="manual", env_factory=factory)
    q = commanded(body)[wrist]
    handle = body.run(ramp(wrist, q, 0.5))
    before = body.advance(0.3)
    holder["env"].fail = "after"
    with pytest.raises(BodyFaulted):
        body.advance(0.01)
    result = handle.result(timeout=5)
    state = body.state()
    actual = holder["env"].after
    # Physics really moved on, but that state was never confirmed as a sample.
    assert actual["time"] > before.sim_time
    assert actual["q"] != before.joints.measured[wrist]
    assert state.physics_time == pytest.approx(actual["time"])
    assert state.sim_time == before.sim_time < state.physics_time
    assert state.joints.measured[wrist] == before.joints.measured[wrist]
    assert state.joints.commanded[wrist] == before.joints.commanded[wrist]
    assert result.status is Status.FAULTED
    assert result.measured_time == before.sim_time
    assert result.end_time == before.sim_time
    assert result.final_measured[wrist] == before.joints.measured[wrist]
    assert result.final_command[wrist] == before.joints.commanded[wrist]
    attempted = result.attempted_command[wrist]
    assert attempted > result.final_command[wrist]  # the unconfirmed next step
    assert result.log()[("commanded", wrist)].iloc[-1] == result.final_command[wrist]
    body.close()
