"""SimBody: the real Michael MuJoCo/robosuite environment behind one owner thread.

The owner thread creates, steps, reads, renders and closes the environment.
Callers talk to it through a request queue and read immutable snapshots; no
public attribute exposes a live MuJoCo or robosuite object.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

import mujoco
import numpy as np
import pandas as pd

from ..build import FINGERS
from ..runtime import Camera, make_environment
from .catalog import BASE_CHANNELS, Catalog, build_catalog
from .program import CompiledCondition, Program, compile_condition, compile_table
from .types import (
    BodyClosed,
    BodyError,
    BodyFaulted,
    BodyState,
    Condition,
    Envelope,
    Frame,
    Health,
    MotionHandle,
    MotionResult,
    Reached,
    ResourceBusy,
    Signal,
    Status,
    TableError,
    _events_frame,
    _frozen,
)

_EPS = 1e-9
_UNTIL_SLACK_S = 2.0  # default timeout_s = table duration + this, when `until` is set
_MAX_UNTIL_EXTRA_S = 60.0
_HAND_VIEWS = {"right_hand": "right", "left_hand": "left"}


class _PhysicsFault(RuntimeError):
    pass


@dataclass
class _Recording:
    view: str
    width: int
    height: int
    period: float
    max_frames: int
    next_time: float
    frames: list = field(default_factory=list)
    truncated: bool = False
    active: bool = True


class Recording:
    """Periodic owner-thread captures at fixed simulation-time spacing."""

    def __init__(self, body: SimBody, rid: int, record: _Recording):
        self._body, self._id, self._record = body, rid, record

    @property
    def truncated(self) -> bool:
        return self._record.truncated

    def frames(self) -> list[Frame]:
        return list(self._record.frames)

    def stop(self) -> list[Frame]:
        if self._record.active:
            try:
                self._body._call("record_stop", self._id)
            except (BodyClosed, BodyFaulted):
                pass
        return self.frames()


class _Motion:
    """Owner-thread state for one table run."""

    def __init__(
        self, ex, mid, program: Program, guards, until, timeout_s, stop_hold, handle
    ):
        cols = program.columns
        start = ex.cmd[cols].copy()
        rate = np.abs(program.values[0] - start) / program.times[0]
        bad = program.position & (rate > program.rates + _EPS)
        if bad.any():
            j = int(np.argmax(bad))
            raise TableError(
                f"{program.names[j]} would move from command {start[j]:.4g} at "
                f"{rate[j]:.3g}/s to its first row; envelope is {program.rates[j]:g}/s"
            )
        self.id, self.program, self.handle = mid, program, handle
        self.guards: list[CompiledCondition] = guards
        self.until: CompiledCondition | None = until
        self.timeout_s, self.stop_hold = timeout_s, stop_hold
        self.knot_t = np.concatenate([[0.0], program.times])
        self.knot_v = np.vstack([start, program.values])
        self.base = np.array([ex.catalog.channels[c].mode == "velocity" for c in cols])
        self.accel = np.array([ex.catalog.channels[c].accel for c in cols])
        self.phase = "run"  # run -> (hold) -> stopping -> released
        self.ticks = 0
        self.tau = 0.0
        self.start_time = ex.sim_time
        self.stop_time = None
        self.status = None
        self.reason = ""
        self.trigger = None
        self.events = []
        self.reached_tol = (
            until if until is not None and until.kind == "reached" else None
        )
        sig = sorted({i for c in [*guards, until] if c is not None for i in c.indices})
        self.sig_idx = np.array(sig, dtype=int)
        horizon = timeout_s if timeout_s is not None else program.duration
        ramp = max(
            [
                abs(ex.catalog.channels[c].limit / ex.catalog.channels[c].accel)
                for c in cols[self.base]
            ]
            or [0.0]
        )
        capacity = math.ceil((horizon + ramp) / ex.dt) + 8
        k = len(cols)
        self.log_time = np.empty(capacity)
        self.log_tau = np.empty(capacity)
        self.log_phase = []
        self.log_cmd = np.empty((capacity, k))
        self.log_meas = np.empty((capacity, k))
        self.log_sig = np.empty((capacity, len(self.sig_idx)))
        self.log_n = 0
        self.truncated = False

    # Called before env.step: write this motion's commands.
    def command(self, ex):
        cols = self.program.columns
        if self.phase == "run":
            tau = min((self.ticks + 1) * ex.dt, self.program.duration)
            i = min(
                int(np.searchsorted(self.knot_t, tau, "right")) - 1,
                len(self.knot_t) - 2,
            )
            a = (tau - self.knot_t[i]) / (self.knot_t[i + 1] - self.knot_t[i])
            ex.cmd[cols] = self.knot_v[i] + a * (self.knot_v[i + 1] - self.knot_v[i])
            self.tau = tau
        elif self.phase == "stopping" and self.base.any():
            idx = cols[self.base]
            v = ex.cmd[idx]
            ex.cmd[idx] = np.sign(v) * np.maximum(
                np.abs(v) - self.accel[self.base] * ex.dt, 0.0
            )

    # Called after env.step and measurement.
    def observe(self, ex):
        if self.phase == "run":
            self.ticks += 1
        self._log(ex)
        if self.phase in ("run", "hold"):
            for cond in self.guards:
                fired, name, value = cond.check(ex.signals)
                if fired:
                    self.stop(
                        ex, Status.GUARDED, f"guard {cond.label}", cond, name, value
                    )
                    return
            at_end = self.tau >= self.program.duration - _EPS
            # Reached() judges the final row, so it waits until that row has
            # been commanded; Signal/touch conditions may stop mid-table.
            if self.until is not None and (at_end or self.until.kind != "reached"):
                fired, name, value = self.until.check(ex.signals)
                if fired:
                    self.stop(
                        ex,
                        Status.CONDITION_MET,
                        f"until {self.until.label}",
                        self.until,
                        name,
                        value,
                    )
                    return
            if self.phase == "run" and self.tau >= self.program.duration - _EPS:
                if self.until is None:
                    self.status, self.reason, self.stop_time = (
                        Status.COMPLETED,
                        "all table rows commanded",
                        ex.sim_time,
                    )
                    self.finalize(ex)
                    return
                self.phase = "hold"
                ex.event(
                    "table_end", self, "holding final row until condition or timeout"
                )
            if (
                self.timeout_s is not None
                and ex.sim_time - self.start_time >= self.timeout_s - _EPS
            ):
                self.stop(
                    ex, Status.TIMED_OUT, f"until not met within {self.timeout_s:g} s"
                )
        elif self.phase == "stopping" and not np.any(
            ex.sent[self.program.columns[self.base]]
        ):
            self.finalize(ex)  # the hold (and zero base) has now been sent

    def _log(self, ex):
        n = self.log_n
        if n >= len(self.log_time):
            self.truncated = True
            return
        cols = self.program.columns
        self.log_time[n] = ex.sim_time
        self.log_tau[n] = self.tau
        self.log_phase.append(self.phase)
        self.log_cmd[n] = ex.sent[cols]
        self.log_meas[n] = ex.meas[cols]
        self.log_sig[n] = ex.signals[self.sig_idx]
        self.log_n = n + 1

    def stop(self, ex, status, reason, cond=None, name=None, value=None):
        """Controlled stop: positions hold (commanded or measured); base ramps."""
        if self.phase in ("stopping", "released"):
            return
        self.status, self.reason, self.stop_time = status, reason, ex.sim_time
        if cond is not None:
            self.trigger = {
                "condition": cond.label,
                "signal": name,
                "value": value,
                "sim_time": ex.sim_time,
            }
        ex.event("stop", self, f"{status.value}: {reason}")
        if self.stop_hold == "measured":
            pos = self.program.columns[self.program.position]
            low = ex.low[pos]
            high = ex.high[pos]
            ex.cmd[pos] = np.clip(ex.meas[pos], low, high)
        self.phase = "stopping"
        # A commanded hold with no base velocity is already what was last sent.
        # A measured hold or base ramp resolves after a tick has sent it.
        if self.stop_hold == "commanded" and not np.any(
            ex.sent[self.program.columns[self.base]]
        ):
            self.finalize(ex)

    def finalize(self, ex, status=None, reason=None):
        """Release leases and resolve the handle. ``status`` forces an outcome
        (fault/close); a close keeps a stop that was already decided."""
        if self.phase == "released":
            return
        if status is Status.CANCELLED and self.status is not None:
            self.reason = f"{self.reason}; stop not completed: {reason}"
        elif status is not None:
            self.status, self.reason = status, reason
            self.stop_time = (
                self.stop_time if self.stop_time is not None else ex.sim_time
            )
        self.phase = "released"
        cols = self.program.columns
        for c in cols:
            if ex.owner[c] == self.id:
                ex.owner[c] = -1
        for r in self.program.resources:
            if ex.leases.get(r) == self.id:
                del ex.leases[r]
        ex.motions.pop(self.id, None)
        pos = self.program.position
        target = self.program.values[-1]
        attained = None
        if pos.any():
            tol = (
                self.reached_tol.tolerance
                if self.reached_tol is not None
                else np.array(
                    [
                        Reached().rad
                        if ex.catalog.channels[c].unit == "rad"
                        else Reached().m
                        for c in cols[pos]
                    ]
                )
            )
            attained = bool(np.all(np.abs(ex.meas[cols[pos]] - target[pos]) <= tol))
        ex.event("released", self, self.status.value)
        n = self.log_n
        result = MotionResult(
            motion_id=self.id,
            status=self.status,
            reason=self.reason,
            trigger=self.trigger,
            resources=self.program.resources,
            channels=self.program.names,
            units=tuple(ex.catalog.channels[c].unit for c in cols),
            stop_hold=self.stop_hold,
            start_time=self.start_time,
            stop_time=float(self.stop_time),
            end_time=ex.sim_time,
            measured_time=ex.measured_time,
            table_duration=self.program.duration,
            table_time_commanded=float(self.tau),
            rows_total=len(self.program.times),
            rows_commanded=int(np.count_nonzero(self.program.times <= self.tau + _EPS)),
            attained=attained,
            log_truncated=self.truncated,
            _final_target=_frozen(target),
            _final_command=_frozen(ex.sent[cols]),
            _final_measured=_frozen(ex.meas[cols]),
            _attempted_command=_frozen(ex.attempt[cols]) if ex.attempting else None,
            _position_mask=_frozen(pos),
            _log_time=_frozen(self.log_time[:n]),
            _log_tau=_frozen(self.log_tau[:n]),
            _log_phase=tuple(self.log_phase[:n]),
            _log_cmd=_frozen(self.log_cmd[:n]),
            _log_meas=_frozen(self.log_meas[:n]),
            _log_signal_names=tuple(ex.catalog.signals[i] for i in self.sig_idx),
            _log_signals=_frozen(self.log_sig[:n]),
            _events=tuple(self.events),
        )
        self.handle._resolve(result)


class _Executor:
    """Everything that touches MuJoCo. Lives only on the owner thread."""

    def __init__(self, body: SimBody, factory):
        self.body = body
        self.env = factory(body._control_hz)
        try:
            self._setup()
        except BaseException:
            self.env.close()
            raise

    def _setup(self):
        body, env = self.body, self.env
        # The Body reads MuJoCo state directly; robosuite camera observables
        # would otherwise render on every control step.
        for name, observable in list(env._observables.items()):
            if observable.modality == "image":
                env.modify_observable(name, "enabled", False)
        self.robot = env.robots[0]
        self.model, self.data = env.sim.model._model, env.sim.data._data
        self.dt = float(env.control_timestep)
        self.physics_dt = float(env.model_timestep)
        substeps = self.dt / self.physics_dt
        if abs(substeps - round(substeps)) > 1e-6 or round(substeps) < 1:
            raise ValueError(
                f"control_hz {body._control_hz} is not an integer divisor of physics rate {1 / self.physics_dt:g} Hz"
            )
        self.catalog: Catalog = build_catalog(env, body._envelope)
        cat = self.catalog
        if [c.native_index for c in cat.channels] != list(range(cat.action_dim)):
            raise RuntimeError("Catalogue order must match the native action order")
        modes = np.array([c.mode for c in cat.channels])
        self.pos = np.flatnonzero(modes == "position")
        self.base_idx = np.array([cat.index[n] for n in BASE_CHANNELS])
        self.low = np.array([c.low for c in cat.channels])
        self.high = np.array([c.high for c in cat.channels])
        self.scale = np.array([c.native_scale for c in cat.channels])
        self.bias = np.array([c.native_bias for c in cat.channels])
        composite = self.robot.composite_controller
        qpos, qvel = np.zeros(cat.action_dim, int), np.zeros(cat.action_dim, int)
        for part, controller in composite.part_controllers.items():
            start, end = composite._action_split_indexes[part]
            if part != "base":
                qpos[start:end] = controller.qpos_index
                qvel[start:end] = controller.qvel_index
        self.qpos_idx, self.qvel_idx = qpos[self.pos], qvel[self.pos]
        self.base_body = self.model.body(
            self.robot.robot_model.base.correct_naming("base")
        ).id
        self.yaw_joint = self.model.joint(cat.channels[cat.index["base.wz"]].joint).id
        if self.model.jnt_bodyid[self.yaw_joint] != self.base_body:
            raise RuntimeError("Expected the base yaw joint on the base body")
        self.yaw_anchor = self.model.jnt_pos[self.yaw_joint].copy()
        body._views = (
            "overview",
            *_HAND_VIEWS,
            *(self.model.camera(i).name for i in range(self.model.ncam)),
        )
        self.touch_idx, self.force_idx, self.torque_idx = [], [], []
        for side in ("right", "left"):
            gripper = self.robot.gripper[side]
            for finger in FINGERS:
                for kind, out in (
                    ("touch", self.touch_idx),
                    ("force", self.force_idx),
                    ("torque", self.torque_idx),
                ):
                    sid = self.model.sensor(
                        gripper.correct_naming(f"{finger}_{kind}")
                    ).id
                    adr, dim = self.model.sensor_adr[sid], self.model.sensor_dim[sid]
                    out.append(np.arange(adr, adr + dim) if dim > 1 else adr)
        self.touch_idx = np.array(self.touch_idx)
        self.force_idx, self.torque_idx = (
            np.array(self.force_idx),
            np.array(self.torque_idx),
        )

        native = self.robot.create_action_vector({})
        self.cmd = native * self.scale + self.bias
        self.cmd[self.base_idx] = 0.0
        # cmd is the next planned command; sent is what the last env.step used.
        self.sent = self.cmd.copy()
        # A failing step may or may not have applied its command and may have
        # advanced physics partially: keep it as "attempted", never as "sent".
        self.attempt = self.cmd.copy()
        self.attempting = False
        self.measured_time = float(self.data.time)
        self.owner = np.full(cat.action_dim, -1)
        self.leases: dict[str, int] = {}
        self.motions: dict[int, _Motion] = {}
        self.next_id = 1
        self.meas = np.zeros(cat.action_dim)
        self.vel = np.zeros(cat.action_dim)
        self.signals = np.zeros(len(cat.signals))
        self.touch = np.zeros(len(cat.tactile))
        self.force = np.zeros((len(cat.tactile), 3))
        self.torque = np.zeros((len(cat.tactile), 3))
        self.pose = np.zeros(3)
        self._objvel = np.zeros(6)
        self.warnings0 = self._warnings()

        self.ticks = 0
        self.late_ticks = 0
        self.dropped = 0.0
        self.max_tick = 0.0
        self.mean_tick = 0.0
        self.wall0 = time.monotonic()
        self.window = (self.wall0, self.sim_time)
        self.ratio = float("nan")
        self.last_overrun_event = -math.inf
        self.faulted, self.error, self.running = False, None, True
        self.budget, self.advance_future = 0, None
        self.deferred: deque = deque()
        self.cameras: dict[tuple[int, int], Camera] = {}
        self.recordings: dict[int, _Recording] = {}
        self.static = (
            tuple(cat.channels[i].name for i in self.pos),
            tuple(cat.channels[i].resource for i in self.pos),
            tuple(cat.channels[i].unit for i in self.pos),
            _frozen(self.low[self.pos]),
            _frozen(self.high[self.pos]),
        )
        n = max(1, round(body._telemetry_s / self.dt))
        body._ring = {
            "time": np.full(n, np.nan),
            "wall": np.full(n, np.nan),
            "tick_s": np.full(n, np.nan),
            "cmd": np.full((n, cat.action_dim), np.nan),
            "meas": np.full((n, cat.action_dim), np.nan),
            "touch": np.full((n, len(cat.tactile)), np.nan),
        }
        body._ring_count = 0
        self._measure()
        self.event("started", None, f"{body._clock} clock at {1 / self.dt:g} Hz")
        self._publish()

    # --- physics -------------------------------------------------------------

    @property
    def sim_time(self) -> float:
        """Time of the last confirmed measurement sample."""
        return self.measured_time

    def physics_time(self) -> float:
        """Raw simulator clock; after a failed step it may be ahead of sim_time."""
        t = float(self.data.time)
        return t if math.isfinite(t) else math.nan

    def _warnings(self) -> int:
        return int(sum(w.number for w in self.data.warning))

    def _base_native(self, v: np.ndarray) -> np.ndarray:
        """Current-base-frame (vx, vy, wz) -> normalized native base action.
        The frame rotation itself lives in models.MichaelBaseVelocityController."""
        native = (v - self.bias[self.base_idx]) / self.scale[self.base_idx]
        return np.clip(native, -1.0, 1.0)

    def _measure(self):
        d = self.data
        self.meas[self.pos] = d.qpos[self.qpos_idx]
        self.vel[self.pos] = d.qvel[self.qvel_idx]
        # Base velocity at the yaw axis (the point the slide joints move), in
        # the current base frame: v_origin + omega x r_anchor.
        mujoco.mj_objectVelocity(
            self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.base_body, self._objvel, 1
        )
        v = self._objvel[3:] + np.cross(self._objvel[:3], self.yaw_anchor)
        self.meas[self.base_idx] = (v[0], v[1], self._objvel[2])
        self.vel[self.base_idx] = self.meas[self.base_idx]
        self.touch[:] = d.sensordata[self.touch_idx]
        self.force[:] = d.sensordata[self.force_idx]
        self.torque[:] = d.sensordata[self.torque_idx]
        r = d.xmat[self.base_body]
        anchor = d.xanchor[self.yaw_joint]
        self.pose[:] = (anchor[0], anchor[1], math.atan2(r[3], r[0]))
        n = len(self.meas)
        self.signals[:n] = self.meas
        self.signals[n:] = self.touch
        self.measured_time = float(d.time)  # stamp only a complete sample

    def tick(self):
        start = time.perf_counter()
        for motion in list(self.motions.values()):
            motion.command(self)
        unowned = self.base_idx[self.owner[self.base_idx] < 0]
        self.cmd[unowned] = 0.0  # base velocity never persists without an owner
        # Tables are validated strictly against these bounds; this only removes
        # interpolation roundoff that the native controller would reject.
        self.cmd[self.pos] = np.clip(
            self.cmd[self.pos], self.low[self.pos], self.high[self.pos]
        )
        self.attempt[:] = self.cmd
        self.attempting = True
        native = self.cmd.copy()
        native[self.base_idx] = self._base_native(self.cmd[self.base_idx])
        self.env.step(native)
        self.ticks += 1
        d = self.data
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            raise _PhysicsFault("non-finite MuJoCo state")
        if self._warnings() != self.warnings0:
            raise _PhysicsFault(
                f"MuJoCo physics warning(s): {self._warnings() - self.warnings0}"
            )
        # Only now is the step confirmed: its command was sent and its state
        # becomes the new measurement sample.
        self.sent[:] = self.attempt
        self.attempting = False
        self._measure()
        for motion in list(self.motions.values()):
            motion.observe(self)
        for record in self.recordings.values():
            if record.active and self.sim_time >= record.next_time - _EPS:
                if len(record.frames) >= record.max_frames:
                    record.truncated, record.active = True, False
                else:
                    try:
                        frame = self.render(record.view, record.width, record.height)
                    except Exception as exc:  # noqa: BLE001 - stop recording, not physics
                        record.active = False
                        self.event("recording_error", None, repr(exc))
                    else:
                        record.frames.append(frame)
                        record.next_time += record.period
        elapsed = time.perf_counter() - start
        self.max_tick = max(self.max_tick, elapsed)
        self.mean_tick += (elapsed - self.mean_tick) * (
            1.0 if self.ticks == 1 else 0.02
        )
        wall = time.monotonic()
        if wall - self.window[0] >= 1.0:
            self.ratio = (self.sim_time - self.window[1]) / (wall - self.window[0])
            self.window = (wall, self.sim_time)
        body = self.body
        with body._ring_lock:
            i = body._ring_count % len(body._ring["time"])
            ring = body._ring
            ring["time"][i], ring["wall"][i], ring["tick_s"][i] = (
                self.sim_time,
                wall,
                elapsed,
            )
            ring["cmd"][i], ring["meas"][i], ring["touch"][i] = (
                self.sent,
                self.meas,
                self.touch,
            )
            body._ring_count += 1
        self._publish()

    def _publish(self):
        names, resources, units, low, high = self.static
        cat = self.catalog
        base_owner = int(self.owner[self.base_idx[0]])
        health = Health(
            clock=self.body._clock,
            control_hz=1.0 / self.dt,
            physics_dt=self.physics_dt,
            running=self.running,
            faulted=self.faulted,
            error=self.error,
            ticks=self.ticks,
            sim_time=self.sim_time,
            wall_elapsed=time.monotonic() - self.wall0,
            realtime_ratio=self.ratio,
            late_ticks=self.late_ticks,
            dropped_s=self.dropped,
            max_tick_s=self.max_tick,
            mean_tick_s=self.mean_tick,
            physics_warnings=self._warnings() - self.warnings0,
            active_motions=tuple(sorted(self.motions)),
        )
        self.body._latest = BodyState(
            sim_time=self.sim_time,
            physics_time=self.physics_time(),
            wall_time=time.monotonic(),
            tick=self.ticks,
            health=health,
            _names=names,
            _resources=resources,
            _units=units,
            _low=low,
            _high=high,
            _commanded=_frozen(self.sent[self.pos]),
            _measured=_frozen(self.meas[self.pos]),
            _velocity=_frozen(self.vel[self.pos]),
            _owner=_frozen(self.owner[self.pos]),
            _base_names=BASE_CHANNELS,
            _base_units=tuple(cat.channels[i].unit for i in self.base_idx),
            _base_commanded=_frozen(self.sent[self.base_idx]),
            _base_measured=_frozen(self.meas[self.base_idx]),
            _base_owner=base_owner,
            _base_pose=_frozen(self.pose),
            _tactile_names=cat.tactile,
            _touch=_frozen(self.touch),
            _force=_frozen(self.force),
            _torque=_frozen(self.torque),
        )

    def event(self, kind, motion, detail):
        row = (
            self.sim_time,
            time.monotonic(),
            kind,
            None if motion is None else motion.id,
            detail,
        )
        if motion is not None:
            motion.events.append(row)
        with self.body._events_lock:
            self.body._events.append(row)

    def render(self, view, width, height) -> Frame:
        camera = self.cameras.get((width, height))
        if camera is None:
            camera = self.cameras[(width, height)] = Camera(self.env, width, height)
        if view == "overview" or view in _HAND_VIEWS:
            camera.set_view(_HAND_VIEWS.get(view, "overview"))
            rgb = camera.render()
        else:
            # Visual geoms only (robosuite's collision geoms are group 0); no
            # spectator cutaway for the robot's own cameras.
            camera.renderer.update_scene(
                self.data, camera=view, scene_option=camera.options
            )
            rgb = camera.renderer.render().copy()
        return Frame(rgb=_frozen(rgb), sim_time=self.sim_time, view=view)

    # --- request handling ----------------------------------------------------

    def handle(self, request) -> bool:
        kind, payload, future = request
        if kind == "close":
            future.set_result(None)
            return True
        try:
            result = getattr(self, f"_req_{kind}")(payload, future)
            if result is not _PENDING:
                future.set_result(result)
        except Exception as exc:  # noqa: BLE001 - reported to the caller, executor continues
            if not future.done():
                future.set_exception(exc)
        return False

    def _req_run(self, payload, _future):
        program, conditions, timeout_s, stop_hold = payload
        busy = sorted(r for r in program.resources if r in self.leases)
        if busy:
            raise ResourceBusy(
                f"{busy} leased by motion(s) {sorted({self.leases[r] for r in busy})}; "
                "v0 rejects conflicts (no preempt/append)"
            )
        guards = [
            compile_condition(c, self.catalog, program, self.dt) for c in conditions[0]
        ]
        until = (
            None
            if conditions[1] is None
            else compile_condition(conditions[1], self.catalog, program, self.dt)
        )
        mid = self.next_id
        handle = MotionHandle(mid, program.resources, program.names, self.body._cancel)
        motion = _Motion(
            self, mid, program, guards, until, timeout_s, stop_hold, handle
        )
        self.next_id += 1
        self.motions[mid] = motion
        self.owner[program.columns] = mid
        for r in program.resources:
            self.leases[r] = mid
        self.event(
            "accepted",
            motion,
            f"{sorted(program.resources)} for {program.duration:g} s",
        )
        self._publish()
        return handle

    def _req_cancel(self, mid, _future):
        motion = self.motions.get(mid)
        if motion is not None:
            motion.stop(self, Status.CANCELLED, "cancel requested")

    def _req_cancel_all(self, _payload, _future):
        handles = [m.handle for m in self.motions.values()]
        for motion in list(self.motions.values()):
            motion.stop(self, Status.CANCELLED, "cancel_all requested")
        return handles

    def _req_capture(self, payload, _future):
        return self.render(*payload)

    def _req_advance(self, ticks, future):
        if self.body._clock != "manual":
            raise BodyError("advance() is only available with clock='manual'")
        if self.budget:
            raise BodyError("another advance() is already in progress")
        self.budget, self.advance_future = ticks, future
        return _PENDING

    def _req_record_start(self, payload, _future):
        rid = self.next_id
        self.next_id += 1
        view, fps, width, height, max_frames = payload
        first = self.render(view, width, height)  # also validates the view
        period = 1.0 / fps
        record = _Recording(
            view, width, height, period, max_frames, self.sim_time + period, [first]
        )
        self.recordings[rid] = record
        return rid, record

    def _req_record_stop(self, rid, _future):
        record = self.recordings.pop(rid, None)
        if record is not None:
            record.active = False

    def _req_probe(self, payload, _future):
        if not hasattr(self, "_probe"):
            names = [self.model.body(i).name for i in range(self.model.nbody)]
            if "tactile_probe" not in names:
                raise BodyError(
                    "The calibration probe exists only in the commissioning room"
                )
            self._probe = self.model.body("tactile_probe").mocapid[0]
        if payload is None:
            self.data.mocap_pos[self._probe] = [2, 0, 1]
            self.event("sim_fixture", None, "probe removed")
            return
        hand, finger, gap = payload
        side = _HAND_VIEWS[hand]
        site = self.model.site(
            self.robot.gripper[side].correct_naming(f"{finger}_tactile_site")
        ).id
        outward = 1 if side == "right" else -1
        offset = outward * (self.model.site_size[site, 1] + 0.008 + gap)
        self.data.mocap_pos[self._probe] = self.data.site_xpos[
            site
        ] + self.data.site_xmat[site].reshape(3, 3) @ np.array([0, offset, 0])
        self.event("sim_fixture", None, f"probe at {hand}.{finger}, gap {gap:g} m")
        return

    # --- loops and shutdown --------------------------------------------------

    def loop(self):
        if self.body._clock == "manual":
            return self._loop_manual()
        q, period, cap = self.body._requests, self.dt, self.body._max_catchup_s
        due = time.monotonic()
        while True:
            # Control requests are cheap and always served. Observation work
            # (rendering) runs only while ahead of the next tick, so one render
            # can delay at most one tick and renders never starve physics.
            while True:
                if self._take_control(q):
                    return
                wait = due - time.monotonic()
                if wait <= 0:
                    break
                if self.deferred:
                    self.handle(self.deferred.popleft())
                    continue
                try:
                    request = q.get(timeout=wait)
                except queue.Empty:
                    break
                if request[0] in _OBSERVATION:
                    self.deferred.append(request)
                elif self.handle(request):
                    return
            late = time.monotonic() - due
            if late > period:
                self.late_ticks += 1
            if late > cap:
                # Do not burst to catch up: simulated time falls behind instead.
                self.dropped += late
                due = time.monotonic()
                wall = time.monotonic()
                if wall - self.last_overrun_event > 1.0:
                    self.last_overrun_event = wall
                    self.event(
                        "overrun", None, f"{late * 1000:.0f} ms late; not caught up"
                    )
            self.tick()
            due += period

    def _take_control(self, q) -> bool:
        """Handle queued control requests now; defer observation requests."""
        while True:
            try:
                request = q.get_nowait()
            except queue.Empty:
                return False
            if request[0] in _OBSERVATION:
                self.deferred.append(request)
            elif self.handle(request):
                return True

    def _loop_manual(self):
        q = self.body._requests
        while True:
            if self.budget == 0:
                if self.handle(q.get()):
                    return
                continue
            while True:
                try:
                    request = q.get_nowait()
                except queue.Empty:
                    break
                if self.handle(request):
                    return
            self.tick()
            self.budget -= 1
            if self.budget == 0:
                self.advance_future.set_result(self.body._latest)
                self.advance_future = None

    def fault(self, exc: BaseException):
        self.faulted = True
        self.error = f"{type(exc).__name__}: {exc}"
        self.event(
            "fault",
            None,
            f"{self.error}; physics clock {self.physics_time():.4f} s, "
            f"last confirmed sample {self.measured_time:.4f} s",
        )
        for motion in list(self.motions.values()):
            motion.finalize(
                self, Status.FAULTED, f"executor fault, physics stopped: {self.error}"
            )

    def shutdown(self):
        reason = "body closed; physics stopped" if not self.faulted else self.error
        for motion in list(self.motions.values()):
            motion.finalize(self, Status.CANCELLED, reason)
        for record in self.recordings.values():
            record.active = False
        if self.advance_future is not None and not self.advance_future.done():
            self.advance_future.set_exception(self.body._closed_error())
        while self.deferred:
            _, _, future = self.deferred.popleft()
            if not future.done():
                future.set_exception(self.body._closed_error())
        self.running = False
        self.event("closed", None, reason)
        self._publish()

    def release(self):
        for camera in self.cameras.values():
            camera.close()
        self.env.close()


_PENDING = object()
_OBSERVATION = frozenset({"capture", "record_start"})
_MAX_OBSERVATION_REQUESTS = 4


class SimBody:
    """Michael in simulation: real MichaelOpenArmWuji model, one owner thread.

    ``clock="realtime"`` advances physics continuously at ``control_hz`` whether
    or not the caller is doing anything; ``clock="manual"`` advances only inside
    ``advance(seconds)`` (debugging and deterministic tests).
    """

    def __init__(
        self,
        scene: str = "room",
        *,
        control_hz: int = 100,
        clock: str = "realtime",
        envelope: Envelope | None = None,
        base_kinematics: str = "differential",
        telemetry_s: float = 10.0,
        max_catchup_s: float = 0.05,
        env_factory: Callable[[int], object] | None = None,
        start_timeout_s: float = 300.0,
    ):
        if scene not in ("room", "kitchen"):
            raise ValueError("scene must be 'room' or 'kitchen'")
        if clock not in ("realtime", "manual"):
            raise ValueError("clock must be 'realtime' or 'manual'")
        if base_kinematics not in ("differential", "holonomic"):
            raise ValueError("base_kinematics must be 'differential' or 'holonomic'")
        if not (isinstance(control_hz, int) and 20 <= control_hz <= 500):
            raise ValueError("control_hz must be an integer from 20 to 500")
        if not (0 < telemetry_s <= 120 and 0 < max_catchup_s <= 1):
            raise ValueError("telemetry_s must be in (0, 120]; max_catchup_s in (0, 1]")
        self._scene, self._clock, self._control_hz = scene, clock, control_hz
        self._envelope = envelope or Envelope()
        self._base_kinematics = base_kinematics
        self._telemetry_s, self._max_catchup_s = telemetry_s, max_catchup_s
        self._requests: queue.SimpleQueue = queue.SimpleQueue()
        self._gate = threading.Lock()
        self._accepting = True
        self._fault: str | None = None
        self._latest: BodyState | None = None
        self._events: deque = deque(maxlen=2000)
        self._events_lock = threading.Lock()
        self._ring: dict | None = None
        self._ring_count = 0
        self._ring_lock = threading.Lock()
        self._dt = 1.0 / control_hz
        # Backpressure: at most this many render requests are outstanding;
        # further capture()/record() callers block instead of growing a queue.
        self._observe_slots = threading.BoundedSemaphore(_MAX_OBSERVATION_REQUESTS)
        factory = env_factory or (
            lambda hz: make_environment(
                kitchen=scene == "kitchen", control_freq=hz, camera_observations=False
            )
        )
        ready: Future = Future()
        self._thread = threading.Thread(
            target=self._owner,
            args=(ready, factory),
            name="michael-simbody",
            daemon=True,
        )
        self._thread.start()
        try:
            self._catalog: Catalog = ready.result(timeout=start_timeout_s)
        except BaseException:
            self.close()
            raise

    # --- owner thread ----------------------------------------------------------

    def _owner(self, ready: Future, factory):
        ex = None
        try:
            ex = _Executor(self, factory)
            ready.set_result(ex.catalog)
            ex.loop()
        except BaseException as exc:  # noqa: BLE001 - every failure must stop motion
            if ex is None:
                ready.set_exception(exc)
            else:
                ex.fault(exc)
                self._fault = ex.error
        finally:
            with self._gate:
                self._accepting = False
            try:
                if ex is not None:
                    ex.shutdown()
            finally:
                self._drain()
                if ex is not None:
                    ex.release()

    def _drain(self):
        while True:
            try:
                _, _, future = self._requests.get_nowait()
            except queue.Empty:
                return
            if not future.done():
                future.set_exception(self._closed_error())

    def _closed_error(self) -> BodyError:
        if self._fault:
            return BodyFaulted(f"body faulted: {self._fault}")
        return BodyClosed("body is closed")

    def _call(self, kind, payload=None):
        future: Future = Future()
        with self._gate:
            if not self._accepting:
                raise self._closed_error()
            self._requests.put((kind, payload, future))
        while True:
            try:
                result = future.result(timeout=0.5)
                return result
            except FutureTimeout:
                if not self._thread.is_alive() and not future.done():
                    raise self._closed_error() from None

    def _observe(self, kind, payload):
        while not self._observe_slots.acquire(timeout=0.5):
            if not self._accepting:
                raise self._closed_error()
        try:
            return self._call(kind, payload)
        finally:
            self._observe_slots.release()

    def _cancel(self, motion_id: int) -> None:
        self._call("cancel", motion_id)

    # --- Body protocol ---------------------------------------------------------

    def channels(self) -> pd.DataFrame:
        """Every commandable channel: resource, mode, unit, bounds, envelope."""
        return self._catalog.frame()

    def signals(self) -> tuple[str, ...]:
        """Names usable in Signal/touch conditions (measured values)."""
        return self._catalog.signals

    def state(self) -> BodyState:
        return self._latest

    def run(
        self,
        table: pd.DataFrame,
        *,
        guard: Condition | Sequence[Condition] = (),
        until: Condition | None = None,
        timeout_s: float | None = None,
        stop_hold: str = "commanded",
    ) -> MotionHandle:
        """Validate, compile and start a movement table. Returns once the owner
        thread has accepted it (or raises TableError/ResourceBusy/BodyClosed)."""
        program = compile_table(
            table, self._catalog, base_kinematics=self._base_kinematics
        )
        guards = [guard] if isinstance(guard, (Signal, Reached)) else list(guard)
        if stop_hold not in ("commanded", "measured"):
            raise TableError("stop_hold must be 'commanded' or 'measured'")
        if until is None:
            if timeout_s is not None:
                raise TableError(
                    "timeout_s applies only with `until`; a table without it ends at its last row"
                )
        else:
            timeout_s = (
                program.duration + _UNTIL_SLACK_S
                if timeout_s is None
                else float(timeout_s)
            )
            if not (
                program.duration <= timeout_s <= program.duration + _MAX_UNTIL_EXTRA_S
            ):
                raise TableError(
                    f"timeout_s must be within [{program.duration:g}, {program.duration + _MAX_UNTIL_EXTRA_S:g}] s"
                )
        # Compile once here so errors surface before anything is sent.
        for condition in [*guards, until]:
            if condition is not None:
                compile_condition(condition, self._catalog, program, self._dt)
        return self._call("run", (program, (guards, until), timeout_s, stop_hold))

    def cancel_all(
        self, wait: bool = True, timeout: float | None = 5.0
    ) -> list[MotionResult] | None:
        """Stop every active motion (same semantics as MotionHandle.cancel)."""
        handles = self._call("cancel_all")
        return [h.result(timeout) for h in handles] if wait else None

    def events(self) -> pd.DataFrame:
        with self._events_lock:
            rows = list(self._events)
        return _events_frame(rows)

    def telemetry(self, last_s: float | None = None) -> pd.DataFrame:
        """Recent per-tick commanded/measured channels and pad touch (bounded ring)."""
        with self._ring_lock:
            ring = {k: v.copy() for k, v in self._ring.items()}
            count = self._ring_count
        size = len(ring["time"])
        order = np.arange(count - min(count, size), count) % size
        names = list(self._catalog.names)
        frame = pd.concat(
            {
                "commanded": pd.DataFrame(ring["cmd"][order], columns=names),
                "measured": pd.DataFrame(ring["meas"][order], columns=names),
                "touch_N": pd.DataFrame(
                    ring["touch"][order], columns=list(self._catalog.tactile)
                ),
                "executor": pd.DataFrame(
                    {"wall_time": ring["wall"][order], "tick_s": ring["tick_s"][order]}
                ),
            },
            axis=1,
        )
        frame.index = pd.Index(ring["time"][order], name="sim_time")
        if last_s is not None and len(frame):
            frame = frame[frame.index >= frame.index[-1] - last_s]
        frame.attrs["units"] = {c.name: c.unit for c in self._catalog.channels}
        return frame

    def views(self) -> tuple[str, ...]:
        """Names accepted by capture()/record(): spectator views, then MJCF cameras."""
        return self._views

    def capture(
        self, view: str = "overview", width: int = 640, height: int = 480
    ) -> Frame:
        """Render on the owner thread. Views: 'overview', 'right_hand',
        'left_hand', or an MJCF camera name. Costs real time in realtime mode."""
        return self._observe("capture", (view, int(width), int(height)))

    def record(
        self,
        view: str = "overview",
        fps: float = 10.0,
        width: int = 640,
        height: int = 480,
        max_frames: int = 600,
        max_bytes: int = 1 << 30,
    ) -> Recording:
        """Capture every 1/fps simulated seconds until stop() or max_frames.
        Frames stay in memory (640x480 RGB is about 0.9 MB each); recording
        stops with ``truncated`` set when the frame or byte budget is used."""
        if not (0 < fps <= 60 and 0 < width <= 4096 and 0 < height <= 4096):
            raise ValueError("fps must be in (0, 60]; width/height in (0, 4096]")
        max_frames = min(int(max_frames), int(max_bytes) // (width * height * 3))
        if max_frames < 1:
            raise ValueError("max_frames/max_bytes allow no frames")
        rid, record = self._observe(
            "record_start", (view, float(fps), int(width), int(height), int(max_frames))
        )
        return Recording(self, rid, record)

    def advance(self, seconds: float) -> BodyState:
        """Manual clock only: step ``seconds`` of simulation, then return state."""
        ticks = round(float(seconds) / self._dt)
        if ticks < 1:
            raise ValueError(
                f"advance needs at least one control period ({self._dt:g} s)"
            )
        return self._call("advance", ticks)

    def close(self, timeout: float = 30.0) -> None:
        """Cancel every motion, stop physics and release the environment."""
        with self._gate:
            if self._accepting:
                self._accepting = False
                self._requests.put(("close", None, Future()))
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise BodyError("owner thread did not stop")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- simulation-only fixtures (not part of the Body protocol) -------------

    def sim_place_probe(
        self, hand: str = "right_hand", finger: str = "index_finger", gap_m: float = 0.0
    ) -> None:
        """Commissioning room only: teleport the stationary 8 mm calibration
        sphere just outside a fingertip pad. A test fixture, not a task object."""
        if hand not in _HAND_VIEWS or finger not in FINGERS or not (0 <= gap_m <= 0.05):
            raise ValueError(
                f"hand in {sorted(_HAND_VIEWS)}, finger in {FINGERS}, 0 <= gap_m <= 0.05"
            )
        self._call("probe", (hand, finger, float(gap_m)))

    def sim_remove_probe(self) -> None:
        self._call("probe", None)
