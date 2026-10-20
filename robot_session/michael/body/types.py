"""Public value types for Michael's Body interface.

Nothing here holds a simulator object. States and results keep private numpy
copies and build a fresh pandas object on every access, so a caller can never
mutate live commands or measurements through a retained DataFrame.
"""

from __future__ import annotations

import enum
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


class BodyError(RuntimeError):
    """Base class for Body runtime errors."""


class BodyClosed(BodyError):
    """The body was closed; it accepts no further requests."""


class BodyFaulted(BodyError):
    """The executor stopped after a fault; physics is no longer advancing."""


class ResourceBusy(BodyError):
    """A requested resource is leased by an active motion (v0 rejects conflicts)."""


class TableError(ValueError):
    """A movement table or condition failed validation; nothing was executed."""


class Status(str, enum.Enum):
    """Terminal motion status. Only CONDITION_MET reflects a measured predicate."""

    COMPLETED = "completed"  # every table row was commanded; not a success claim
    CONDITION_MET = "condition_met"  # the caller's `until` held on measurements
    GUARDED = "guarded"  # a guard held on measurements; motion stopped early
    TIMED_OUT = "timed_out"  # `until` did not hold before `timeout_s`
    CANCELLED = "cancelled"  # handle.cancel(), cancel_all() or body close
    FAULTED = "faulted"  # executor/physics fault; physics stopped advancing


@dataclass(frozen=True)
class Envelope:
    """Software motion envelope enforced at submission. Simulation defaults only;
    none of these values is validated as safe for hardware."""

    arm_rate: float = 1.0  # rad/s, any OpenArm joint
    hand_rate: float = 3.0  # rad/s, any Wuji joint
    lift_rate: float = 0.05  # m/s
    base_linear: float = 0.25  # m/s, |base.vx| and |base.vy|
    base_yaw: float = 0.5  # rad/s, |base.wz|
    base_linear_accel: float = 0.5  # m/s^2, also used to ramp to zero on stop
    base_yaw_accel: float = 1.0  # rad/s^2, also used to ramp to zero on stop

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"Envelope.{name} must be finite and positive")


# --- Conditions -------------------------------------------------------------

_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}


@dataclass(frozen=True)
class Signal:
    """Threshold on measured signals, evaluated by the executor every control tick.

    ``pattern`` is an fnmatch pattern over ``body.signals()`` names: measured
    joint positions use channel names (``right_arm.joint4``, rad), measured base
    body-frame velocities use ``base.vx``/``base.vy``/``base.wz``, and simulated
    fingertip normal contact force uses ``right_hand.index_finger.touch_N``.
    ``mode="any"`` fires when any matched signal satisfies ``op value``;
    ``mode="all"`` needs every match. It must hold for ``for_s`` seconds.
    """

    pattern: str
    op: str
    value: float
    for_s: float = 0.0
    mode: str = "any"

    def __post_init__(self):
        if self.op not in _OPS:
            raise TableError(f"Signal op must be one of {sorted(_OPS)}")
        if self.mode not in ("any", "all"):
            raise TableError("Signal mode must be 'any' or 'all'")
        if not np.isfinite(self.value) or not (
            np.isfinite(self.for_s) and self.for_s >= 0
        ):
            raise TableError("Signal value and for_s must be finite; for_s >= 0")


def touch(where: str, above_N: float, for_s: float = 0.0) -> Signal:
    """Simulated fingertip normal force above ``above_N`` on any matched pad.

    ``where`` is a hand, a finger or a pattern: ``"right_hand"``,
    ``"right_hand.index_finger"``, ``"*_hand.thumb"``.
    """
    pattern = where if where.endswith(".touch_N") else f"{where}.touch_N"
    if where.endswith("_hand"):
        pattern = f"{where}.*.touch_N"
    return Signal(pattern, ">", float(above_N), for_s)


@dataclass(frozen=True)
class Reached:
    """Measured attainment of the table's final row for its position channels:
    |measured - final target| <= tolerance on every channel for ``for_s``.
    Base velocity channels are not part of this test."""

    rad: float = 0.05
    m: float = 0.01
    for_s: float = 0.05

    def __post_init__(self):
        if not all(np.isfinite(v) and v >= 0 for v in (self.rad, self.m, self.for_s)):
            raise TableError("Reached tolerances must be finite and non-negative")


Condition = Signal | Reached


# --- Snapshots ---------------------------------------------------------------


@dataclass(frozen=True)
class Health:
    clock: str  # "realtime" or "manual"
    control_hz: float
    physics_dt: float
    running: bool
    faulted: bool
    error: str | None
    ticks: int
    sim_time: float
    wall_elapsed: float
    realtime_ratio: float  # sim seconds per wall second over the last window
    late_ticks: int  # ticks that started more than one period late
    dropped_s: float  # wall time not simulated because catch-up was capped
    max_tick_s: float
    mean_tick_s: float
    physics_warnings: int
    active_motions: tuple[int, ...]


def _frozen(array) -> np.ndarray:
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class BodyState:
    """Immutable snapshot published by the owner thread after a control tick."""

    sim_time: float  # time of the last confirmed sample all fields describe
    physics_time: float  # raw simulator clock; later than sim_time after a failed step
    wall_time: float  # time.monotonic() when published
    tick: int
    health: Health
    _names: tuple[str, ...] = field(repr=False)
    _resources: tuple[str, ...] = field(repr=False)
    _units: tuple[str, ...] = field(repr=False)
    _low: np.ndarray = field(repr=False)
    _high: np.ndarray = field(repr=False)
    _commanded: np.ndarray = field(repr=False)
    _measured: np.ndarray = field(repr=False)
    _velocity: np.ndarray = field(repr=False)
    _owner: np.ndarray = field(repr=False)
    _base_names: tuple[str, ...] = field(repr=False)
    _base_units: tuple[str, ...] = field(repr=False)
    _base_commanded: np.ndarray = field(repr=False)
    _base_measured: np.ndarray = field(repr=False)
    _base_owner: int = field(repr=False)
    _base_pose: np.ndarray = field(repr=False)
    _tactile_names: tuple[str, ...] = field(repr=False)
    _touch: np.ndarray = field(repr=False)
    _force: np.ndarray = field(repr=False)
    _torque: np.ndarray = field(repr=False)

    @property
    def joints(self) -> pd.DataFrame:
        """Position channels: arm/hand joints (rad) and lift (m)."""
        owner = [None if o < 0 else int(o) for o in self._owner]
        frame = pd.DataFrame(
            {
                "resource": list(self._resources),
                "unit": list(self._units),
                "commanded": self._commanded.copy(),
                "measured": self._measured.copy(),
                "error": self._measured - self._commanded,
                "velocity": self._velocity.copy(),
                "low": self._low.copy(),
                "high": self._high.copy(),
                "owner": pd.array(owner, dtype="Int64"),
            },
            index=pd.Index(list(self._names), name="channel"),
        )
        frame.attrs["sim_time"] = self.sim_time
        return frame

    @property
    def base(self) -> pd.DataFrame:
        """Base velocity channels in the current base frame (m/s, rad/s)."""
        owner = None if self._base_owner < 0 else int(self._base_owner)
        frame = pd.DataFrame(
            {
                "unit": list(self._base_units),
                "commanded": self._base_commanded.copy(),
                "measured": self._base_measured.copy(),
                "owner": pd.array([owner] * len(self._base_names), dtype="Int64"),
            },
            index=pd.Index(list(self._base_names), name="channel"),
        )
        frame.attrs["sim_time"] = self.sim_time
        return frame

    @property
    def base_pose(self) -> pd.Series:
        """Simulator ground truth (not odometry): world-frame position of the
        base yaw axis, the point base velocity commands move, and heading."""
        return pd.Series(
            self._base_pose.copy(), index=["x_m", "y_m", "yaw_rad"], name=self.sim_time
        )

    @property
    def tactile(self) -> pd.DataFrame:
        """Simulated fingertip pads. touch_N is normal contact force; force/torque
        are pad-frame transmitted loads including pad gravity and inertia."""
        frame = pd.DataFrame(
            np.column_stack([self._touch, self._force, self._torque]),
            columns=[
                "touch_N",
                "force_x_N",
                "force_y_N",
                "force_z_N",
                "torque_x_Nm",
                "torque_y_Nm",
                "torque_z_Nm",
            ],
            index=pd.Index(list(self._tactile_names), name="pad"),
        )
        frame.attrs["sim_time"] = self.sim_time
        return frame


@dataclass(frozen=True, eq=False)
class Frame:
    """RGB image rendered by the owner thread. ``rgb`` is a read-only copy."""

    rgb: np.ndarray
    sim_time: float
    view: str


# --- Motion results and handles ----------------------------------------------


@dataclass(frozen=True, eq=False)
class MotionResult:
    """Receipt for one table run. ``status`` says why it ended; ``attained``
    separately reports whether measured positions ended within tolerance of
    the table's final row. Neither is a task-success claim."""

    motion_id: int
    status: Status
    reason: str
    trigger: dict | None  # condition label, signal name, measured value, sim time
    resources: frozenset[str]
    channels: tuple[str, ...]
    units: tuple[str, ...]
    stop_hold: str
    start_time: float  # sim seconds when the first tick was commanded
    stop_time: float  # sim seconds when the terminal status was decided
    end_time: float  # sim seconds when leases were released (after base ramp)
    measured_time: float  # sim time of the confirmed sample in final_measured
    table_duration: float
    table_time_commanded: float  # how far through the table commands progressed
    rows_total: int
    rows_commanded: int  # rows whose time was reached by the command stream
    attained: bool | None  # None when the table has no position channels
    log_truncated: bool
    _final_target: np.ndarray = field(repr=False)
    _final_command: np.ndarray = field(repr=False)
    _final_measured: np.ndarray = field(repr=False)
    _attempted_command: np.ndarray | None = field(repr=False)
    _position_mask: np.ndarray = field(repr=False)
    _log_time: np.ndarray = field(repr=False)
    _log_tau: np.ndarray = field(repr=False)
    _log_phase: tuple[str, ...] = field(repr=False)
    _log_cmd: np.ndarray = field(repr=False)
    _log_meas: np.ndarray = field(repr=False)
    _log_signal_names: tuple[str, ...] = field(repr=False)
    _log_signals: np.ndarray = field(repr=False)
    _events: tuple[tuple, ...] = field(repr=False)

    def _series(self, values, name):
        return pd.Series(
            np.array(values, copy=True),
            index=pd.Index(list(self.channels), name="channel"),
            name=name,
        )

    @property
    def final_target(self) -> pd.Series:
        """The table's last row (base channels are zero by validation)."""
        return self._series(self._final_target, "final_target")

    @property
    def final_command(self) -> pd.Series:
        """Last command actually sent to physics before the leases were released."""
        return self._series(self._final_command, "final_command")

    @property
    def attempted_command(self) -> pd.Series | None:
        """Command passed to a step that then failed (fault receipts only).
        Whether physics applied it, fully or partly, is unknown."""
        if self._attempted_command is None:
            return None
        return self._series(self._attempted_command, "attempted_command")

    @property
    def final_measured(self) -> pd.Series:
        return self._series(self._final_measured, "final_measured")

    @property
    def tracking_error(self) -> pd.Series:
        """measured - final command (position channels; base is velocity)."""
        return self._series(
            self._final_measured - self._final_command, "tracking_error"
        )

    @property
    def target_error(self) -> pd.Series:
        """measured - table final row; large after an early stop by design."""
        return self._series(self._final_measured - self._final_target, "target_error")

    def log(self) -> pd.DataFrame:
        """Per-tick commanded vs measured for this motion's channels, plus every
        signal referenced by its conditions. Columns are (kind, name)."""
        parts = {
            ("commanded", c): self._log_cmd[:, i] for i, c in enumerate(self.channels)
        }
        parts |= {
            ("measured", c): self._log_meas[:, i] for i, c in enumerate(self.channels)
        }
        parts |= {
            ("signal", s): self._log_signals[:, i]
            for i, s in enumerate(self._log_signal_names)
        }
        frame = pd.DataFrame(
            {k: np.array(v, copy=True) for k, v in parts.items()},
            index=pd.Index(self._log_time.copy(), name="sim_time"),
        )
        frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["kind", "name"])
        frame.insert(0, ("table", "t"), self._log_tau.copy())
        frame.insert(1, ("table", "phase"), list(self._log_phase))
        frame.attrs["units"] = dict(zip(self.channels, self.units, strict=True))
        return frame

    def events(self) -> pd.DataFrame:
        return _events_frame(self._events)


def _events_frame(rows: Sequence[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        list(rows), columns=["sim_time", "wall_time", "kind", "motion_id", "detail"]
    ).astype({"motion_id": "Int64"})


class MotionHandle:
    """Asynchronous handle for one table run. Resolved exactly once by the
    owner thread (normal end, guard, cancel, fault or close)."""

    def __init__(
        self,
        motion_id: int,
        resources: frozenset[str],
        channels: tuple[str, ...],
        cancel: Callable[[int], None],
    ):
        self.id = motion_id
        self.resources = resources
        self.channels = channels
        self._cancel = cancel
        self._done = threading.Event()
        self._result: MotionResult | None = None
        self._lock = threading.Lock()

    def _resolve(self, result: MotionResult) -> None:
        with self._lock:
            if self._result is None:
                self._result = result
                self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def result(self, timeout: float | None = None) -> MotionResult:
        if not self._done.wait(timeout):
            raise TimeoutError(
                f"motion {self.id} still running (with clock='manual', a base "
                "ramp or measured hold is only sent inside advance())"
            )
        return self._result

    def cancel(
        self, wait: bool = True, timeout: float | None = 5.0
    ) -> MotionResult | None:
        """Request a controlled stop. Positions hold per the run's
        ``stop_hold``; base velocity ramps to zero at the envelope deceleration.
        The handle resolves once the hold has actually been sent to physics, so
        ``final_command`` is what was sent. ``wait=False`` returns None at once."""
        if not self.done():
            try:
                self._cancel(self.id)
            except (BodyClosed, BodyFaulted):
                pass  # shutdown resolves every handle itself
        return self.result(timeout) if wait else None

    def __repr__(self):
        state = self._result.status.value if self._result else "running"
        return f"<MotionHandle {self.id} {sorted(self.resources)} {state}>"


@runtime_checkable
class Body(Protocol):
    """Backend-independent body. SimBody implements it; a ROS 2 backend could."""

    def channels(self) -> pd.DataFrame: ...

    def signals(self) -> tuple[str, ...]: ...

    def state(self) -> BodyState: ...

    def run(
        self,
        table: pd.DataFrame,
        *,
        guard: Sequence[Condition] = (),
        until: Condition | None = None,
        timeout_s: float | None = None,
        stop_hold: str = "commanded",
    ) -> MotionHandle: ...

    def cancel_all(
        self, wait: bool = True, timeout: float | None = 5.0
    ) -> list[MotionResult] | None: ...

    def events(self) -> pd.DataFrame: ...

    def telemetry(self, last_s: float | None = None) -> pd.DataFrame: ...

    def capture(
        self, view: str = "overview", width: int = 640, height: int = 480
    ) -> Frame: ...

    def close(self) -> None: ...
