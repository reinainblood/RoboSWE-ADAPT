"""Validate pandas movement tables and compile them into numeric programs.

A movement table is a finite DataFrame with a ``t`` column (or an index named
``t``): seconds after the motion starts, strictly increasing, first row > 0.
Every other column is a channel name from ``body.channels()``. The executor
prepends the current *commanded* value at t=0 and interpolates linearly.

* Position channels (arm/hand rad, lift m): absolute targets within model
  bounds; the per-segment rate must stay within the envelope, including the
  first segment from the current command (checked at activation).
* Base velocity channels (m/s, rad/s, current base frame): an omitted base
  channel is zero. The last row must be zero: base motion always expires.
* Omitted channels of a leased resource keep their current command; omitted
  resources are not leased and keep holding.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .catalog import BASE_CHANNELS, Catalog
from .types import _OPS, Condition, Reached, Signal, TableError

MAX_TABLE_S = 120.0
_EPS = 1e-9


def _readonly(array):
    array = np.array(array, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class Program:
    names: tuple[str, ...]
    columns: np.ndarray  # catalogue indices, int
    times: np.ndarray  # (rows,) seconds, without the implicit t=0 knot
    values: np.ndarray  # (rows, channels) in channel units
    resources: frozenset[str]
    position: np.ndarray  # bool mask over columns
    rates: np.ndarray  # per-column envelope (position: unit/s; velocity: |v|)

    @property
    def duration(self) -> float:
        return float(self.times[-1])


def compile_table(
    table: pd.DataFrame,
    catalog: Catalog,
    *,
    base_kinematics: str = "differential",
    max_duration_s: float = MAX_TABLE_S,
) -> Program:
    if not isinstance(table, pd.DataFrame):
        raise TableError("A movement table must be a pandas DataFrame")
    if any(not isinstance(c, str) for c in table.columns):
        raise TableError("Table columns must be channel-name strings")
    if table.columns.duplicated().any():
        raise TableError("Duplicate table columns")
    if "t" in table.columns:
        time = table["t"]
        names = [c for c in table.columns if c != "t"]
    elif table.index.name == "t":
        time = table.index.to_series()
        names = list(table.columns)
    else:
        raise TableError(
            "Table needs a 't' column (seconds after start) or index named 't'"
        )
    if len(table) == 0 or not names:
        raise TableError("Table needs at least one row and one channel column")
    unknown = [c for c in names if c not in catalog.index]
    if unknown:
        raise TableError(f"Unknown channels {unknown}; see body.channels()")
    for label, series in [("t", time), *((c, table[c]) for c in names)]:
        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_complex_dtype(series)
        ):
            raise TableError(
                f"Column {label!r} must be real numeric, got {series.dtype}"
            )
    times = time.to_numpy(dtype=float, na_value=np.nan)
    values = table[names].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(times).all():
        raise TableError("t must be finite; missing times are not allowed")
    bad = ~np.isfinite(values)
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise TableError(
            f"Missing/non-finite target in {names[col]!r} at t={times[row]:g}; "
            "omit the column to hold, never leave gaps"
        )
    if times[0] <= 0:
        raise TableError(
            "First row must have t > 0; the current command is the t=0 knot"
        )
    if np.any(np.diff(times) <= 0):
        raise TableError("t must be strictly increasing")
    if times[-1] > max_duration_s:
        raise TableError(
            f"Table lasts {times[-1]:g} s; v0 allows at most {max_duration_s:g} s"
        )

    # Base velocity is all-or-nothing: missing base channels are commanded zero.
    if any(n in BASE_CHANNELS for n in names):
        for n in BASE_CHANNELS:
            if n not in names:
                names.append(n)
                values = np.column_stack([values, np.zeros(len(times))])
    channels = [catalog.channels[catalog.index[n]] for n in names]

    for j, ch in enumerate(channels):
        col = values[:, j]
        if ch.mode == "position":
            # Strict: the native controller rejects anything outside these.
            outside = (col < ch.low) | (col > ch.high)
            if outside.any():
                k = int(np.argmax(outside))
                raise TableError(
                    f"{ch.name}={col[k]:g} {ch.unit} at t={times[k]:g} is outside "
                    f"[{ch.low:g}, {ch.high:g}]"
                )
            if len(col) > 1:
                rate = np.abs(np.diff(col)) / np.diff(times)
                if rate.max() > ch.limit + _EPS:
                    k = int(np.argmax(rate))
                    raise TableError(
                        f"{ch.name} moves at {rate[k]:.3g} {ch.unit}/s before t={times[k + 1]:g}; "
                        f"envelope is {ch.limit:g} {ch.unit}/s"
                    )
        else:
            if np.abs(col).max() > ch.limit + _EPS:
                raise TableError(f"|{ch.name}| exceeds envelope {ch.limit:g} {ch.unit}")
            if col[-1] != 0:
                raise TableError(
                    f"{ch.name} must be 0 in the last row; base motion must expire"
                )
            if (
                ch.name == "base.vy"
                and base_kinematics == "differential"
                and np.any(col != 0)
            ):
                raise TableError(
                    "base.vy is rejected in differential mode (software constraint; "
                    "the simulated base itself is an ideal holonomic abstraction)"
                )
            knots_t = np.concatenate([[0.0], times])
            accel = np.abs(np.diff(np.concatenate([[0.0], col]))) / np.diff(knots_t)
            if accel.max() > ch.accel + _EPS:
                k = int(np.argmax(accel))
                raise TableError(
                    f"{ch.name} accelerates at {accel[k]:.3g} {ch.unit}^2 before t={times[k]:g}; "
                    f"envelope is {ch.accel:g} {ch.unit}^2"
                )

    columns = np.array([catalog.index[n] for n in names], dtype=int)
    columns.setflags(write=False)
    position = np.array([c.mode == "position" for c in channels])
    position.setflags(write=False)
    return Program(
        names=tuple(names),
        columns=columns,
        times=_readonly(times),
        values=_readonly(values),
        resources=frozenset(c.resource for c in channels),
        position=position,
        rates=_readonly([c.limit for c in channels]),
    )


@dataclass
class CompiledCondition:
    label: str
    kind: str  # "signal" or "reached"
    indices: np.ndarray  # into the signal vector
    names: tuple[str, ...]
    ticks_required: int
    op: str = ">"
    value: float = 0.0
    mode: str = "any"
    target: np.ndarray | None = None
    tolerance: np.ndarray | None = None
    count: int = 0

    def check(self, signals: np.ndarray):
        """Return (fired, signal name, value) and update the hold counter."""
        x = signals[self.indices]
        if self.kind == "reached":
            err = np.abs(x - self.target)
            ok = bool(np.all(err <= self.tolerance))
            k = int(np.argmax(err - self.tolerance))
            name, value = self.names[k], float(err[k])
        else:
            hit = _OPS[self.op](x, self.value)
            ok = bool(hit.any() if self.mode == "any" else hit.all())
            k = (
                int(np.argmax(hit))
                if self.mode == "any" and ok
                else int(np.argmin(hit))
            )
            name, value = self.names[k], float(x[k])
        self.count = self.count + 1 if ok else 0
        return self.count >= self.ticks_required, name, value


def compile_condition(
    condition: Condition, catalog: Catalog, program: Program, dt: float
) -> CompiledCondition:
    signals = catalog.signals
    if isinstance(condition, Signal):
        matched = [
            i
            for i, s in enumerate(signals)
            if fnmatch.fnmatchcase(s, condition.pattern)
        ]
        if not matched:
            raise TableError(
                f"Signal pattern {condition.pattern!r} matches nothing; see body.signals()"
            )
        return CompiledCondition(
            label=f"{condition.pattern} {condition.op} {condition.value:g} ({condition.mode}, {condition.for_s:g}s)",
            kind="signal",
            indices=np.array(matched),
            names=tuple(signals[i] for i in matched),
            ticks_required=max(1, math.ceil(condition.for_s / dt - _EPS)),
            op=condition.op,
            value=float(condition.value),
            mode=condition.mode,
        )
    if isinstance(condition, Reached):
        if not program.position.any():
            raise TableError(
                "Reached() needs at least one position channel in the table"
            )
        cols = program.columns[program.position]
        units = [catalog.channels[i].unit for i in cols]
        return CompiledCondition(
            label=f"reached final row (rad<={condition.rad:g}, m<={condition.m:g}, {condition.for_s:g}s)",
            kind="reached",
            indices=np.array(cols),
            names=tuple(catalog.channels[i].name for i in cols),
            ticks_required=max(1, math.ceil(condition.for_s / dt - _EPS)),
            target=np.array(program.values[-1, program.position]),
            tolerance=np.array(
                [condition.rad if u == "rad" else condition.m for u in units]
            ),
        )
    raise TableError(
        f"Unsupported condition {condition!r}; use Signal, touch() or Reached"
    )
