"""Numeric verdicts evaluated only from verifier-owned MuJoCo state.

The agent's return code, rewards, reports, and snapshots are never inputs here.
This module is not itself a sandbox: the calling runtime must protect its state.
"""

import math
from dataclasses import asdict, dataclass

from .spec import GOALS


@dataclass(frozen=True)
class JointBinding:
    name: str
    low: float
    high: float
    physical_low: float | None = None
    physical_high: float | None = None


@dataclass(frozen=True)
class FixtureBinding:
    role: str
    name: str
    joints: tuple[JointBinding, ...]

    def to_dict(self):
        return asdict(self)


def evaluate(task, binding, positions):
    goal = GOALS[task]
    if binding.role != goal.fixture or not binding.joints:
        raise ValueError("Missing or incorrect fixture binding")
    measures = []
    for joint in binding.joints:
        if joint.name not in positions:
            raise ValueError(f"Missing joint measurement: {joint.name}")
        q = float(positions[joint.name])
        lo, hi = joint.low, joint.high
        if not all(math.isfinite(v) for v in (q, lo, hi)) or hi <= lo:
            raise ValueError("Non-finite measurement or invalid joint range")
        physical_low = lo if joint.physical_low is None else joint.physical_low
        physical_high = hi if joint.physical_high is None else joint.physical_high
        if (
            not all(math.isfinite(v) for v in (physical_low, physical_high))
            or physical_high <= physical_low
        ):
            raise ValueError("Invalid physical range")
        if q < physical_low - 0.01 or q > physical_high + 0.01:
            raise ValueError("Joint outside physical range beyond tolerance")
        # Upstream reverses normalized joint coordinates for negative-range hinges.
        normalized = (q - lo) / (hi - lo)
        if lo < 0:
            normalized = 1 - normalized
        if goal.fixture == "drawer":
            # Drawer.get_door_state uses absolute displacement from closed,
            # including small positive solver overshoots of its negative slide.
            normalized = abs(q) / (hi - lo)
        if goal.predicate in ("water_on", "water_off"):
            angle = q % (2 * math.pi)
            on = goal.threshold < angle < math.pi
            ok = on if goal.predicate == "water_on" else not on
            margin = min(angle - goal.threshold, math.pi - angle)
            if goal.predicate == "water_off":
                margin = -margin
        elif goal.predicate == "open":
            margin = normalized - goal.threshold
            ok = margin >= 0
        else:
            margin = goal.threshold - normalized
            ok = margin >= 0
        measures.append(
            {
                "joint": joint.name,
                "qpos": q,
                "range": [lo, hi],
                "normalized": normalized,
                "margin": margin,
                "satisfied": bool(ok),
            }
        )
    return {
        "task": task,
        "fixture": binding.name,
        "predicate": goal.predicate,
        "threshold": goal.threshold,
        "success": all(m["satisfied"] for m in measures),
        "measurements": measures,
    }


def bind_fixtures(raw):
    """Bind by fixture type once, before the run; never fuzzy-match agent names."""
    from robocasa.models.fixtures import FixtureType, fixture_is_type

    kinds = {
        "microwave": FixtureType.MICROWAVE,
        "oven": FixtureType.OVEN,
        "drawer": FixtureType.TOP_DRAWER,
        "sink": FixtureType.SINK,
    }
    result = {}
    for role, kind in kinds.items():
        candidates = sorted(
            (f for f in raw.fixtures.values() if fixture_is_type(f, kind)),
            key=lambda f: f.name,
        )
        if not candidates:
            raise ValueError(f"Kitchen has no supported {role}")
        fixture = candidates[0]
        if role == "sink":
            names = [fixture.naming_prefix + "handle_joint"]
        elif role == "drawer":
            names = [fixture.name + "_slidejoint"]
        else:
            names = list(fixture.door_joint_names)
        joints = []
        for name in names:
            model = raw.sim.model
            try:
                jid = model.joint_name2id(name)
            except ValueError as exc:
                raise ValueError(f"Unsupported fixture joint {name}") from exc
            lo, hi = map(float, model.jnt_range[jid])
            if role == "drawer":
                # Upstream Drawer.get_door_state uses 55% of fixture depth,
                # not the full XML slide-joint range, as its normalized opening.
                joints.append(
                    JointBinding(name, -float(fixture.size[1]) * 0.55, 0, lo, hi)
                )
            else:
                joints.append(JointBinding(name, lo, hi))
        if not joints:
            raise ValueError(f"No measurable joints for {role}")
        result[role] = FixtureBinding(role, fixture.name, tuple(joints))
    return result


def read_positions(raw, bindings):
    return {
        joint.name: float(
            raw.sim.data.qpos[raw.sim.model.get_joint_qpos_addr(joint.name)]
        )
        for binding in bindings.values()
        for joint in binding.joints
    }


class TaskVerifier:
    """Require a real state transition and sustained success; no time-of-check trick."""

    def __init__(self, task, binding, initial_positions, *, settle_steps=10):
        if type(settle_steps) is not int or settle_steps < 1:
            raise ValueError("settle_steps must be a positive integer")
        self.task, self.binding = task, binding
        self.initial = evaluate(task, binding, initial_positions)
        self.precondition_valid = not self.initial["success"]
        self.settle_steps = settle_steps
        self.consecutive = 0
        self.samples = 0
        self.last = self.initial
        self.first_satisfied_step = None
        self.last_step = None
        self.measurement_valid = True
        self.validation_error = None

    def update(self, positions, control_step):
        if (
            type(control_step) is not int
            or control_step < 1
            or (self.last_step is not None and control_step != self.last_step + 1)
        ):
            raise ValueError(
                "Success must be checked on consecutive actual control steps"
            )
        self.last_step = control_step
        self.samples += 1
        try:
            self.last = evaluate(self.task, self.binding, positions)
        except ValueError as exc:
            self.measurement_valid = False
            self.validation_error = str(exc)
            self.consecutive = 0
            self.first_satisfied_step = None
            raise
        self.measurement_valid = True
        self.validation_error = None
        self.consecutive = self.consecutive + 1 if self.last["success"] else 0
        if self.consecutive == 1:
            self.first_satisfied_step = control_step
        return self.verdict()

    def verdict(self):
        return {
            "task": self.task,
            "precondition_valid": self.precondition_valid,
            "success": self.precondition_valid
            and self.measurement_valid
            and self.consecutive >= self.settle_steps,
            "status": "precondition_invalid"
            if not self.precondition_valid
            else "invalid_measurement"
            if not self.measurement_valid
            else "passed"
            if self.consecutive >= self.settle_steps
            else "not_yet_passed",
            "initial": self.initial,
            "final": self.last,
            "measurement_valid": self.measurement_valid,
            "validation_error": self.validation_error,
            "samples": self.samples,
            "sustained_steps": self.consecutive,
            "required_sustained_steps": self.settle_steps,
            "first_satisfied_step": self.first_satisfied_step,
        }
