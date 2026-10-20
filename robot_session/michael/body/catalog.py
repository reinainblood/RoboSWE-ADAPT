"""Named channels for MichaelOpenArmWuji, discovered from the live model.

Built once on the owner thread; the resulting Catalog is plain immutable data
that callers use to validate tables without touching MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from ..build import FINGERS
from .types import Envelope

RESOURCES = ("right_arm", "right_hand", "left_arm", "left_hand", "base", "lift")
_PART_RESOURCE = {
    "right": "right_arm",
    "right_gripper": "right_hand",
    "left": "left_arm",
    "left_gripper": "left_hand",
    "base": "base",
    "torso": "lift",
}
BASE_CHANNELS = ("base.vx", "base.vy", "base.wz")


@dataclass(frozen=True)
class Channel:
    name: str
    resource: str
    mode: str  # "position" or "velocity"
    unit: str
    low: float  # model bound in `unit`
    high: float
    limit: float  # software envelope: |rate| for position, |value| for velocity
    accel: float  # software |d/dt| for velocity channels; nan for position
    native_index: int  # internal: slot in the 58-value robosuite action
    native_scale: float  # native = (value - native_bias) / native_scale
    native_bias: float
    joint: str  # MuJoCo joint driven by this channel


@dataclass(frozen=True)
class Catalog:
    channels: tuple[Channel, ...]
    index: MappingProxyType  # name -> position in `channels`
    tactile: tuple[str, ...]  # "right_hand.thumb", ... (pads, FINGERS order)
    action_dim: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)

    @property
    def signals(self) -> tuple[str, ...]:
        """Measured signal names: every channel, then each pad's touch_N."""
        return self.names + tuple(f"{p}.touch_N" for p in self.tactile)

    def frame(self) -> pd.DataFrame:
        rows = {
            c.name: {
                "resource": c.resource,
                "mode": c.mode,
                "unit": c.unit,
                "low": c.low,
                "high": c.high,
                "limit": c.limit,
                "limit_unit": f"{c.unit}/s" if c.mode == "position" else c.unit,
                "accel_limit": c.accel,
                "joint": c.joint,
            }
            for c in self.channels
        }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "channel"
        return frame


def _hand_channel(resource, joint, side_letter):
    tail = joint.split(f"_{side_letter}_", 1)[1]
    finger = next(f for f in FINGERS if tail.startswith(f + "_"))
    return f"{resource}.{finger}.{tail[len(finger) + 1 :]}"


def build_catalog(env, envelope: Envelope) -> Catalog:
    robot = env.robots[0]
    composite = robot.composite_controller
    model = env.sim.model._model
    low, high = composite.action_limits
    channels = []
    for part, controller in composite.part_controllers.items():
        resource = _PART_RESOURCE[part]
        start, end = composite._action_split_indexes[part]
        joints = list(controller.joint_names)
        actuators = robot._ref_actuators_indexes_dict[part]
        if len(joints) != end - start or len(actuators) != end - start:
            raise RuntimeError(f"Unexpected native layout for {part}")
        for offset, joint in enumerate(joints):
            # The commissioning model relies on actuator i driving joint i.
            driven = model.joint(int(model.actuator_trnid[actuators[offset], 0])).name
            if driven != joint:
                raise RuntimeError(
                    f"{part} actuator {offset} drives {driven}, not {joint}"
                )
            i = start + offset
            if resource in ("right_arm", "left_arm"):
                name = f"{resource}.joint{joint.rsplit('joint', 1)[1]}"
                unit, mode, limit = "rad", "position", envelope.arm_rate
                scale, bias, lo, hi = 1.0, 0.0, low[i], high[i]
            elif resource in ("right_hand", "left_hand"):
                name = _hand_channel(resource, joint, resource[0])
                unit, mode, limit = "rad", "position", envelope.hand_rate
                scale, bias, lo, hi = 1.0, 0.0, low[i], high[i]
            elif resource == "lift":
                name, unit, mode, limit = (
                    "lift.height",
                    "m",
                    "position",
                    envelope.lift_rate,
                )
                scale, bias, lo, hi = 1.0, 0.0, low[i], high[i]
            else:
                # robosuite maps normalized [-1, 1] onto the velocity actuator's
                # ctrlrange: ctrl = bias + weight * action.
                ctrl_lo, ctrl_hi = model.actuator_ctrlrange[actuators[offset]]
                bias, scale = (ctrl_hi + ctrl_lo) / 2, (ctrl_hi - ctrl_lo) / 2
                name = BASE_CHANNELS[offset]
                mode = "velocity"
                yaw = name == "base.wz"
                unit = "rad/s" if yaw else "m/s"
                limit = envelope.base_yaw if yaw else envelope.base_linear
                lo, hi = bias + scale * low[i], bias + scale * high[i]
                if limit > min(-lo, hi):
                    raise ValueError(f"Envelope for {name} exceeds model bound {hi}")
            channels.append(
                Channel(
                    name=name,
                    resource=resource,
                    mode=mode,
                    unit=unit,
                    low=float(lo),
                    high=float(hi),
                    limit=float(limit),
                    accel=float(
                        (
                            envelope.base_yaw_accel
                            if name == "base.wz"
                            else envelope.base_linear_accel
                        )
                        if mode == "velocity"
                        else np.nan
                    ),
                    native_index=i,
                    native_scale=float(scale),
                    native_bias=float(bias),
                    joint=joint,
                )
            )
    names = [c.name for c in channels]
    if len(set(names)) != len(names) or len(channels) != len(low):
        raise RuntimeError("Channel catalogue does not cover the native action once")
    if tuple(c.name for c in channels if c.resource == "base") != BASE_CHANNELS:
        raise RuntimeError("Unexpected base channel layout")
    tactile = tuple(
        f"{side}_hand.{finger}" for side in ("right", "left") for finger in FINGERS
    )
    return Catalog(
        channels=tuple(channels),
        index=MappingProxyType({n: i for i, n in enumerate(names)}),
        tactile=tactile,
        action_dim=len(low),
    )
