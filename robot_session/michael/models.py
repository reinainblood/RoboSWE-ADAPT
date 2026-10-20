"""Opt-in robosuite models, direct manufacturer position control and tactile IO."""

from copy import deepcopy
from typing import ClassVar

import numpy as np
from robosuite.controllers import controller_factory
from robosuite.controllers.composite.composite_controller import (
    CompositeController,
    register_composite_controller,
)
from robosuite.controllers.parts.controller import Controller
from robosuite.controllers.parts.generic.joint_pos import JointPositionController
from robosuite.controllers.parts.mobile_base.joint_vel import (
    MobileBaseJointVelocityController,
)
from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.wheeled_robot import WheeledRobot
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import mat2euler

from .build import FINGERS, build_models

PATHS = build_models()


class ManufacturerPositionController(JointPositionController):
    """Send radians/metres to MJCF position actuators; never torque-as-position."""

    def __init__(self, **kwargs):
        kwargs = {**kwargs, "input_type": "absolute"}
        super().__init__(**kwargs)
        self.input_min = np.asarray(self.actuator_min).copy()
        self.input_max = np.asarray(self.actuator_max).copy()

    def run_controller(self):
        self.update()
        if self.goal_qpos is None:
            self.goal_qpos = self.joint_pos.copy()
        result = np.clip(self.goal_qpos, self.actuator_min, self.actuator_max)
        Controller.run_controller(self)
        return result

    @property
    def name(self):
        return "MANUFACTURER_POSITION"


class MichaelBaseVelocityController(MobileBaseJointVelocityController):
    """Current-base-frame (x, y, yaw) velocity -> Omron slide/yaw joint targets.

    The slide joints move in the initial base frame, so a base-frame command
    must be rotated by +theta. The inherited controller rotates by -theta,
    which turns commands by -2*theta once the base has yawed.
    """

    def set_goal(self, action, set_qpos=None):
        if self.impedance_mode != "fixed":
            raise NotImplementedError("Michael's base uses fixed impedance")
        self.update()
        _, orientation = self.get_base_pose()
        theta = mat2euler(orientation)[2] - mat2euler(self.init_ori)[2]
        x, y, yaw = np.asarray(action, dtype=float)
        c, s = np.cos(theta), np.sin(theta)
        self.goal_qvel = np.array([c * x - s * y, s * x + c * y, yaw])
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qvel)


@register_composite_controller
class MichaelNativePosition(CompositeController):
    name = "MICHAEL_NATIVE_POSITION"

    def _init_controllers(self):
        for part, params in self.part_controller_config.items():
            if part in ("right", "left", "torso"):
                controller = ManufacturerPositionController(**params)
            elif part == "base":
                controller = MichaelBaseVelocityController(interpolator=None, **params)
            else:
                controller = controller_factory(
                    part_name=part,
                    controller_type=params["type"],
                    controller_params=params,
                )
            if part in self.grippers:
                controller.input_min = np.asarray(controller.actuator_min).copy()
                controller.input_max = np.asarray(controller.actuator_max).copy()
            if not hasattr(controller, "joint_names"):
                controller.joint_names = [
                    self.sim.model.joint_id2name(i) for i in controller.joint_index
                ]
            self.part_controllers[part] = controller

    @property
    def action_limits(self):
        lows, highs = [], []
        for part, controller in self.part_controllers.items():
            if part in self.grippers:
                lo, hi = controller.actuator_min, controller.actuator_max
            else:
                lo, hi = controller.control_limits
            lows.extend(lo)
            highs.extend(hi)
        return np.asarray(lows), np.asarray(highs)

    def set_goal(self, action):
        values = np.asarray(action, dtype=float)
        low, high = self.action_limits
        if (
            values.shape != low.shape
            or not np.isfinite(values).all()
            or np.any(values < low)
            or np.any(values > high)
        ):
            raise ValueError(
                "Michael action must be finite and within each native joint/control range"
            )
        super().set_goal(values.copy())

    def create_action_vector(self, action_dict):
        """Partial commands hold each omitted arm/hand at its previous target."""
        values = []
        for part, controller in self.part_controllers.items():
            if part == "base":
                hold = np.zeros(3)
            elif part in self.grippers:
                hold = controller.goal_qvel
                if hold is None:
                    hold = self.sim.data.qpos[controller.qpos_index]
            else:
                hold = controller.goal_qpos
                if hold is None:
                    hold = self.sim.data.qpos[controller.qpos_index]
            values.extend(action_dict.get(part, hold))
        unknown = set(action_dict) - set(self.part_controllers)
        if unknown:
            raise ValueError(f"Unknown Michael controls: {sorted(unknown)}")
        return np.asarray(values, dtype=float)


def controller_config():
    arm = {
        "type": "JOINT_POSITION",
        "input_type": "absolute",
        "interpolation": None,
        "gripper": {"type": "GRIP", "use_action_scaling": False},
    }
    return {
        "type": "MICHAEL_NATIVE_POSITION",
        "body_parts": {
            "right": deepcopy(arm),
            "left": deepcopy(arm),
            "base": {"type": "JOINT_VELOCITY", "interpolation": None},
            "torso": {
                "type": "JOINT_POSITION",
                "input_type": "absolute",
                "interpolation": None,
            },
        },
        "composite_controller_specific_configs": {
            "body_part_ordering": [
                "right",
                "right_gripper",
                "left",
                "left_gripper",
                "base",
                "torso",
            ],
        },
    }


class WujiHand2(GripperModel):
    side = "right"

    def __init__(self, idn=0):
        super().__init__(str(PATHS[self.side]), idn=idn)

    def format_action(self, action):
        return np.asarray(action, dtype=float)

    @property
    def init_qpos(self):
        return np.zeros(20)

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["thumb_pad_collision"],
            "right_finger": [f"{f}_pad_collision" for f in FINGERS[1:]],
            "left_fingerpad": ["thumb_pad_collision"],
            "right_fingerpad": [f"{f}_pad_collision" for f in FINGERS[1:]],
        }


@register_gripper
class WujiHand2Right(WujiHand2):
    side = "right"


@register_gripper
class WujiHand2Left(WujiHand2):
    side = "left"


class MichaelOpenArmWuji(ManipulatorModel):
    arms: ClassVar[list[str]] = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(str(PATHS["robot"]), idn=idn)

    def add_base(self, base):
        # robosuite's Omron is a planar velocity-actuated abstraction (no wheel
        # dynamics or tipping). Remove its artificial stiction for low-speed
        # commissioning. The lift is a position servo, limited to its real range.
        for joint in base.worldbody.iter("joint"):
            joint.set("frictionloss", "0")
        lift = next(a for a in base.actuator if "torso" in a.get("name", ""))
        lift.tag = "position"
        lift.attrib.update(
            kp="12000",
            kv="1000",
            ctrllimited="true",
            ctrlrange="0 0.34",
            forcelimited="true",
            forcerange="-6000 6000",
        )
        super().add_base(base)

    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_gripper(self):
        return {"right": "WujiHand2Right", "left": "WujiHand2Left"}

    @property
    def default_controller_config(self):
        return {
            "right": "joint_position",
            "left": "joint_position",
            "torso": "joint_position",
        }

    @property
    def init_qpos(self):
        return np.array([0, 0.35, 0, 1.25, 0, 0, 0, 0, -0.35, 0, 1.25, 0, 0, 0])

    @property
    def init_torso_qpos(self):
        return np.array([0.10])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.7, 0, 0),
            "empty": (-0.7, 0, 0),
            "table": lambda length: (-0.5 - length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array([0, 0, 1.5])

    @property
    def _horizontal_radius(self):
        return 0.65

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        return {"right": "right_hand", "left": "left_hand"}


class MichaelWheeledRobot(WheeledRobot):
    def __init__(self, *args, **kwargs):
        kwargs["composite_controller_config"] = controller_config()
        super().__init__(*args, **kwargs)

    def setup_observables(self):
        observables = super().setup_observables()
        modality = f"{self.robot_model.naming_prefix}tactile"
        for side in self.arms:
            gripper = self.gripper[side]
            for kind in ("touch", "force", "torque"):
                names = [
                    gripper.correct_naming(f"{finger}_{kind}") for finger in FINGERS
                ]

                @sensor(modality=modality)
                def read_pad(obs_cache, names=names, kind=kind):
                    values = np.array([self.get_sensor_measurement(n) for n in names])
                    return values.ravel()

                name = f"{self.robot_model.naming_prefix}{side}_fingertip_{kind}"
                observables[name] = Observable(
                    name=name, sensor=read_pad, sampling_rate=self.control_freq
                )
        return observables


ROBOT_CLASS_MAPPING["MichaelOpenArmWuji"] = MichaelWheeledRobot
