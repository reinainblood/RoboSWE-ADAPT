"""Gymnasium interface using the chosen RoboCasa robot's native action space.

Unlike upstream's PandaOmronKeyConverter, this adapter does not prescribe arm,
gripper, base or leg dimensions, nor reinterpret controller action values.
"""

import string
from typing import ClassVar
from xml.etree import ElementTree

import gymnasium as gym
import numpy as np


class RobotEnvironment(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, task, *, robot, split, seed, **kwargs):
        from robocasa.utils.env_utils import create_env
        from robosuite import macros as suite_macros
        from robosuite.robots import ROBOT_CLASS_MAPPING

        if robot == "MichaelOpenArmWuji":
            from .michael import register

            register()

        if robot not in ROBOT_CLASS_MAPPING:
            raise ValueError(
                f"Unknown robot {robot!r}. Installed robots: {', '.join(sorted(ROBOT_CLASS_MAPPING))}"
            )
        self.robot_name = robot
        self._raw_images_bottom_up = suite_macros.IMAGE_CONVENTION == "opengl"
        self._requested_cameras = kwargs.pop("camera_names", None)
        self._camera_options = {
            "camera_widths": kwargs.get("camera_widths", 128),
            "camera_heights": kwargs.get("camera_heights", 128),
            "camera_depths": kwargs.get("camera_depths", False),
            "camera_segmentations": kwargs.get("camera_segmentations", None),
        }
        kwargs.setdefault("renderer", "mujoco")
        self.env = create_env(
            task.removeprefix("robocasa/"),
            robots=robot,
            split=split,
            seed=seed,
            render_onscreen=False,
            camera_names=["auto"],
            **kwargs,
        )
        # RoboCasa builds the scene lazily on reset. Its supported processor hook
        # lets us inspect the final camera names before sensors are constructed.
        self.env.set_xml_processor(self._select_cameras)
        self.observation_space = None
        self.camera_keys = []

    def _select_cameras(self, xml):
        available = [
            camera.attrib["name"]
            for camera in ElementTree.fromstring(xml).iter("camera")
            if "name" in camera.attrib
        ]
        selected = self._requested_cameras
        if selected is None:
            selected = [
                name
                for name in available
                if name.endswith(("agentview_left", "agentview_right"))
                or "eye_in" in name
            ]
            selected = selected or available[:1]
        elif isinstance(selected, str):
            selected = [selected]
        if (
            not selected
            or len(set(selected)) != len(selected)
            or set(selected) - set(available)
        ):
            raise ValueError(
                f"Invalid cameras for {self.robot_name}: {selected}. Available: {available}"
            )
        self.env.camera_names = list(selected)
        self.env.num_cameras = len(selected)
        for key, value in self._camera_options.items():
            values = (
                list(value)
                if isinstance(value, (tuple, list))
                else [value] * len(selected)
            )
            if len(values) != len(selected):
                raise ValueError(
                    f"{key} must have one value per selected camera ({len(selected)})"
                )
            setattr(self.env, key, values)
        return xml

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.rng = np.random.default_rng(seed)
        raw = self.env.reset()
        actual = type(self.env.robots[0].robot_model).__name__
        if actual != self.robot_name:
            raise ValueError(
                f"Requested {self.robot_name}, but environment instantiated {actual}"
            )
        # Controllers can be reconstructed during reset; read the live limits now.
        low, high = self.env.action_spec
        self.action_space = gym.spaces.Box(
            np.asarray(low), np.asarray(high), dtype=np.float64
        )
        observation = self._observation(raw)
        spaces = {}
        for key, value in observation.items():
            if isinstance(value, str):
                spaces[key] = gym.spaces.Text(
                    max_length=max(16384, len(value)),
                    charset=set(string.printable + value),
                )
            else:
                array = np.asarray(value)
                if np.issubdtype(array.dtype, np.integer):
                    bounds = np.iinfo(array.dtype)
                    spaces[key] = gym.spaces.Box(
                        bounds.min, bounds.max, array.shape, dtype=array.dtype
                    )
                else:
                    spaces[key] = gym.spaces.Box(
                        -np.inf, np.inf, array.shape, dtype=array.dtype
                    )
        self.observation_space = gym.spaces.Dict(spaces)
        return observation, {"success": False}

    def _observation(self, raw):
        observation = {}
        self.camera_keys = []
        for key, value in raw.items():
            array = np.asarray(value)
            if key.endswith(("_image", "_depth")):
                array = (
                    array[::-1].copy()
                    if getattr(self, "_raw_images_bottom_up", True)
                    else array.copy()
                )
                if key.endswith("_image") and array.ndim == 3 and array.shape[-1] == 3:
                    self.camera_keys.append(key)
            else:
                array = array.copy()
            observation[key] = array
        observation["annotation.human.task_description"] = self.env.get_ep_meta().get(
            "lang", ""
        )
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info = {**info, "success": bool(self.env._check_success())}
        return self._observation(observation), reward, bool(done), False, info

    def enable_native_hand_ranges(self):
        """Opt-in correction for raw-position hands; commands pass through unchanged."""
        from .interface import raw_hand_command_bounds

        low, high = self.action_space.low.copy(), self.action_space.high.copy()
        corrections = {}
        offset = 0
        for robot in self.env.robots:
            controller = robot.composite_controller
            parts = controller.get_action_info_dict()
            for arm, gripper in robot.gripper.items():
                part = arm + "_gripper"
                bounds = raw_hand_command_bounds(gripper, robot.part_controllers[part])
                if bounds is None:
                    continue
                start, end = parts[part]
                if end - start != len(bounds[0]):
                    raise ValueError("Native hand slice dimension mismatch")
                low[offset + start:offset + end] = np.minimum(low[offset + start:offset + end], bounds[0])
                high[offset + start:offset + end] = np.maximum(high[offset + start:offset + end], bounds[1])
                corrections[f"{offset}/{part}"] = {"slice": [offset + start, offset + end],
                    "low": low[offset + start:offset + end].tolist(), "high": high[offset + start:offset + end].tolist(),
                    "source": "Native raw-position actuator ranges and upstream gripper grasp_qpos; no action scaling or controller mutation"}
            offset += len(controller.action_limits[0])
        self.action_space = gym.spaces.Box(low, high, dtype=np.float64)
        self._native_hand_range_corrections = corrections
        return corrections

    def action_schema(self):
        from robosuite.controllers.composite.composite_controller import (
            HybridMobileBase,
        )

        from .interface import controller_parameters

        def bounds(values):
            # JSON null denotes an unbounded lower/upper limit, respectively.
            return [float(v) if np.isfinite(v) else None for v in values]

        controllers = []
        offset = 0
        for robot in self.env.robots:
            controller = robot.composite_controller
            parts = {
                key: list(value)
                for key, value in controller.get_action_info_dict().items()
                if key != "Action Dimension"
            }
            dimension = len(controller.action_limits[0])
            if isinstance(controller, HybridMobileBase):
                parts["base_mode"] = [dimension - 1, dimension]
            controllers.append(
                {
                    "robot": type(robot.robot_model).__name__,
                    "controller": type(controller).__name__,
                    "offset": offset,
                    "dimension": dimension,
                    "parts": parts,
                    "joint_observation_names": list(getattr(robot, "robot_joints", ())),
                    "parameters": controller_parameters(controller),
                    "parameter_scope": "Individual part-controller parameters after composite mappings; native-vector slices above remain authoritative (especially for whole-body IK and grippers).",
                    "mode_semantics": "HybridMobileBase last action > 0 updates arm goals from desired pose; <= 0 updates from achieved pose. Both base and arm actions are passed to their controllers."
                    if isinstance(controller, HybridMobileBase)
                    else None,
                }
            )
            offset += dimension
        result = {
            "format": "native_vector",
            "shape": list(self.action_space.shape),
            "dtype": str(self.action_space.dtype),
            "low": bounds(self.action_space.low),
            "high": bounds(self.action_space.high),
            "controllers": controllers,
            "control_frequency_hz": getattr(self.env, "control_freq", None),
        }
        if getattr(self, "_native_hand_range_corrections", None):
            result["native_hand_contract"] = "native-raw-position-v2"
            result["native_hand_range_corrections"] = self._native_hand_range_corrections
        return result

    def camera_calibration(self, camera_key, shape):
        from .interface import camera_calibration

        if camera_key not in self.camera_keys:
            raise ValueError("Camera is not an enabled robot observation")
        return camera_calibration(
            self.env.sim, camera_key.removesuffix("_image"), shape[0], shape[1]
        )

    def additional_robot_sensors(self):
        from .interface import wrist_sensors

        values, descriptions = wrist_sensors(self.env)
        if self.robot_name == "MichaelOpenArmWuji":
            from .michael.runtime import fingertip_sensors

            tactile_values, tactile_descriptions = fingertip_sensors(self.env)
            values.update(tactile_values)
            descriptions.update(tactile_descriptions)
        return values, descriptions

    def close(self):
        self.env.close()
