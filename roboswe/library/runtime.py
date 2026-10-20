"""Trusted persistent kitchen and task progression, outside the code sandbox."""

import base64
import copy
import hashlib
import json
import math
import re
import time
from io import BytesIO

from .spec import GOALS, goal_criteria
from .verifiers import TaskVerifier, bind_fixtures, read_positions

# Explicit sensor names only. In particular, never export object-state,
# object-to-gripper poses, fixture coordinates, segment labels, or XML state.
PROPRIO = re.compile(
    r"robot\d+_(?:(?:left|right)_)?(?:joint_pos(?:_cos|_sin)?|joint_vel|joint_acc|eef_pos|eef_quat(?:_site)?|gripper_qpos|gripper_qvel|base_pos|base_quat|base_to_eef_pos|base_to_eef_quat(?:_site)?)$"
)


def public_observation(session):
    from PIL import Image

    cache_enabled = getattr(session, "_cache_public_observation", False)
    cache_key = (id(session.observation), session.steps, session.phase, getattr(session, "_nested_sensor_documentation", False))
    if cache_enabled:
        cached = getattr(session, "_public_observation_cache", None)
        if cached is not None and cached[0] == cache_key:
            return copy.deepcopy(cached[1])
    sensors = {}
    images = {}
    cameras = {}
    depths = {}
    for name, value in session.observation.items():
        if PROPRIO.fullmatch(name):
            sensors[name] = value.tolist()
        elif name in session.env.camera_keys:
            output = BytesIO()
            Image.fromarray(value).save(output, format="PNG")
            images[name] = base64.b64encode(output.getvalue()).decode()
            cameras[name] = session.env.camera_calibration(name, value.shape)
            depth_name = name.removesuffix("_image") + "_depth"
            if depth_name in session.observation:
                from robot_session.interface import metric_depth

                depths[depth_name] = metric_depth(
                    session.raw.sim, session.observation[depth_name]
                )
                cameras[name]["depth_key"] = depth_name
    result = {
        "control_step": session.steps,
        "simulator_phase": session.phase,
        "sensors": sensors,
        "images_png_base64": images,
        "action_schema": session.env.action_schema(),
        "camera_calibration": cameras,
        "sensor_conventions": {
            "quaternions": "xyzw",
            "eef_pos": "grip-site position in world metres",
            "eef_quat_site": "grip-site orientation in world; use with eef_pos",
            "eef_quat": "legacy end-effector BODY orientation, which may differ from the grip site",
            "base_to_eef": "base-relative pose; prefer the _quat_site variant for the grip site",
            "localization": "ideal simulated world poses and calibrated camera kinematics",
            "lidar": "not configured",
        },
    }
    if hasattr(session.env, "additional_robot_sensors"):
        extra, descriptions = session.env.additional_robot_sensors()
        result["sensors"].update(extra)
        result["sensor_descriptions"] = descriptions
    if getattr(session, "_nested_sensor_documentation", False):
        result["image_data_contract"] = {
            "rgb_payload": "robot.observe()['images_png_base64'][camera_key]; images are not top-level obs[camera_key] fields",
            "rgb_decode_example": "Image.open(BytesIO(base64.b64decode(obs['images_png_base64'][camera_key])))",
            "camera_names": "Use keys in images_png_base64 or camera_calibration; do not invent camera names",
            "sensor_vectors": "obs['sensors'][sensor_name] is a JSON list. Use numpy.asarray(...) for NumPy operations; lists have no .tolist() method.",
        }
    if depths:
        import numpy as np

        output = BytesIO()
        np.savez_compressed(output, **depths)
        result["depth_npz_base64"] = base64.b64encode(output.getvalue()).decode()
        result["depth_arrays"] = {
            name: {
                "shape": list(value.shape),
                "dtype": "float32",
                "units": "optical-axis metres",
            }
            for name, value in depths.items()
        }
        result["depth_data_contract"] = {
            "depth_arrays": "Metadata only (shape, dtype, units); not pixel measurements.",
            "payload": "robot.observe()['depth_npz_base64']: one base64-encoded NPZ containing all named depth arrays.",
            "decode_example": "with numpy.load(BytesIO(base64.b64decode(obs['depth_npz_base64'])), allow_pickle=False) as arrays: depth = arrays[obs['camera_calibration'][image_key]['depth_key']].copy()",
            "calibration": "Use camera_calibration from the SAME observation as the depth. Camera-to-world transforms can change when the arm or base moves; do not reuse initial transforms after motion.",
            "missing_data": "Missing or undecodable depth is unavailable measurement, not evidence of empty space.",
        }
    if cache_enabled:
        session._public_observation_cache = (cache_key, copy.deepcopy(result))
    return result


def controller_reply(observation, operation):
    """Label transport completion without implying successful physical motion."""
    return {
        **observation,
        "controller_receipt": {
            "operation": operation,
            "completed_control_steps": 1 if operation == "step" else 0,
            "physics_advanced": operation == "step",
            "control_step_after": observation["control_step"],
            "meaning": (
                "Native action processed for one control tick. This does not establish target "
                "pose attainment, contact, grasp, or task success. Sensors report the measured state."
                if operation == "step" else
                "Current sensors read only; no action sent and physics did not advance."
            ),
        },
    }


class LibraryRuntime:
    """Only the coordinator can select/finish tasks or inspect numeric verdicts."""

    def __init__(self, session, spec, *, timeout=None, max_steps=None, evidence=None):
        self.session = session
        self.spec = spec
        for task in spec["tasks"]:
            goal = GOALS[task["task"]]
            if any(
                task[key] != getattr(goal, key)
                for key in ("fixture", "predicate", "threshold")
            ):
                raise ValueError(
                    "Task predicates differ from this simulator image's implementation"
                )
        self.bindings = bind_fixtures(session.raw)
        self.timeout = timeout or spec["task_timeout_seconds"]
        self.max_steps = max_steps or spec["max_control_steps_per_task"]
        self.evidence = evidence
        self.index = -1
        self.verifier = None
        self.started = None
        self.start_step = None
        self.finished = True
        self.action_digest = hashlib.sha256()

    def start_task(self, index):
        if not self.finished or index != self.index + 1:
            raise ValueError("Tasks must run sequentially, without skipping or overlap")
        task = self.spec["tasks"][index]
        binding = self.bindings[GOALS[task["task"]].fixture]
        self.verifier = TaskVerifier(
            task["task"],
            binding,
            read_positions(self.session.raw, self.bindings),
            settle_steps=self.spec["settle_steps"],
        )
        self.index = index
        self.start_fingerprint = self.session.fingerprint()
        self.started = time.monotonic()
        self.start_step = self.session.steps
        self.finished = False
        return {
            "index": index,
            "instruction": task["instruction"],
            "task": task["task"],
            "round": task["round"],
            "precondition_valid": self.verifier.precondition_valid,
            "time_budget_seconds": self.timeout,
            "control_step_budget": self.max_steps,
            "required_sustained_steps": self.spec["settle_steps"],
            "success_criteria": goal_criteria(
                task["task"],
                ranges=[(joint.low, joint.high) for joint in binding.joints],
                settle_steps=self.spec["settle_steps"],
            ),
            "observation": self.observe(),
        }

    def observe(self):
        result = public_observation(self.session)
        if self.verifier is not None:
            result["task_success"] = self.verifier.verdict()["success"]
        return result

    def bridge(self, message):
        if self.finished or self.verifier is None:
            raise RuntimeError("No active task")
        if time.monotonic() - self.started >= self.timeout:
            raise RuntimeError("Task wall-clock budget exhausted")
        if message == {"operation": "observe"}:
            return controller_reply(self.observe(), "observe")
        if set(message) != {"operation", "action"} or message["operation"] != "step":
            raise ValueError("Only observe and step are available")
        if self.session.steps - self.start_step >= self.max_steps:
            raise RuntimeError("Task control-step budget exhausted")
        self.session.step(message["action"])
        encoded = json.dumps(
            message["action"], allow_nan=False, separators=(",", ":")
        ).encode()
        self.action_digest.update(encoded + b"\n")
        positions = read_positions(self.session.raw, self.bindings)
        error = None
        try:
            verdict = self.verifier.update(positions, self.session.steps)
        except ValueError as exc:
            error = exc
            verdict = self.verifier.verdict()
        if self.evidence is not None:
            self.evidence.write(
                json.dumps(
                    {
                        "task_index": self.index,
                        "control_step": self.session.steps,
                        "action": message["action"],
                        "elapsed_seconds": time.monotonic() - self.started,
                        "positions": {
                            k: v if math.isfinite(v) else str(v)
                            for k, v in positions.items()
                        },
                        "success": verdict["success"],
                        "verification_error": str(error) if error else None,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            self.evidence.flush()
        if error is not None:
            raise error
        return controller_reply(self.observe(), "step")

    def finish_task(self, *, code_errors=0):
        if self.finished:
            raise ValueError("Task already finished")
        self.finished = True
        task = self.spec["tasks"][self.index]
        verdict = self.verifier.verdict()
        if self.session.phase != "active":
            verdict.update(success=False, status="simulator_fault")
        return {
            "index": self.index,
            "round": task["round"],
            "pair_key": task["pair_key"],
            "task": task["task"],
            "elapsed_seconds": time.monotonic() - self.started,
            "instruction": task["instruction"],
            "control_steps": self.session.steps - self.start_step,
            "start_fingerprint": self.start_fingerprint,
            "end_fingerprint": self.session.fingerprint(),
            "code_errors": code_errors,
            "precondition_valid": verdict["precondition_valid"],
            "observed_success": verdict["success"],
            "verdict": verdict,
            "cumulative_action_sha256": self.action_digest.hexdigest(),
        }

    def metadata(self):
        return {
            "config": self.session.config,
            "initial_fingerprint": self.session.initial,
            "spec_sha256": self.spec["sha256"],
            "spec": self.spec,
            "bindings": {
                name: binding.to_dict() for name, binding in self.bindings.items()
            },
            "observation_policy": "RGB and explicitly whitelisted robot proprioception; simulated world poses are exposed as localization",
        }
