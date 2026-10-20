"""Public hardware/controller metadata; no task geometry or motion routines."""

import numpy as np


def raw_hand_command_bounds(gripper, controller):
    """Bounds covering native Fourier teleop targets without changing scaling.

    Upstream Device.input2action deliberately does not clip grasp_qpos to the
    controller's normalized sampling bounds when hand scaling is disabled.
    """
    if type(gripper).__name__ not in {"FourierLeftHand", "FourierRightHand"}:
        return None
    if getattr(controller, "use_action_scaling", True):
        return None
    dof = gripper.dof
    mapping = np.asarray(gripper.format_action(np.arange(dof, dtype=float)))
    if not np.all(np.isfinite(mapping)) or not np.all(mapping == mapping.astype(int)):
        raise ValueError("Unsupported non-indexed Fourier hand formatter")
    mapping = mapping.astype(int)
    actuator_low = np.asarray(controller.actuator_min)
    actuator_high = np.asarray(controller.actuator_max)
    examples = np.asarray(list(gripper.grasp_qpos.values()), dtype=float)
    if examples.ndim != 2 or examples.shape[1] != dof or mapping.shape != actuator_low.shape or actuator_low.shape != actuator_high.shape:
        raise ValueError("Fourier hand/controller dimensions differ")
    low, high = [], []
    for channel in range(dof):
        selected = mapping == channel
        if not selected.any():
            raise ValueError("Unmapped hand input channel")
        low.append(min(float(actuator_low[selected].min()), float(examples[:, channel].min())))
        high.append(max(float(actuator_high[selected].max()), float(examples[:, channel].max())))
    if not np.isfinite(low + high).all():
        raise ValueError("Nonfinite native hand limits")
    return np.asarray(low), np.asarray(high)


def numeric(value):
    if isinstance(value, str) or value is None:
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        return scalar if not isinstance(scalar, float) or np.isfinite(scalar) else None
    return [numeric(v) for v in array]


def controller_parameters(composite):
    result = {}
    allowed = (
        "input_type",
        "input_ref_frame",
        "impedance_mode",
        "input_min",
        "input_max",
        "output_min",
        "output_max",
        "control_dim",
        "kp",
        "kd",
        "damping_ratio",
    )
    for name, controller in getattr(composite, "part_controllers", {}).items():
        parameters = {
            key: numeric(getattr(controller, key))
            for key in allowed
            if hasattr(controller, key)
        }
        parameters["class"] = type(controller).__name__
        joint_limits = []
        for joint in getattr(controller, "joint_names", ()):
            jid = composite.sim.model.joint_name2id(joint)
            limited = bool(composite.sim.model.jnt_limited[jid])
            joint_limits.append(
                {
                    "name": joint,
                    "limited": limited,
                    "range": numeric(composite.sim.model.jnt_range[jid])
                    if limited
                    else None,
                    "units": "metres for sliding joints; radians for hinge joints",
                }
            )
        if joint_limits:
            parameters["robot_joint_limits"] = joint_limits
        if name in getattr(composite, "arms", ()) and hasattr(
            composite, "get_controller_base_pose"
        ):
            position, rotation = composite.get_controller_base_pose(name)
            parameters["base_frame_position_world"] = numeric(position)
            parameters["base_frame_rotation_world"] = numeric(rotation)
        gripper = getattr(composite, "grippers", {}).get(name)
        if gripper is not None:
            parameters["gripper_class"] = type(gripper).__name__
            parameters["bounds_apply_after_gripper_formatting"] = True
            if type(gripper).__name__ == "PandaGripper":
                parameters["native_gripper_command"] = (
                    "-1 opens, +1 closes, 0 retains the current formatted gripper command"
                )
                site = composite.sim.model.site_name2id(
                    gripper.important_sites["grip_site"]
                )
                site_rotation = composite.sim.data.site_xmat[site].reshape(3, 3)
                joints = []
                for joint in gripper.joints:
                    jid = composite.sim.model.joint_name2id(joint)
                    body_rotation = composite.sim.data.xmat[
                        composite.sim.model.jnt_bodyid[jid]
                    ].reshape(3, 3)
                    axis = (
                        site_rotation.T
                        @ body_rotation
                        @ composite.sim.model.jnt_axis[jid]
                    )
                    joints.append(
                        {
                            "name": joint,
                            "slide_axis_in_grip_site": axis.tolist(),
                            "travel_range_metres": composite.sim.model.jnt_range[
                                jid
                            ].tolist(),
                        }
                    )
                parameters["grip_site_geometry"] = {
                    "approach_axis": [0, 0, 1],
                    "finger_joints": joints,
                    "note": "Panda fingers extend along grip-site +Z and open/close along its X line, not the legacy hand-body Y convention.",
                }
        if type(controller).__name__ == "OperationalSpaceController":
            parameters["pose_order"] = [
                "dx" if getattr(controller, "input_type", None) == "delta" else "x",
                "dy" if getattr(controller, "input_type", None) == "delta" else "y",
                "dz" if getattr(controller, "input_type", None) == "delta" else "z",
                "rotation_vector_x",
                "rotation_vector_y",
                "rotation_vector_z",
            ]
            parameters["output_units"] = [
                "metres",
                "metres",
                "metres",
                "radians",
                "radians",
                "radians",
            ]
            if not getattr(controller, "control_ori", True):
                parameters["pose_order"] = parameters["pose_order"][:3]
                parameters["output_units"] = parameters["output_units"][:3]
        if type(controller).__name__ in {
            "MobileBaseJointVelocityController",
            "LegacyMobileBaseJointVelocityController",
        }:
            parameters["command_frame"] = (
                "current mobile-base frame; the controller transforms XY into its initial-base joint frame"
            )
        result[name] = parameters
    return result


def camera_calibration(sim, camera, height, width):
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    return {
        "camera": camera,
        "width": width,
        "height": height,
        "intrinsic_matrix": get_camera_intrinsic_matrix(
            sim, camera, height, width
        ).tolist(),
        "camera_to_world": get_camera_extrinsic_matrix(sim, camera).tolist(),
        "camera_axes": "OpenCV: x right, y down, z forward",
        "pixel_axes": "column right, row down; top-left origin",
        "world_units": "metres",
        "depth_units": "optical-axis distance in metres",
    }


def metric_depth(sim, depth):
    values = np.asarray(depth)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Depth sensor contains invalid normalized values")
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    if not 0 < near < far:
        raise ValueError("Invalid camera clipping planes")
    return (
        np.asarray(near / (1 - values * (1 - near / far)), dtype=np.float32).squeeze(-1)
        if values.ndim == 3
        else np.asarray(near / (1 - values * (1 - near / far)), dtype=np.float32)
    )


def wrist_sensors(raw):
    """Only transducers explicitly belonging to the selected robot grippers."""
    import mujoco

    values, descriptions = {}, {}
    for index, robot in enumerate(raw.robots):
        for arm, gripper in robot.gripper.items():
            for kind, units in (
                ("force_ee", "newtons"),
                ("torque_ee", "newton-metres"),
            ):
                sensor = gripper.important_sensors.get(kind)
                if sensor is None:
                    continue
                key = f"robot{index}_{arm}_eef_{'force' if kind == 'force_ee' else 'torque'}"
                values[key] = np.asarray(robot.get_sensor_measurement(sensor)).tolist()
                descriptions[key] = {
                    "units": units,
                    "frame": "native force/torque sensor-local frame",
                    "includes": "transmitted force/torque, including dynamics; not a labeled object-contact signal",
                }
                sensor_id = raw.sim.model.sensor_name2id(sensor)
                if raw.sim.model.sensor_objtype[sensor_id] == int(
                    mujoco.mjtObj.mjOBJ_SITE
                ):
                    site_id = raw.sim.model.sensor_objid[sensor_id]
                    descriptions[key]["sensor_position_world"] = raw.sim.data.site_xpos[
                        site_id
                    ].tolist()
                    descriptions[key]["sensor_rotation_world"] = (
                        raw.sim.data.site_xmat[site_id].reshape(3, 3).tolist()
                    )
    return values, descriptions
