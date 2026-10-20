"""Small, owner-thread runtime shared by commissioning and the local panel."""

import mujoco
import numpy as np

from . import register
from .build import FINGERS


def make_environment(kitchen=False, control_freq=20, camera_observations=True):
    """Build the commissioning room or kitchen. ``control_freq`` sets robosuite's
    command rate (Hz); physics stays at the installed 0.002 s timestep.

    ``camera_observations=False`` (kitchen) skips robosuite's per-step robot
    camera observations and its GLFW offscreen context, which macOS allows only
    on the main thread. The same MJCF cameras remain renderable with
    ``mujoco.Renderer``.
    """
    register()
    if kitchen and not camera_observations:
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        env = robosuite.make(
            "MichaelCommissioningKitchen",
            robots="MichaelOpenArmWuji",
            controller_configs=load_composite_controller_config(
                controller=None, robot="MichaelOpenArmWuji"
            ),
            seed=0,
            layout_ids=1,
            style_ids=1,
            initialization_noise=None,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            ignore_done=True,
            control_freq=control_freq,
            renderer="mujoco",
        )
        env.rng = np.random.default_rng(0)
        env.reset()
        return env
    if kitchen:
        from robot_session.environment import RobotEnvironment

        wrapper = RobotEnvironment(
            "robocasa/MichaelCommissioningKitchen",
            robot="MichaelOpenArmWuji",
            split=None,
            seed=0,
            layout_ids=1,
            style_ids=1,
            initialization_noise=None,
            camera_widths=64,
            camera_heights=64,
            control_freq=control_freq,
        )
        wrapper.reset(seed=0)
        return wrapper.env
    import robosuite

    env = robosuite.make(
        "MichaelCommissioning",
        robots="MichaelOpenArmWuji",
        initialization_noise=None,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=control_freq,
        ignore_done=True,
    )
    env.reset()
    return env


def tactile(robot):
    result = {}
    for side, gripper in robot.gripper.items():
        result[side] = {
            finger: {
                "normal_N": float(
                    robot.get_sensor_measurement(
                        gripper.correct_naming(f"{finger}_touch")
                    )[0]
                ),
                "force_N": robot.get_sensor_measurement(
                    gripper.correct_naming(f"{finger}_force")
                ).tolist(),
                "torque_Nm": robot.get_sensor_measurement(
                    gripper.correct_naming(f"{finger}_torque")
                ).tolist(),
            }
            for finger in FINGERS
        }
    return result


def describe(env):
    robot = env.robots[0]
    composite = robot.composite_controller
    low, high = composite.action_limits
    parts = {}
    for name, controller in composite.part_controllers.items():
        start, end = composite._action_split_indexes[name]
        qpos = env.sim.data.qpos[controller.qpos_index]
        units = (
            "radians"
            if name in ("right", "left", "right_gripper", "left_gripper")
            else "metres"
            if name == "torso"
            else "normalized velocity"
        )
        parts[name] = {
            "slice": [start, end],
            "joints": controller.joint_names,
            "units": units,
            "low": low[start:end].tolist(),
            "high": high[start:end].tolist(),
            "qpos": qpos.tolist(),
        }
    return {
        "robot": "MichaelOpenArmWuji",
        "action_dimension": len(low),
        "parts": parts,
        "finger_order": list(FINGERS),
        "control_frequency_hz": env.control_freq,
        "tactile": tactile(robot),
        "time": float(env.sim.data.time),
        "tactile_semantics": "Simulated normal contact force at manufacturer fingertip pads. Force/torque vectors are in each pad's local frame and include pad dynamics; no taxel/firmware emulation.",
        "base_semantics": "Planar Omron abstraction, velocity commands in current base frame; x/y = action m/s, yaw = action * 1.5 rad/s. No wheel, balance or tipping dynamics.",
    }


def fingertip_sensors(env):
    """Expose the same tactile channels through the persistent session API."""
    values, descriptions = {}, {}
    for index, robot in enumerate(env.robots):
        if type(robot.robot_model).__name__ != "MichaelOpenArmWuji":
            continue
        for side, gripper in robot.gripper.items():
            for kind, units in (
                ("touch", "newtons"),
                ("force", "newtons"),
                ("torque", "newton-metres"),
            ):
                key = f"robot{index}_{side}_fingertip_{kind}"
                names = [gripper.correct_naming(f"{f}_{kind}") for f in FINGERS]
                values[key] = (
                    np.array([robot.get_sensor_measurement(n) for n in names])
                    .ravel()
                    .tolist()
                )
                descriptions[key] = {
                    "units": units,
                    "finger_order": list(FINGERS),
                    "sensor_names": names,
                    "components_per_finger": 1 if kind == "touch" else 3,
                    "frame": "scalar normal contact force"
                    if kind == "touch"
                    else "each fingertip pad's local XYZ frame",
                    "includes": "contact normal forces at manufacturer Wuji Hand 2 Beta 2 pads"
                    if kind == "touch"
                    else "pad transmitted load, including contact, gravity, and acceleration",
                    "simulation_model": "MuJoCo ideal transducers on actual pad bodies; not hardware firmware/taxel emulation",
                }
    return values, descriptions


class Camera:
    def __init__(self, env, width=960, height=720):
        self.env = env
        model = env.sim.model._model
        model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(height, model.vis.global_.offheight)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 2.4
        self.camera.azimuth = 140
        self.camera.elevation = -16
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = [0, 1, 0, 0, 0, 0]
        self.options.sitegroup[:] = 0
        self.hidden = {
            i
            for i in range(model.ngeom)
            if hasattr(env, "layout_id")
            and (
                model.geom(i).name.startswith("wall_")
                or "ceiling" in model.geom(i).name
            )
        }
        self.set_view("overview")

    def set_view(self, view):
        self.view = view if view in ("overview", "right", "left") else "overview"
        robot = self.env.robots[0]
        model, data = self.env.sim.model._model, self.env.sim.data._data
        if view in ("right", "left"):
            site = model.site(robot.gripper[view].important_sites["grip_site"]).id
            pad = model.site(
                robot.gripper[view].correct_naming("middle_finger_tactile_site")
            ).id
            self.camera.lookat[:] = (data.site_xpos[site] + data.site_xpos[pad]) / 2
            self.camera.distance = 0.48
            normal = data.site_xmat[pad].reshape(3, 3)[:, 1] * (
                1 if view == "right" else -1
            )
            self.camera.azimuth = np.degrees(np.arctan2(normal[1], normal[0]))
            self.camera.elevation = -np.degrees(
                np.arctan2(normal[2], np.linalg.norm(normal[:2]))
            )
        else:
            body = model.body(robot.robot_model.base.correct_naming("base")).id
            self.camera.lookat[:] = data.xpos[body] + np.array([0.05, 0, 0.77])
            self.camera.distance = 2.4
            self.camera.azimuth = 140
            self.camera.elevation = -16

    def render(self):
        self.renderer.update_scene(
            self.env.sim.data._data, camera=self.camera, scene_option=self.options
        )
        # Human cutaway only. Geometry, collisions, and robot cameras are intact.
        for geom in self.renderer.scene.geoms[: self.renderer.scene.ngeom]:
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in self.hidden:
                geom.rgba[3] = 0
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()
