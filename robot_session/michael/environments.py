"""Empty commissioning environments: no task, reward, or success claim."""

from xml.etree import ElementTree as ET

from robocasa.environments.kitchen.kitchen import Kitchen
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask


class MichaelCommissioning(ManipulationEnv):
    def _load_model(self):
        super()._load_model()
        arena = EmptyArena()
        arena.set_origin([0, 0, 0])
        self.robots[0].robot_model.set_base_xpos([0, 0, 0])
        probe = ET.SubElement(
            arena.worldbody, "body", name="tactile_probe", mocap="true", pos="2 0 1"
        )
        ET.SubElement(
            probe,
            "geom",
            name="tactile_probe_geom",
            type="sphere",
            size="0.008",
            rgba="0.9 0.28 0.12 1",
            friction="1 0.01 0.001",
            group="1",
        )
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[],
        )
        self.model.root.find("option").set("integrator", "implicitfast")
        self.model.root.find("option").set("cone", "elliptic")

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False


class MichaelCommissioningKitchen(Kitchen):
    """RoboCasa's full kitchen scene, without task objects or objectives."""

    def _reset_internal(self):
        self._michael_settling = True
        try:
            super()._reset_internal()
        finally:
            self._michael_settling = False

    def _pre_action(self, action, policy_step=False):
        if getattr(self, "_michael_settling", False):
            action = self.robots[0].create_action_vector({})
        super()._pre_action(action, policy_step)

    def get_ep_meta(self):
        return {
            **super().get_ep_meta(),
            "lang": "Michael robot commissioning: basic motion and tactile inspection; no task.",
        }
