"""Opt-in Panda hand fitting experiment; leaves upstream and benchmark defaults alone."""


def register_panda_inspire_hand():
    import numpy as np
    import robosuite.utils.transform_utils as transforms
    from robosuite.models.grippers import register_gripper
    from robosuite.models.grippers.inspire_hands import InspireRightHand
    from robosuite.models.robots.compositional import PandaDexRH
    from robosuite.utils.mjcf_utils import find_elements, string_to_array

    @register_gripper
    class PandaInspireRightHand(InspireRightHand):
        """Inspire hand using upstream PandaDexRH's documented mount transform.

        Six normalized commands control four finger curls, thumb curl, and thumb
        opposition. PandaOmron's scaling GRIP controller maps the expanded commands
        into the hand's actual actuator ranges.
        """

        def __init__(self, idn=0):
            super().__init__(idn=idn)
            # These properties do not depend on an instantiated Panda model.
            position = PandaDexRH.gripper_mount_pos_offset.fget(None)["right"]
            quaternion = PandaDexRH.gripper_mount_quat_offset.fget(None)["right"]
            body = self.worldbody.find("body")
            body.set("pos", " ".join(map(str, position)))
            body.set("quat", " ".join(map(str, quaternion)))
            eef = find_elements(
                self.root,
                tags="body",
                attribs={"name": self.correct_naming("eef")},
                return_first=True,
            )
            relative = string_to_array(eef.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]
            self.rotation_offset = transforms.quat_multiply(
                relative, np.asarray(quaternion)[[1, 2, 3, 0]]
            )

    return PandaInspireRightHand
