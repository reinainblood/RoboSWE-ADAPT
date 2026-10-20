"""Local reset compatibility for stock GR1's unactuated base; no upstream edits."""
import numpy as np


def install_stock_gr1_placement():
    from robocasa.utils import env_utils as eu
    from scipy.spatial.transform import Rotation
    original_base, original_position = eu.set_robot_base, eu.set_robot_to_position

    def applicable(env):
        return type(env.robots[0].robot_model).__name__ == 'GR1' and 'mobilebase0_joint_mobile_yaw' not in env.sim.model.joint_names

    def position(env, global_pos):
        if not applicable(env):return original_position(env,global_pos)
        model=env.sim.model._model
        root=model.body(env.robots[0].robot_model.root_body).id
        if model.body_parentid[root]!=0:raise ValueError('GR1 root is not a world child')
        # set_base_xpos normally subtracts bottom_offset before compilation.
        model.body_pos[root] = np.asarray(global_pos)-env.robots[0].robot_model.bottom_offset
        env.sim.forward()

    def spawn(env,anchor_pos,anchor_ori,rot_dev,pos_dev_x,pos_dev_y):
        if not applicable(env):return original_base(env,anchor_pos,anchor_ori,rot_dev,pos_dev_x,pos_dev_y)
        model=env.sim.model._model
        root=model.body(env.robots[0].robot_model.root_body).id
        yaw=env.rng.uniform(-rot_dev,rot_dev)
        euler=np.asarray(anchor_ori)+np.array([0,0,yaw])
        model.body_quat[root]=Rotation.from_euler('xyz',euler).as_quat()[[3,0,1,2]]
        for attempt in range(100):
            candidate=eu.generate_random_robot_pos(env,anchor_pos,anchor_ori,pos_dev_x+(attempt//25)*.10,pos_dev_y+(attempt//25)*.05)
            position(env,candidate)
            if not eu.detect_robot_collision(env):return candidate
        raise RuntimeError('No collision-free stock GR1 placement after 100 samples')

    eu.set_robot_base=spawn
    eu.set_robot_to_position=position
