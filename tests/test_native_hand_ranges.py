from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from robot_session.environment import RobotEnvironment
from robot_session.interface import raw_hand_command_bounds


def hand():
    cls=type('FourierRightHand',(),{})
    g=cls();g.dof=6
    g.grasp_qpos={-1:np.array([-1.5,-1.5,-1.5,-1.5,-3.,3.]),1:np.array([1.5,1.5,1.5,1.5,3.,3.])}
    g.format_action=lambda action:np.asarray(action)[[0,0,1,1,2,2,3,3,4,4,5]]
    return g


def controller(scaling=False):
    return SimpleNamespace(use_action_scaling=scaling,actuator_min=np.zeros(11),
        actuator_max=np.array([1.74,1.57]*4+[1.23,1.22,1.74]))


def test_native_teleop_targets_and_actuator_range_are_covered():
    g=hand();c=controller();low,high=raw_hand_command_bounds(g,c)
    for target in g.grasp_qpos.values():assert np.all(target>=low) and np.all(target<=high)
    np.testing.assert_allclose(high,[1.74]*4+[3,3])
    assert c.use_action_scaling is False
    assert raw_hand_command_bounds(g,controller(True)) is None
    assert raw_hand_command_bounds(SimpleNamespace(),c) is None


def test_only_hand_slices_expand_and_commands_are_not_reinterpreted():
    g=hand();c=controller()
    composite=SimpleNamespace(action_limits=(np.full(18,-1.),np.full(18,1.)),
        get_action_info_dict=lambda:{'Action Dimension':18,'right_gripper':[12,18]})
    robot=SimpleNamespace(gripper={'right':g},part_controllers={'right_gripper':c},composite_controller=composite)
    env=object.__new__(RobotEnvironment);env.env=SimpleNamespace(robots=[robot])
    low=np.r_[np.full(12,-np.inf),np.full(6,-1.)];high=np.r_[np.full(12,np.inf),np.full(6,1.)]
    env.action_space=gym.spaces.Box(low,high,dtype=np.float64)
    action=np.r_[np.zeros(12),g.grasp_qpos[1]]
    assert not env.action_space.contains(action)
    changes=env.enable_native_hand_ranges()
    assert env.action_space.contains(action)
    np.testing.assert_array_equal(env.action_space.low[:12],low[:12])
    np.testing.assert_array_equal(env.action_space.high[:12],high[:12])
    np.testing.assert_array_equal(action,np.r_[np.zeros(12),g.grasp_qpos[1]])
    assert changes['0/right_gripper']['slice']==[12,18]
    assert c.use_action_scaling is False
