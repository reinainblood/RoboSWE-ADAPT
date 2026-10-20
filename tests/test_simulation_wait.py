from types import SimpleNamespace
import pytest

from pairing.simulation_wait import hold_action, passive_wait, validate_wait_steps


def schema():
    return {'shape':[17], 'controllers':[{'controller':'HybridMobileBase', 'parts':{
        'right':[0,6], 'right_gripper':[6,12], 'base':[12,15], 'torso':[15,16], 'base_mode':[16,17]},
        'parameters':{'right':{'input_type':'delta'},'base':{'class':'LegacyMobileBaseJointVelocityController'},'torso':{'input_type':'delta'}}}]}


def test_wait_holds_digits_but_does_not_repeat_motion():
    session=SimpleNamespace(env=SimpleNamespace(action_schema=schema))
    previous=[.5]*17
    result=hold_action(session,previous)
    assert result[:6]==[0]*6 and result[6:12]==[.5]*6 and result[12:]==[0]*5
    assert previous==[.5]*17
    with pytest.raises(ValueError):hold_action(session,None)
    altered=schema();altered['controllers'][0]['parameters']['right']['input_type']='absolute'
    session.env.action_schema=lambda:altered
    with pytest.raises(ValueError):hold_action(session,previous)


@pytest.mark.parametrize('steps',[0,-1,601,True,1.5,'500'])
def test_wait_bounds(steps):
    with pytest.raises(ValueError):validate_wait_steps(steps)


def test_passive_physics_preserves_work_budget_and_records_each_tick():
    clock=[100.]
    runtime=SimpleNamespace(started=90.,finished=False,timeout=20,max_steps=6000,
        session=SimpleNamespace(steps=0,env=SimpleNamespace(action_schema=schema)))
    actions=[];recorded=[]
    def bridge(message):
        actions.append(message['action']);runtime.session.steps+=1;clock[0]+=2
        return {'control_step':runtime.session.steps,'task_success':False,'simulator_phase':'active'}
    runtime.bridge=bridge
    result=passive_wait(runtime,[.5]*17,10,on_step=lambda:recorded.append(runtime.session.steps),clock=lambda:clock[0])
    assert result['control_step']==10
    assert clock[0]-runtime.started==10 # The original ten seconds of work remain charged.
    assert result['passive_wait']['host_seconds']==20
    assert recorded==list(range(1,11))
    assert all(a==[0]*6+[.5]*6+[0]*5 for a in actions)
    clock[0]+=11 # Thinking still spends the normal work budget.
    with pytest.raises(ValueError,match='budget exhausted'):
        passive_wait(runtime,[.5]*17,1,clock=lambda:clock[0])


def test_wait_cannot_escape_episode_allowance():
    with pytest.raises(ValueError):validate_wait_steps(2,used=1199)
    validate_wait_steps(1,used=1199)


def test_wait_still_counts_control_steps_and_stops_on_success():
    clock=[1.]
    runtime=SimpleNamespace(started=0.,finished=False,timeout=20,max_steps=3,
        session=SimpleNamespace(steps=2,env=SimpleNamespace(action_schema=schema)))
    with pytest.raises(ValueError,match='control-step budget'):
        passive_wait(runtime,[0]*17,2,clock=lambda:clock[0])
    assert runtime.session.steps==2
    def bridge(message):
        runtime.session.steps+=1;clock[0]+=.1
        return {'control_step':runtime.session.steps,'task_success':True,'simulator_phase':'active'}
    runtime.max_steps=6000;runtime.bridge=bridge
    result=passive_wait(runtime,[0]*17,10,clock=lambda:clock[0])
    assert runtime.session.steps==3 and result['passive_wait']['advanced_steps']==1


def test_robot_specific_hold_preserves_absolute_pose_targets():
    previous=[.5]*24
    expected=[.2]*12+[.5]*12
    session=SimpleNamespace(passive_hold_action=lambda a:expected if a is previous else None)
    assert hold_action(session,previous)==expected


def test_panda_wait_uses_zero_increment_to_preserve_grip():
    data=schema();data['controllers'][0]['parameters']['right_gripper']={'gripper_class':'PandaGripper'}
    session=SimpleNamespace(env=SimpleNamespace(action_schema=lambda:data))
    assert hold_action(session,[1.]*17)==[0.]*17
