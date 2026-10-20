from types import SimpleNamespace

from pairing.overnight_suite import fixture_diagnostics


def test_fixture_diagnostics_are_detached_read_only_and_failure_tolerant():
    state={'lid_on_blender':False,'turned_on':False}
    def bad_sink(env):raise ValueError('unsupported diagnostic')
    fixture=SimpleNamespace(get_state=lambda:state)
    raw=SimpleNamespace(blender=fixture,sink=SimpleNamespace(get_handle_state=bad_sink))
    result=fixture_diagnostics(raw)
    assert result['blender']==state
    result['blender']['turned_on']=True
    assert state['turned_on'] is False
    assert result['sink']=={'diagnostic_unavailable':'ValueError'}
    assert not hasattr(raw,'fixture_diagnostics')


def test_missing_fixture_does_not_invent_a_state():
    assert fixture_diagnostics(SimpleNamespace())=={}


def test_generic_door_joint_getter_is_supported():
    fixture=SimpleNamespace(door_joint_names=['hinge'])
    raw=SimpleNamespace(fxtr=fixture)
    def joints(env,names):
        assert env is raw and names==['hinge']
        return {'hinge':.25}
    fixture.get_joint_state=joints
    assert fixture_diagnostics(raw)=={'fxtr':{'hinge':.25}}
