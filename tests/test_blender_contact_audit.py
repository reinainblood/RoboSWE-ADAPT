from pairing.blender_contact_audit import summarize_contacts


def test_contact_release_requires_previous_contact_and_closed_lid():
    def row(step,left,right,lid):
        return {'step':step,'fixture_diagnostics':{'blender':{
            'lid_on_blender':lid,'audit_button_contacts':{'left':left,'right':right}}}}
    result=summarize_contacts([row(0,False,False,True),row(1,True,False,True),
        row(2,False,True,True),row(3,False,False,False),row(4,True,False,False),
        row(5,False,False,False)])
    assert result['samples']==6
    assert result['left_contact_samples']==2
    assert result['right_contact_samples']==1
    assert result['left_release_events_with_lid_closed']==[2]
    assert result['right_release_events_with_lid_closed']==[]
