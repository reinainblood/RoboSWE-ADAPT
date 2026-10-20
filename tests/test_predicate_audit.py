import json
from pathlib import Path
import pytest
from roboswe.library.predicate_audit import predicate_contract, audit_job, comparison_status
from roboswe.review import episode, atif, export


def test_blender_contract_is_source_bound():
    root=Path('upstream/robocasa/robocasa')
    if not root.exists(): pytest.skip('Requires local upstream checkout')
    c=predicate_contract('BlendIngredients',root)
    assert len(c['sources'])==2
    assert 'fruit_in_blender and blender_state' in c['sources'][0]['source']
    assert {r['id'] for r in c['rubric']}=={'fruit_inside','lid_closed','activation'}
    assert 'RELEASE' in c['rubric'][2]['description']


def test_unknown_predicate_fails_closed(tmp_path):
    with pytest.raises(ValueError): predicate_contract('Invented',tmp_path)


@pytest.mark.parametrize('visible,mechanical,expected', [
    (False, False, 'consistent_failure'),
    (True, True, 'consistent_success'),
    (False, True, 'visual_mechanical_disagreement'),
    (True, False, 'visual_mechanical_disagreement'),
    (None, True, 'not_observable'),
])
def test_visual_failure_does_not_imply_a_broken_mechanical_verifier(visible, mechanical, expected):
    assert comparison_status(visible, mechanical) == expected


def test_running_cannot_trigger_judge(tmp_path):
    (tmp_path/'job.json').write_text(json.dumps({'status':'running'}))
    with pytest.raises(ValueError,match='finalized'): audit_job(tmp_path)


def test_export_preserves_reasoning_and_actions(tmp_path):
    job='20261009T221300Z-0daa0ac68c';p=tmp_path/'jobs'/job;p.mkdir(parents=True)
    (p/'job.json').write_text(json.dumps({'id':job,'name':'Test','status':'completed','finished_at':'now'}))
    events=[{'event':'request','turn':0,'prompt':'{"observation":{"control_step":0}}'},
            {'event':'provider_thinking','turn':0,'trace':'exposed thinking'},
            {'event':'decision','turn':0,'decision':{'operation':'observe','message':'look','path':'','content':''}}]
    (p/'model-transcript.jsonl').write_text('\n'.join(map(json.dumps,events)))
    (p/'actions.jsonl').write_text('{"control_step":1,"action":[0]}\n')
    d=episode(p)
    assert d['turns'][0]['reasoning']==['exposed thinking']
    assert d['events']['actions.jsonl'][0]['control_step']==1
    a=atif(d);assert [s['step_id'] for s in a['steps']]==[1,2]
    assert a['steps'][1]['reasoning_content']=='["exposed thinking"]'
    out=tmp_path/'export';export(tmp_path/'jobs',out,[job])
    assert json.loads((out/'jobs'/job/'episode.json').read_text())==d


def test_symlinked_evidence_not_exported(tmp_path):
    secret=tmp_path/'private';secret.write_text('{"secret":"dont export"}')
    p=tmp_path/'job';p.mkdir();(p/'actions.jsonl').symlink_to(secret)
    assert 'actions.jsonl' not in episode(p)['events'] or episode(p)['events']['actions.jsonl']==[]


def test_toast_contract_covers_native_conditions_without_claiming_cooking():
    import hashlib
    root=Path('upstream/robocasa/robocasa')
    if not root.exists():pytest.skip('Requires local upstream checkout')
    contract=predicate_contract('ToastBagel',root)
    rubric={r['id']:r['description'] for r in contract['rubric']}
    assert set(rubric)=={'bagel_rack_contact','door_closed','timer_set','gripper_withdrawn'}
    assert 'Either lower or upper' in rubric['bagel_rack_contact']
    assert '<=0.005' in rubric['door_closed']
    assert '>=0.1' in rubric['timer_set'] and 'NOT a test of completed heating' in rubric['timer_set']
    assert '0.25 metres' in rubric['gripper_withdrawn']
    assert all('null' in text for text in rubric.values())
    assert len(contract['sources'])==4
    for source in contract['sources']:
        assert source['sha256']==hashlib.sha256((root/source['path']).read_bytes()).hexdigest()
        for excerpt in source.get('excerpts',[]):
            text='\n'.join((root/source['path']).read_text().splitlines()[excerpt['line_start']-1:excerpt['line_end']])
            assert text.strip() in source['source']
    assert 'timer_val >= 0.1' in contract['sources'][0]['source']
    assert 'bagel_on_lower or bagel_on_upper' in contract['sources'][0]['source']


def test_toast_contract_fails_closed_if_reviewed_helper_missing(tmp_path):
    taskdir=tmp_path/'environments/kitchen';taskdir.mkdir(parents=True)
    (taskdir/'task.py').write_text('class ToastBagel:\n def _check_success(self): return True\n')
    fixtures=tmp_path/'models/fixtures';fixtures.mkdir(parents=True)
    (fixtures/'toaster_oven.py').write_text('class ToasterOven: pass\n')
    with pytest.raises(ValueError,match='dependency is missing'):
        predicate_contract('ToastBagel',tmp_path)
