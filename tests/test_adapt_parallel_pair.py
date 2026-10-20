import json
from pathlib import Path
import pytest
from pairing.adapt_parallel_pair import child_command,child_result,validate_pair_configs


def configs(root):
    for provider in ('openrouter','anthropic'):
        (root/(provider+'-toast-modal.json')).write_text(json.dumps({'participant':{'provider':provider},'execution':{'backend':'modal','modal_environment':'FDR','code_image':'same'},'limits':{'per_script_seconds':90},'trials':5}))


def test_pair_enforces_matching_execution_and_five_serial_trials(tmp_path):
    configs(tmp_path);assert len(validate_pair_configs(tmp_path))==2
    p=tmp_path/'anthropic-toast-modal.json';value=json.loads(p.read_text());value['limits']['per_script_seconds']=60;p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='must match'):validate_pair_configs(tmp_path)


def test_child_commands_are_distinct_and_one_series_bound(tmp_path):
    a=child_command(tmp_path,'openrouter','python');b=child_command(tmp_path,'anthropic','python')
    assert a!=b and a[-2:]==b[-2:]==['--max-series','1']
    assert a[a.index('--state-dir')+1]!=b[b.index('--state-dir')+1]


def test_child_failure_never_counts_completed(tmp_path):
    p=tmp_path/'openrouter-toast-modal';p.mkdir()
    (p/'active.json').write_text(json.dumps({'status':'stopped_child_result','child_status':'verification_invalid'}))
    assert child_result(tmp_path,'openrouter',0)['classification']=='blocked_infrastructure'
    assert child_result(tmp_path,'anthropic',1)['status']=='missing_receipt'


def test_predecessor_gate_does_not_launch_child_or_resolve_keys(tmp_path,monkeypatch):
    from pairing.adapt_parallel_pair import main
    configs(tmp_path);folder=tmp_path/'modal-toast-parallel-pair';folder.mkdir();(folder/'STOP').write_text('stop')
    monkeypatch.setattr('pairing.adapt_parallel_pair.pid_live',lambda pid:True)
    monkeypatch.setattr('pairing.adapt_parallel_pair.subprocess.Popen',lambda *a,**kw:pytest.fail('Must not launch before predecessor ends'))
    monkeypatch.setattr('pairing.adapt_parallel_pair.subprocess.run',lambda *a,**kw:pytest.fail('No key lookup before predecessor ends'))
    main(['--root',str(tmp_path),'--predecessor-pid','42','--predecessor-job','missing'])
    state=json.loads((folder/'active.json').read_text())
    assert state['children']==[] and state['status']=='stopped'


def test_completed_label_with_partial_trials_is_not_complete(tmp_path):
    lane=tmp_path/'openrouter-toast-modal';lane.mkdir();series=tmp_path/'series';series.mkdir()
    (series/'job.json').write_text(json.dumps({'status':'completed_unscored'}))
    (series/'report.json').write_text(json.dumps({'trials':[{'termination_reason':'task_deadline'}]}))
    (lane/'active.json').write_text(json.dumps({'status':'completed','completed':[str(series)]}))
    assert child_result(tmp_path,'openrouter',0)['classification']=='blocked_infrastructure'


def recovery_fixture(tmp_path):
    import hashlib
    from pairing.adapt_parallel_pair import COUNT_TRANSPORT_ERROR
    parent_id='20261020T091224Z-d45d77bdd7';child_id='20261020T102156Z-a1e31d7462'
    parent=tmp_path/parent_id;child=tmp_path/child_id;parent.mkdir();child.mkdir()
    (parent/'job.json').write_text(json.dumps({'id':parent_id,'provider':'anthropic','status':'infrastructure_error','finished_at':'now','current_job':child_id}))
    stderr=('Traceback: count-only network failure\nRuntimeError: '+COUNT_TRANSPORT_ERROR+'\n').encode()
    (parent/'stderr.txt').write_bytes(stderr)
    (child/'job.json').write_text(json.dumps({'id':child_id,'campaign_id':parent_id,'provider':'anthropic','kind':'adaptive-direct-attempt','status':'infrastructure_error','finished_at':'now'}))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'kind':'anthropic_count_transport_recovery_v1','predecessor_job_id':parent_id,'child_job_id':child_id,'status':'infrastructure_error','stderr_sha256':hashlib.sha256(stderr).hexdigest(),'error':COUNT_TRANSPORT_ERROR,'action':'allow_fresh_toast_only','fix_verified':True}))
    return audit,parent,child


def test_recovery_binds_exact_audit_and_original_stderr(tmp_path):
    import hashlib
    from pairing.adapt_parallel_pair import validate_predecessor_recovery
    audit,parent,child=recovery_fixture(tmp_path)
    value=validate_predecessor_recovery(audit,parent)
    assert value['audit_sha256']==hashlib.sha256(audit.read_bytes()).hexdigest()
    assert value['child_job_id']==child.name
    assert json.loads((parent/'job.json').read_text())['status']=='infrastructure_error'


@pytest.mark.parametrize('damage',['hash','error','identity','fix_unverified','wrong_provider','completed_child','verification','artifact_error'])
def test_recovery_rejects_other_failures_or_completed_trials(tmp_path,damage):
    from pairing.adapt_parallel_pair import validate_predecessor_recovery
    audit,parent,child=recovery_fixture(tmp_path)
    if damage in {'hash','error','identity','fix_unverified'}:
        value=json.loads(audit.read_text())
        key,changed={'hash':('stderr_sha256','bad'),'error':('error','Anthropic HTTP 402'),'identity':('predecessor_job_id','other'),'fix_unverified':('fix_verified',False)}[damage]
        value[key]=changed;audit.write_text(json.dumps(value))
    elif damage=='verification':(child/'verification.json').write_text('{}')
    else:
        value=json.loads((child/'job.json').read_text())
        key,changed={'wrong_provider':('provider','openrouter'),'completed_child':('status','verified_fail'),'artifact_error':('artifact_errors',['readback failure'])}[damage]
        value[key]=changed;(child/'job.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):validate_predecessor_recovery(audit,parent)


def test_recovery_rejects_different_exception_even_with_matching_hash(tmp_path):
    import hashlib
    from pairing.adapt_parallel_pair import validate_predecessor_recovery
    audit,parent,_=recovery_fixture(tmp_path)
    raw=b'RuntimeError: ArtifactDurabilityError\n';(parent/'stderr.txt').write_bytes(raw)
    value=json.loads(audit.read_text());value['stderr_sha256']=hashlib.sha256(raw).hexdigest();audit.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='exact count-only'):validate_predecessor_recovery(audit,parent)


def test_recovery_requires_separate_state_directory(tmp_path):
    from pairing.adapt_parallel_pair import main
    with pytest.raises(ValueError,match='new separate'):
        main(['--root',str(tmp_path),'--predecessor-pid','42','--predecessor-job','missing','--predecessor-recovery','audit.json'])


def test_recovery_fresh_task_is_toast_only(tmp_path):
    from pairing.adapt_parallel_pair import validate_fresh_toast
    p=tmp_path/'toast-modal-dataset.json'
    p.write_text(json.dumps({'tasks':[{'task':'BlendIngredients','config':{'task':'robocasa/BlendIngredients'}}]}))
    with pytest.raises(ValueError):validate_fresh_toast(tmp_path)
    p.write_text(json.dumps({'tasks':[{'task':'ToastBagel','config':{'task':'robocasa/ToastBagel'}}]}))
    validate_fresh_toast(tmp_path)


def test_recovery_still_waits_and_preserves_old_state(tmp_path,monkeypatch):
    from pairing.adapt_parallel_pair import main
    configs(tmp_path);audit,parent,_=recovery_fixture(tmp_path)
    (tmp_path/'toast-modal-dataset.json').write_text(json.dumps({'tasks':[{'task':'ToastBagel','config':{'task':'robocasa/ToastBagel'}}]}))
    original=tmp_path/'modal-toast-parallel-pair';original.mkdir();(original/'active.json').write_text('{"status":"stopped","pid":66915}')
    new=tmp_path/'replacement';new.mkdir();(new/'STOP').touch()
    monkeypatch.setattr('pairing.adapt_parallel_pair.pid_live',lambda pid:True)
    monkeypatch.setattr('pairing.adapt_parallel_pair.subprocess.Popen',lambda *a,**kw:pytest.fail('Must wait and respect STOP'))
    main(['--root',str(tmp_path),'--state-dir',str(new),'--predecessor-pid','42','--predecessor-job',str(parent),'--predecessor-recovery',str(audit)])
    state=json.loads((new/'active.json').read_text())
    assert state['status']=='stopped' and state['children']==[] and state['predecessor_recovery']['audit_sha256']
    assert json.loads((original/'active.json').read_text())=={'status':'stopped','pid':66915}
