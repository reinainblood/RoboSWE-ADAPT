import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from pairing import adapt_provider_queue as q


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def transition(tmp_path):
    plan=put(tmp_path/'plan.json',{'pairs':[]})
    retired=put(tmp_path/'retired.json',{'pid':10,'status':'stopped','finished_epoch':1,'children':[]})
    bindings={}
    for n,p in enumerate(q.PROVIDERS,20):
        path=put(tmp_path/(p+'.json'),{'pid':n,'provider':p})
        bindings[p]={'pid':n,'state':str(path)}
    audit=put(tmp_path/'transition.json',{'kind':'provider_queue_transition_v1','lock_policy':'provider-locks-after-audited-legacy-transition','max_parallel_provider_series':2,'plan_sha256':q.digest(plan),'retired_waiter':{'pid':10,'state':str(retired)},'provider_predecessors':bindings})
    return audit,plan,bindings


def test_transition_allows_other_legacy_provider_live(tmp_path,monkeypatch):
    audit,plan,b=transition(tmp_path);monkeypatch.setattr(q,'pid_live',lambda pid:pid==21)
    assert q.validate_transition(audit,plan,'openrouter',20,b['openrouter']['state'])['sha256']==q.digest(audit)


@pytest.mark.parametrize('damage',['live_waiter','wrong_pid','plan_changed','old_children','wrong_concurrency','implicit_global_bypass'])
def test_transition_rejects_unsafe_handoff(tmp_path,monkeypatch,damage):
    audit,plan,b=transition(tmp_path);monkeypatch.setattr(q,'pid_live',lambda pid:damage=='live_waiter' and pid==10)
    d=q.read(audit)
    if damage=='wrong_pid':d['provider_predecessors']['openrouter']['pid']=99
    if damage=='plan_changed':put(plan,{'changed':True})
    if damage=='old_children':put(Path(d['retired_waiter']['state']),{'pid':10,'status':'stopped','finished_epoch':1,'children':[{'pid':2}]})
    if damage=='wrong_concurrency':d['max_parallel_provider_series']=3
    if damage=='implicit_global_bypass':d.pop('lock_policy')
    put(audit,d)
    with pytest.raises(ValueError):q.validate_transition(audit,plan,'openrouter',20,b['openrouter']['state'])


def test_terminal_lane_requires_dead_owned_pinned_provider(tmp_path,monkeypatch):
    p=put(tmp_path/'active.json',{'pid':20,'provider':'openrouter','model':q.PINS['openrouter']['model'],'finished_epoch':1})
    monkeypatch.setattr(q,'pid_live',lambda pid:True)
    with pytest.raises(ValueError):q.terminal_lane(p,20,'openrouter')
    monkeypatch.setattr(q,'pid_live',lambda pid:False)
    assert q.terminal_lane(p,20,'openrouter')['pid']==20
    with pytest.raises(ValueError):q.terminal_lane(p,21,'openrouter')


def test_existing_failed_task_cannot_be_retried(tmp_path):
    put(tmp_path/'prior/job.json',{'kind':'adaptive-direct-series','provider':'openrouter','task':'BeverageSorting','status':'model_refusal'})
    put(tmp_path/'prior/plan.json',{'participant':{'model':q.PINS['openrouter']['model']}})
    with pytest.raises(ValueError,match='duplicate'):q.ensure_fresh_task(tmp_path,'openrouter','BeverageSorting')
    q.ensure_fresh_task(tmp_path,'anthropic','BeverageSorting')
    q.ensure_fresh_task(tmp_path,'openrouter','DateNight')


def test_stop_controls_all_boundaries(tmp_path):
    folder=tmp_path/'queue';root=tmp_path/'campaign';lane=folder/'lane'
    for p in (folder,root,lane):p.mkdir(parents=True,exist_ok=True)
    assert not q.stop_requested(folder,root,lane)
    for p in (folder,root,lane):
        (p/'STOP').touch();assert q.stop_requested(folder,root,lane);(p/'STOP').unlink()


def test_cleanup_only_polls_recorded_owned_ids(tmp_path,monkeypatch):
    jobs=tmp_path/'jobs';series=jobs/'parent'
    put(series/'job.json',{'id':'parent'})
    child=jobs/'child';put(child/'job.json',{'campaign_id':'parent'})
    put(child/'modal-session.json',{'environment':'FDR','sandbox_id':'sb-owned'})
    put(child/'remote-artifacts.json',{'persisted_readback_verified':True})
    put(jobs/'unrelated/job.json',{'campaign_id':'other'})
    put(jobs/'unrelated/modal-session.json',{'environment':'FDR','sandbox_id':'sb-other'})
    def fake(cmd,**kwargs):
        assert 'sb-owned' in cmd[-1] and 'sb-other' not in cmd[-1]
        assert kwargs['env']['MODAL_ENVIRONMENT']=='FDR'
        return Mock(stdout='{"sb-owned":137}')
    monkeypatch.setattr(q.subprocess,'run',fake)
    assert q.series_cleanup(series,jobs,{})=={'sb-owned':137}
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**kw:Mock(stdout='{"sb-owned":null}'))
    with pytest.raises(ValueError,match='incomplete'):q.series_cleanup(series,jobs,{})


def test_missing_cleanup_receipts_blocks_progress(tmp_path):
    series=tmp_path/'p';put(series/'job.json',{'id':'p'})
    with pytest.raises(ValueError,match='No owned'):q.series_cleanup(series,tmp_path,{})


def refusal_fixture(tmp_path,monkeypatch):
    jobs=tmp_path/'jobs';parent=jobs/'series';child=jobs/'attempt'
    config={'participant':{'model':q.PINS['openrouter']['model']},'execution':{'backend':'modal'}}
    cfg=put(tmp_path/'config.json',config)
    put(parent/'job.json',{'id':'series','kind':'adaptive-direct-series','provider':'openrouter','current_job':'attempt','finished_at':'now'})
    put(parent/'plan.json',{**config,'cohort':{'id':'prior-task'}})
    put(child/'job.json',{'id':'attempt','kind':'adaptive-direct-attempt','campaign_id':'series','provider':'openrouter','status':'infrastructure_error','finished_at':'now'})
    response={'call':1,'response':{'model':q.PINS['openrouter']['model'],'choices':[{'finish_reason':'content_filter','native_finish_reason':'SAFETY','message':{'content':None}}],'usage':{'cost':.01}}}
    (child/'provider-responses.jsonl').write_text(json.dumps(response)+'\n')
    review=put(child/'failure-review.json',{'schema_version':1,'job_id':'attempt','job_sha256':q.digest(child/'job.json'),'response_file_sha256':q.digest(child/'provider-responses.jsonl'),'response_call':1,'choice_index':0,'finish_reason':'content_filter','native_finish_reason':'SAFETY','classification':'model_refusal'})
    state=put(tmp_path/'active.json',{'pid':20,'provider':'openrouter','model':q.PINS['openrouter']['model'],'status':'stopped_error','finished_epoch':1,'current_task':'prior-task','config':str(cfg),'dataset':'unused','error':'OpenRouter incomplete decision; no operation executed'})
    monkeypatch.setattr(q,'pid_live',lambda p:False)
    monkeypatch.setattr(q,'series_cleanup',lambda *a:{'sb-owned':137})
    import roboswe.adaptive
    monkeypatch.setattr(roboswe.adaptive,'build_plan',lambda *a,**k:{'tasks':[config]})
    return review,state,jobs,child,parent


def test_exact_reviewed_refusal_allows_only_fresh_progress(tmp_path,monkeypatch):
    review,state,jobs,child,parent=refusal_fixture(tmp_path,monkeypatch)
    result=q.reviewed_refusal(review,state,20,'openrouter',jobs)
    assert result['classification']=='reviewed_task_refusal'
    assert result['cleanup']=={'sb-owned':137}


@pytest.mark.parametrize('damage',['response_hash','wrong_child','funding','verification','missing_review'])
def test_review_does_not_bypass_other_failures(tmp_path,monkeypatch,damage):
    review,state,jobs,child,parent=refusal_fixture(tmp_path,monkeypatch)
    if damage=='response_hash':(child/'provider-responses.jsonl').write_text('{}\n')
    if damage=='wrong_child':d=q.read(parent/'job.json');d['current_job']='other';put(parent/'job.json',d)
    if damage=='funding':d=q.read(state);d['error']='OpenRouter HTTP 402';put(state,d)
    if damage=='verification':put(child/'verification.json',{})
    if damage=='missing_review':review=None
    with pytest.raises(ValueError):q.reviewed_refusal(review,state,20,'openrouter',jobs)


def test_future_review_only_typed_refusal_new_file(tmp_path,monkeypatch):
    review,statepath,jobs,child,parent=refusal_fixture(tmp_path,monkeypatch)
    review.unlink()
    state=q.read(statepath);state.update(status='model_refusal',started_epoch=0)
    dataset=put(tmp_path/'dataset.json',{'tasks':[{'task':'BeverageSorting'}]});state['dataset']=str(dataset)
    job=q.read(parent/'job.json');job.update(task='BeverageSorting',created_at='2026-01-01T00:00:00+00:00');put(parent/'job.json',job)
    job=q.read(child/'job.json');job['status']='model_refusal';put(child/'job.json',job)
    created=q.future_refusal_review(state,jobs,'openrouter');assert created==review
    original=review.read_bytes();q.future_refusal_review(state,jobs,'openrouter');assert review.read_bytes()==original
    state['status']='infrastructure_error'
    with pytest.raises(ValueError):q.future_refusal_review(state,jobs,'openrouter')


def test_refusal_unknown_cost_cannot_advance(tmp_path,monkeypatch):
    review,state,jobs,child,parent=refusal_fixture(tmp_path,monkeypatch)
    row=q.read_last_response(child);row['response']['usage']['cost']=None
    (child/'provider-responses.jsonl').write_text(json.dumps(row)+'\n')
    d=q.read(review);d['response_file_sha256']=q.digest(child/'provider-responses.jsonl');put(review,d)
    with pytest.raises(ValueError,match='cost'):q.reviewed_refusal(review,state,20,'openrouter',jobs)


def test_stop_before_paid_launch(tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir();folder=tmp_path/'queue';jobs=tmp_path/'jobs';jobs.mkdir()
    plan=put(tmp_path/'plan.json',{});audit=put(tmp_path/'transition.json',{})
    dataset=put(tmp_path/'dataset.json',{'tasks':[{'task':'BeverageSorting'}]})
    previous_dataset=put(tmp_path/'prior.json',{'tasks':[{'task':'ToastBagel'}]})
    config=put(tmp_path/'config.json',{'execution':{}})
    pair={'id':'beverage-sorting','dataset':str(dataset),'configs':{'openrouter':str(config)}}
    monkeypatch.setattr(q,'validate_plan',lambda *a:([pair],{str(plan):q.digest(plan)}))
    monkeypatch.setattr(q,'validate_transition',lambda *a:{'path':str(audit),'sha256':q.digest(audit)})
    monkeypatch.setattr(q,'pid_live',lambda p:False)
    monkeypatch.setattr(q,'terminal_lane',lambda *a:{'status':'completed','dataset':str(previous_dataset)})
    monkeypatch.setattr(q,'completed_lane',lambda *a:{'classification':'completed'})
    monkeypatch.setattr(q.subprocess,'Popen',lambda *a,**kw:pytest.fail('Paid child launched after STOP'))
    (root/'STOP').touch()
    result=q.main(['--plan',str(plan),'--state-dir',str(folder),'--campaign-root',str(root),'--predecessor-state','unused','--predecessor-pid','20','--transition-review',str(audit),'--provider','openrouter','--jobs-dir',str(jobs)])
    assert result['status']=='stopped' and result['children']==[]


def recovery_fixture(tmp_path,monkeypatch):
    from zipfile import ZipFile
    jobs=tmp_path/'jobs';parent=jobs/q.RECOVERY_PARENT;child=jobs/q.RECOVERY_CHILD
    config={'participant':{'model':q.PINS['openrouter']['model']},'execution':{'backend':'modal'},'cohort':{'id':'beverage'}}
    cfg=put(tmp_path/'config.json',config)
    put(parent/'job.json',{'id':q.RECOVERY_PARENT,'kind':'adaptive-direct-series','provider':'openrouter','task':'BeverageSorting','status':'infrastructure_error','current_job':q.RECOVERY_CHILD,'current_attempt':1,'finished_at':'now'})
    put(child/'job.json',{'id':q.RECOVERY_CHILD,'kind':'adaptive-direct-attempt','provider':'openrouter','task':'BeverageSorting','status':'infrastructure_error','campaign_id':q.RECOVERY_PARENT,'attempt_number':1,'finished_at':'now'})
    put(parent/'plan.json',config);put(parent/'report.json',{'trials':[]})
    for d in (parent,child):(d/'stderr.txt').write_text('ValueError: '+q.RECOVERY_ERROR+'\n')
    (child/'actions.jsonl').write_text('');(child/'submissions.jsonl').write_text('{}\n')
    row={'call':1,'response':{'model':q.PINS['openrouter']['model'],'usage':{'cost':.01},'choices':[{'finish_reason':'stop','native_finish_reason':'STOP','message':{'refusal':None,'content':'{}'}}]}}
    (child/'provider-responses.jsonl').write_text(json.dumps(row)+'\n')
    for name in ('initial-snapshot.zip','final-snapshot.zip'):
        with ZipFile(child/name,'w') as z:z.writestr('manifest.json',json.dumps({'steps':0,'current_fingerprint':'same'}))
    test=put(tmp_path/'tests.json',{'status':'passed','returncode':0,'tests_passed':1})
    review=put(tmp_path/'recovery.json',{'kind':'openrouter_null_refusal_counter_recovery_v1','action':'replace_zero_action_BeverageSorting_once','fix_verified':True,'predecessor_pid':28686,'parent_job_id':q.RECOVERY_PARENT,'child_job_id':q.RECOVERY_CHILD,'evidence_sha256':{str(Path(j)/f):q.digest(jobs/j/f) for j,names in q.RECOVERY_FILES.items() for f in names},'fix_files_sha256':{'pairing/openrouter_google_count.py':q.digest(Path(q.__file__).resolve().parents[1]/'pairing/openrouter_google_count.py')},'test_receipt':str(test),'test_receipt_sha256':q.digest(test)})
    state=put(tmp_path/'active.json',{'pid':28686,'provider':'openrouter','model':q.PINS['openrouter']['model'],'status':'stopped_error','finished_epoch':1,'error':q.RECOVERY_ERROR,'error_type':'ValueError','config':str(cfg),'dataset':'unused'})
    monkeypatch.setattr(q,'pid_live',lambda p:False);monkeypatch.setattr(q,'series_cleanup',lambda *a:{'sb-owned':137})
    import roboswe.adaptive
    monkeypatch.setattr(roboswe.adaptive,'build_plan',lambda *a,**kw:{'tasks':[config]})
    return review,state,jobs,parent,child


def test_exact_zero_action_recovery(tmp_path,monkeypatch):
    review,state,jobs,parent,child=recovery_fixture(tmp_path,monkeypatch)
    result=q.validate_infrastructure_recovery(review,state,28686,'openrouter',jobs)
    assert result['ignored_parent_id']==q.RECOVERY_PARENT
    q.ensure_fresh_task(jobs,'openrouter','BeverageSorting',ignored_parent_id=result['ignored_parent_id'])
    put(jobs/'another/job.json',{'id':'another','kind':'adaptive-direct-series','provider':'openrouter','task':'BeverageSorting'})
    put(jobs/'another/plan.json',{'participant':{'model':q.PINS['openrouter']['model']}})
    with pytest.raises(ValueError,match='duplicate'):q.ensure_fresh_task(jobs,'openrouter','BeverageSorting',ignored_parent_id=result['ignored_parent_id'])


@pytest.mark.parametrize('damage',['actions','second_response','wrong_error','completed_trial','verification','changed_fix','failed_test','physical_steps','non_null_refusal','wrong_parent'])
def test_recovery_never_generalizes(tmp_path,monkeypatch,damage):
    from zipfile import ZipFile
    review,state,jobs,parent,child=recovery_fixture(tmp_path,monkeypatch)
    d=q.read(review)
    if damage=='actions':(child/'actions.jsonl').write_text('{}\n')
    if damage=='second_response':p=child/'provider-responses.jsonl';p.write_text(p.read_text()*2)
    if damage=='wrong_error':(child/'stderr.txt').write_text('ValueError: other\n')
    if damage=='completed_trial':put(parent/'report.json',{'trials':[{}]})
    if damage=='verification':put(child/'verification.json',{})
    if damage=='changed_fix':d['fix_files_sha256']['pairing/openrouter_google_count.py']='wrong'
    if damage=='failed_test':p=Path(d['test_receipt']);put(p,{'status':'failed','returncode':1,'tests_passed':0});d['test_receipt_sha256']=q.digest(p)
    if damage=='physical_steps':
        with ZipFile(child/'final-snapshot.zip','w') as z:z.writestr('manifest.json',json.dumps({'steps':1,'current_fingerprint':'same'}))
    if damage=='non_null_refusal':
        p=child/'provider-responses.jsonl';row=json.loads(p.read_text());row['response']['choices'][0]['message']['refusal']='No';p.write_text(json.dumps(row)+'\n')
    if damage=='wrong_parent':d['parent_job_id']='another'
    # Rebind bytes to prove semantic checks, not merely stale-hash rejection.
    d['evidence_sha256']={str(Path(j)/f):q.digest(jobs/j/f) for j,names in q.RECOVERY_FILES.items() for f in names};put(review,d)
    with pytest.raises(ValueError):q.validate_infrastructure_recovery(review,state,28686,'openrouter',jobs)
