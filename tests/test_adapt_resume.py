import hashlib,json
from pathlib import Path
import pytest
from pairing import adapt_resume as r


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path

def fixture(tmp_path,monkeypatch):
    parent=tmp_path/'jobs/parent';child=tmp_path/'jobs/first';prefix=tmp_path/'jobs/second';watch=tmp_path/'watch'
    boundary={'parent_job_id':'parent','completed_trials':1,'next_trial_number':2,'completed_attempt_ids':['first'],'zero_input_prefix_job_id':'second'}
    put(tmp_path/'ledger.json',{'reserved':{}})
    required=[]
    required.append(put(parent/'job.json',{'id':'parent','status':'infrastructure_error','current_job':'second'}))
    required.append(put(parent/'plan.json',{'protocol':'adaptive-public-sqlite-v1','trials':5,'participant':{'provider':'openrouter','context_policy':'sqlite-sliding-or-google-count-v2','ledger_path':str(tmp_path/'ledger.json')},'code_image':'image','initial_fingerprint':'fixed'}))
    required.append(put(child/'job.json',{'id':'first','campaign_id':'parent','attempt_number':1,'status':'verified_fail'}))
    required.append(put(child/'report.json',{'observed_success':False}))
    required.append(put(child/'verification.json',{'verified_success':False}))
    required.append(put(child/'continuity.json',{'initial_fingerprint':'fixed'}))
    for name in ('context-end.json.gz','history.sqlite','history-manifest.json','provider-responses.jsonl','model-usage.jsonl'):
        required.append(put(child/name,{}))
    for name in ('model.json','source.zip'):
        required.append(put(parent/name,{}))
    required.append(put(prefix/'job.json',{'id':'second','campaign_id':'parent','attempt_number':2,'status':'infrastructure_error'}))
    required.append(put(watch/'active.json',{'status':'preserved_and_stopped_at_boundary','signal_sent':True,'queue_terminal':True,'worker_terminal':True,'pending_no_model_input_after_stop':True,'parent':'parent','pending_trial2':'second','queue_pid':123,'worker_pid':124}))
    required.append(put(watch/'pre-signal-boundary.json',{'parent':{'id':'parent'},'pending_child':{'id':'second'},'no_new_model_input':True,'trial1_job_sha256':r.sha(child/'job.json'),'trial1_verification_sha256':r.sha(child/'verification.json')}))
    from roboswe.execution.workspace_checkpoint import checkpoint
    source=tmp_path/'source';source.mkdir();(source/'binary.npy').write_bytes(b'\0binary')
    workspace=tmp_path/'workspace.zip';checkpoint(source,workspace)
    required.append(put(watch/'checkpoint-provenance.json',{'checkpoint_sha256':r.sha(workspace),'code_image':'image','evidence_sha256':{'report.json':r.sha(child/'report.json')}}))
    archive=put(tmp_path/'archive.json',{'scope':'finalized-job','job_id':'first','remote_readback_verified':True,'restore_test_passed':True,'files':[{'path':n,'sha256':r.sha(child/n)} for n in ('job.json','report.json','verification.json')]});required.append(archive)
    model=put(tmp_path/'checkpoint.json',{'test':'model'} )
    put(tmp_path/'history.sqlite',{});put(tmp_path/'history-manifest.json',{})
    cert={'schema_version':'certified-adaptive-boundary-v1','boundary':boundary,'plan_sha256':r.sha(parent/'plan.json'),
          'evidence_sha256':{str(p.resolve()):r.sha(p) for p in required},'watcher_directory':str(watch),
          'workspace':str(workspace),'workspace_sha256':r.sha(workspace),'completed_archive_receipt':str(archive),
          'model_checkpoint':str(model),'model_checkpoint_sha256':r.sha(model),'campaign_root':str(tmp_path),'continuation_archive_receipt':str(tmp_path/'durability.json')}
    certpath=put(tmp_path/'cert.json',cert)
    put(tmp_path/'durability.json',{'schema_version':'continuation-durability-v1','environment':'FDR','modal_profile':'snorkelai','files':[{**item,'remote_readback_verified':True,'remote_path':'/immutable/'+item['sha256']} for item in r.continuation_artifacts(certpath)]})
    monkeypatch.setattr('pairing.openrouter_checkpoint.validate_against_receipts',lambda *a,**k:None,raising=False)
    monkeypatch.setattr('pairing.adapt_modal_pair.pid_live',lambda _:False)
    monkeypatch.setattr('roboswe.adaptive_results._attempt',lambda *a:{'official_success':False,'category':'verified_fail'})
    return parent,certpath,cert,prefix


def test_validation_preserves_identity_and_allocates_nothing(tmp_path,monkeypatch):
    parent,path,cert,prefix=fixture(tmp_path,monkeypatch)
    info=r.validate(parent,path)
    assert info['parent']==str(parent) and info['prefix']==str(prefix)
    assert info['boundary']['next_trial_number']==2
    assert not (parent/'resume-history').exists()


@pytest.mark.parametrize('filename',['provider-responses.jsonl','actions.jsonl','timing.jsonl','context-estimates.jsonl','submissions.jsonl','report.json'])
def test_any_prefix_activity_blocks_resume(tmp_path,monkeypatch,filename):
    parent,path,cert,prefix=fixture(tmp_path,monkeypatch)
    (prefix/filename).write_text('{}\n')
    with pytest.raises(ValueError):r.validate(parent,path)


def test_evidence_tamper_and_live_original_process_fail_closed(tmp_path,monkeypatch):
    parent,path,cert,prefix=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr('pairing.adapt_modal_pair.pid_live',lambda _:True)
    with pytest.raises(ValueError,match='terminal'):r.validate(parent,path)
    monkeypatch.setattr('pairing.adapt_modal_pair.pid_live',lambda _:False)
    (parent/'job.json').write_text('{}')
    with pytest.raises(ValueError):r.validate(parent,path)


def test_unverified_completed_trial_never_resumes(tmp_path,monkeypatch):
    parent,path,cert,prefix=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr('roboswe.adaptive_results._attempt',lambda *a:{'official_success':None,'category':'awaiting_verification'})
    with pytest.raises(ValueError,match='independent replay'):r.validate(parent,path)


def test_stop_file_blocks_certified_boundary(tmp_path,monkeypatch):
    parent,path,cert,prefix=fixture(tmp_path,monkeypatch)
    (tmp_path/'STOP').write_text('stop')
    with pytest.raises(ValueError,match='STOP'):r.validate(parent,path)


def test_missing_new_bundle_readback_blocks_boundary(tmp_path,monkeypatch):
    parent,path,cert,prefix=fixture(tmp_path,monkeypatch)
    durability=Path(cert['continuation_archive_receipt'])
    data=json.loads(durability.read_text());data['files'][0]['remote_readback_verified']=False
    durability.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='readback'):r.validate(parent,path)


def test_exact_model_reconstruction_is_required(tmp_path,monkeypatch):
    parent,path,cert,prefix=fixture(tmp_path,monkeypatch)
    def reject(*args,**kwargs):raise ValueError('Original context mismatch')
    monkeypatch.setattr('pairing.openrouter_checkpoint.validate_against_receipts',reject)
    with pytest.raises(ValueError,match='context mismatch'):r.validate(parent,path)


@pytest.mark.parametrize('count',[2,4])
def test_multiple_completed_trials_require_ordered_verification_and_archives(tmp_path,monkeypatch,count):
    parent,path,cert,prefix=fixture(tmp_path,monkeypatch)
    first=parent.parent/'first';ids=['first'];extra=[]
    archives={'first':cert.pop('completed_archive_receipt')}
    for number in range(2,count+1):
        jobid='completed-'+str(number);ids.append(jobid);directory=parent.parent/jobid
        directory.mkdir()
        for source in first.iterdir():
            if source.is_file():(directory/source.name).write_bytes(source.read_bytes())
        put(directory/'job.json',{'id':jobid,'campaign_id':'parent','attempt_number':number,'status':'verified_fail'})
        archive=put(tmp_path/(jobid+'-archive.json'),{'scope':'finalized-job','job_id':jobid,'remote_readback_verified':True,'restore_test_passed':True,
            'files':[{'path':n,'sha256':r.sha(directory/n)} for n in ('job.json','report.json','verification.json')]})
        archives[jobid]=str(archive);extra.extend([*directory.iterdir(),archive])
    put(prefix/'job.json',{'id':'second','campaign_id':'parent','attempt_number':count+1,'status':'infrastructure_error'})
    cert['boundary'].update(completed_trials=count,next_trial_number=count+1,completed_attempt_ids=ids)
    cert['completed_archive_receipts']=archives
    watch=Path(cert['watcher_directory']);state=json.loads((watch/'active.json').read_text());state['pending_trial_job']='second';put(watch/'active.json',state)
    proof=json.loads((watch/'pre-signal-boundary.json').read_text())
    proof['completed_trials']=[{'job_id':i,'job_sha256':r.sha(parent.parent/i/'job.json'),'verification_sha256':r.sha(parent.parent/i/'verification.json')} for i in ids]
    put(watch/'pre-signal-boundary.json',proof)
    provenance=json.loads((watch/'checkpoint-provenance.json').read_text());provenance['completed_attempt_id']=ids[-1];put(watch/'checkpoint-provenance.json',provenance)
    cert['evidence_sha256']={str(p):r.sha(p) for p in [*(Path(p) for p in cert['evidence_sha256']),*extra]}
    put(path,cert)
    put(Path(cert['continuation_archive_receipt']),{'schema_version':'continuation-durability-v1','environment':'FDR','modal_profile':'snorkelai',
        'files':[{**row,'remote_readback_verified':True,'remote_path':'/immutable/'+row['sha256']} for row in r.continuation_artifacts(path)]})
    info=r.validate(parent,path)
    assert len(info['completed_directories'])==count
    assert info['boundary']['next_trial_number']==count+1
    assert [Path(p).name for p in info['completed_directories']]==ids
    # Missing any older archive cannot be waved through by a valid latest trial.
    del cert['completed_archive_receipts']['first'];put(path,cert)
    with pytest.raises(ValueError,match='Every completed trial'):r.validate(parent,path)


def test_only_exact_reviewed_external_holds_are_preserved_without_ledger_mutation():
    import copy
    hold={'amount_usd':2.179072,'label':'historical visual audit','created_epoch':123}
    ledger={'reserved':{'external':hold},'spent_usd':20,'limit_usd':150}
    original=copy.deepcopy(ledger)
    r._check_reserved_spend(ledger,{'external':copy.deepcopy(hold)})
    assert ledger==original
    with pytest.raises(ValueError):r._check_reserved_spend(ledger,{})
    with pytest.raises(ValueError):r._check_reserved_spend(ledger,{'external':{**hold,'amount_usd':0}})
    with pytest.raises(ValueError):r._check_reserved_spend({'reserved':{}},{'external':hold})
    with pytest.raises(ValueError):r._check_reserved_spend({'reserved':{**ledger['reserved'],'new-model-request':hold}},{'external':hold})


def test_actor_request_hold_cannot_be_approved_as_external():
    actor={'amount_usd':2.424832,'label':'google/gemini-4-argon:original-trial:call29','created_epoch':123}
    with pytest.raises(ValueError,match='Actor request'):
        r._check_reserved_spend({'reserved':{'actor':actor}},{'actor':actor})


def test_completed_timeout_hold_is_separate_exact_and_never_settled():
    import copy
    external={'amount_usd':2.179072,'label':'Gemini visual calibration historical-positive','created_epoch':100}
    timeout={'reservation_id':'timed-out','amount_usd':2.424832,'label':'google/gemini-4-argon:trial2:call43','created_epoch':200,'attempt_id':'trial2','pending_call':43}
    ledger={'reserved':{'external':external,'timed-out':{k:timeout[k] for k in ('amount_usd','label','created_epoch')}},'limit_usd':150}
    before=copy.deepcopy(ledger)
    r._check_reserved_spend(ledger,{'external':external},completed_timeouts=[timeout],completed_attempt_ids=['trial1','trial2'])
    assert ledger==before
    with pytest.raises(ValueError):r._check_reserved_spend(ledger,{'external':external},completed_timeouts=[timeout],completed_attempt_ids=['trial1'])
    for field,value in [('amount_usd',0),('label','other'),('created_epoch',201),('pending_call',44),('attempt_id','trial3')]:
        with pytest.raises(ValueError):r._check_reserved_spend(ledger,{'external':external},completed_timeouts=[{**timeout,field:value}],completed_attempt_ids=['trial1','trial2'])
    with pytest.raises(ValueError):r._check_reserved_spend(ledger,{'external':external,'timed-out':ledger['reserved']['timed-out']},completed_timeouts=[timeout],completed_attempt_ids=['trial2'])
