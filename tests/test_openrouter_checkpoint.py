import json
import sqlite3
from pathlib import Path

import pytest

from pairing.openrouter_checkpoint import capture, restore
from pairing.openrouter_model import OpenRouterLearningModel
from pairing.context_history import INSTRUCTION

SCHEMA={'type':'object','properties':{k:({'type':'string','enum':['finish']} if k=='operation' else {'type':'string'}) for k in ('operation','path','content','message')},'required':['operation','path','content','message'],'additionalProperties':False}
BOUNDARY={'parent_job_id':'original-series','completed_trials':1,'next_trial_number':2,'completed_attempt_ids':['trial-1']}


def client(tmp_path,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','offline-count-key')
    monkeypatch.setattr('pairing.openrouter_model.credential',lambda:'private-key-never-serialize')
    config={'provider':'openrouter','model':'google/gemini-4-argon','context_window':1048576,'max_output_tokens':32768,'effort':'high','upstream_provider':'google-vertex/global','context_policy':'sqlite-sliding-or-google-count-v2','ledger_path':str(tmp_path/'ledger.json'),'spend_limit_usd':150.0}
    model=OpenRouterLearningModel('Original instructions',tmp_path/'trial-1',model=config['model'],context_window=config['context_window'],max_output_tokens=config['max_output_tokens'],provider=config['upstream_provider'],context_policy=config['context_policy'],ledger_path=config['ledger_path'])
    requests=[]
    def post(url,body,timeout):
        requests.append(body)
        return {'model':'google/gemini-4-argon','choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'operation':'finish','path':'','content':'','message':'Recorded rationale'}),'reasoning_details':[{'type':'reasoning.encrypted','data':'b3BhcXVl','format':'google-gemini-v1'}]}}],'usage':{'prompt_tokens':20,'completion_tokens':4,'cost':0.01}}
    monkeypatch.setattr(OpenRouterLearningModel,'_post',staticmethod(post))
    model.approximate_counter.native.count_native_request=lambda request,timeout:10
    model.ask('Trial one observation',schema=SCHEMA)
    return model,config,requests


def resumed(path,receipt,config,tmp_path):
    return restore(path,checkpoint_sha256=receipt['sha256'],destination=tmp_path/'restored',key='new-key',expected_participant=config,expected_instructions='Original instructions'+INSTRUCTION,expected_boundary=BOUNDARY)


def test_next_generation_and_count_requests_exactly_match_uninterrupted(tmp_path,monkeypatch):
    model,config,requests=client(tmp_path,monkeypatch)
    # Calibration grew from 1.30 to 2.20 based on actual usage; resetting it
    # would alter sliding behavior and fail the continuation contract.
    assert model.approximate_counter.multiplier==2.2
    receipt=capture(model,tmp_path/'checkpoint',boundary=BOUNDARY)
    snapshot=Path(receipt['checkpoint']).read_text();assert 'private-key-never-serialize' not in snapshot
    count_requests=[]
    model.approximate_counter.native.count_native_request=lambda req,timeout:(count_requests.append(req) or 10)
    model.ask('Identical trial two reset observation',schema=SCHEMA)
    expected_generation=requests[-1];expected_counts=list(count_requests)
    clone=resumed(receipt['checkpoint'],receipt,config,tmp_path)
    assert clone.thread_id==model.thread_id and clone.calls==1
    assert clone.instructions.count(INSTRUCTION)==1
    assert clone.approximate_counter.multiplier==2.2
    got_counts=[];clone.approximate_counter.native.count_native_request=lambda req,timeout:(got_counts.append(req) or 10)
    clone.ask('Identical trial two reset observation',schema=SCHEMA)
    assert requests[-1]==expected_generation
    assert got_counts==expected_counts
    assert clone.messages==model.messages and clone.usage==model.usage and clone.calls==model.calls
    assert clone.messages[1]['reasoning_details'][0]['data']=='b3BhcXVl'


def test_archive_search_ids_and_active_representation_preserved(tmp_path,monkeypatch):
    model,config,_=client(tmp_path,monkeypatch)
    with model.history_store.connect() as db:
        db.execute('INSERT INTO slides(trial,epoch,removed_messages,estimated_removed_tokens,estimated_before_tokens,estimated_after_tokens) VALUES(?,?,?,?,?,?)',('trial-1',1,0,0,20,20))
    expected=model.history_store.search('Recorded')
    receipt=capture(model,tmp_path/'checkpoint',boundary=BOUNDARY)
    clone=resumed(receipt['checkpoint'],receipt,config,tmp_path)
    assert clone.history_store.active==model.history_store.active
    assert clone.history_store.search('Recorded')==expected
    with clone.history_store.connect() as db:assert db.execute('SELECT COUNT(*) FROM messages').fetchone()[0]==2


@pytest.mark.parametrize('fault',['state_hash','sqlite_hash','policy','instructions','thread','accounting','unsettled'])
def test_missing_or_changed_evidence_fails_closed(tmp_path,monkeypatch,fault):
    model,config,_=client(tmp_path,monkeypatch)
    if fault=='thread':
        model.metadata['thread_id']='other'
        with pytest.raises(ValueError):capture(model,tmp_path/'checkpoint',boundary=BOUNDARY)
        return
    if fault=='accounting':
        model.calls=2
        with pytest.raises(ValueError):capture(model,tmp_path/'checkpoint',boundary=BOUNDARY)
        return
    if fault=='unsettled':
        model.ledger.reserve(1,label='google/gemini-4-argon:trial-1:call2')
        with pytest.raises(ValueError):capture(model,tmp_path/'checkpoint',boundary=BOUNDARY)
        return
    receipt=capture(model,tmp_path/'checkpoint',boundary=BOUNDARY)
    if fault=='state_hash':Path(receipt['checkpoint']).write_text('{}')
    if fault=='sqlite_hash':(tmp_path/'checkpoint/history.sqlite').write_bytes(b'bad')
    if fault=='policy':config['context_policy']='different'
    if fault=='instructions':config['model']='different'
    with pytest.raises(ValueError):resumed(receipt['checkpoint'],receipt,config,tmp_path)


def reconstruction_fixture(tmp_path,monkeypatch):
    from zipfile import ZipFile
    model,config,requests=client(tmp_path,monkeypatch)
    attempt=tmp_path/'trial-1';parent=tmp_path/'original-series';parent.mkdir()
    (parent/'job.json').write_text(json.dumps({'id':parent.name,'current_job':attempt.name}))
    (parent/'plan.json').write_text(json.dumps({'participant':config,'prompt':'Original instructions'}))
    (parent/'model.json').write_text(json.dumps(model.metadata))
    (attempt/'job.json').write_text(json.dumps({'id':attempt.name,'campaign_id':parent.name,'attempt_number':1,'status':'verified_fail','finished_at':'now'}))
    (attempt/'verification.json').write_text(json.dumps({'kind':'independent_code_replay','reproduced':True,'verified_success':False,'mismatches':[],'trials':[{'reproduced':True,'infrastructure_valid':True,'precondition_valid':True,'verified_success':False,'observed_success':False,'replay':{'precondition_valid':True,'observed_success':False}}]}))
    (attempt/'continuity.json').write_text(json.dumps({'thread_id':model.thread_id}))
    import gzip
    (attempt/'context-start.json.gz').write_bytes(gzip.compress(b'[]'))
    model.save_context(attempt/'context-end.json.gz')
    project=Path(__file__).parents[1]
    with ZipFile(parent/'source.zip','w') as z:
        for name in ('pairing/openrouter_model.py','pairing/openai_model.py','pairing/openrouter_google_count.py','pairing/gemini_token_count.py','pairing/context_history.py'):z.write(project/name,name)
    return model,config,parent,attempt


def test_reconstruction_from_complete_original_receipts(tmp_path,monkeypatch):
    from pairing.openrouter_checkpoint import reconstruct_from_receipts
    model,config,parent,attempt=reconstruction_fixture(tmp_path,monkeypatch)
    receipt=reconstruct_from_receipts(parent,attempt,tmp_path/'imported',boundary=BOUNDARY)
    clone=resumed(receipt['checkpoint'],receipt,config,tmp_path)
    for field in ('messages','calls','usage','instructions','thread_id','metadata'):assert getattr(clone,field)==getattr(model,field)
    assert clone.approximate_counter.multiplier==model.approximate_counter.multiplier
    assert clone.history_store.active==model.history_store.active


@pytest.mark.parametrize('missing',['context-end.json.gz','model-usage.jsonl','context-estimates.jsonl','history-manifest.json'])
def test_reconstruction_missing_evidence_never_approximates(tmp_path,monkeypatch,missing):
    from pairing.openrouter_checkpoint import reconstruct_from_receipts
    _,_,parent,attempt=reconstruction_fixture(tmp_path,monkeypatch);(attempt/missing).unlink()
    with pytest.raises((ValueError,FileNotFoundError)):reconstruct_from_receipts(parent,attempt,tmp_path/'imported',boundary=BOUNDARY)
    assert not (tmp_path/'imported').exists()


def test_prefix_with_any_model_input_is_not_between_trial_resume(tmp_path,monkeypatch):
    from pairing.openrouter_checkpoint import reconstruct_from_receipts
    _,_,parent,attempt=reconstruction_fixture(tmp_path,monkeypatch)
    prefix=tmp_path/'trial-2';prefix.mkdir();(prefix/'job.json').write_text(json.dumps({'campaign_id':parent.name,'attempt_number':2}))
    (parent/'job.json').write_text(json.dumps({'current_job':prefix.name}))
    (prefix/'model-transcript.jsonl').write_text('{"event":"request"}\n')
    with pytest.raises(ValueError,match='input/activity'):
        reconstruct_from_receipts(parent,attempt,tmp_path/'imported',boundary={**BOUNDARY,'zero_input_prefix_job_id':prefix.name})


def test_next_request_equivalence_across_a_real_sliding_event(tmp_path,monkeypatch):
    model,config,requests=client(tmp_path,monkeypatch)
    model.ask('Second exchange',schema=SCHEMA)
    receipt=capture(model,tmp_path/'checkpoint',boundary=BOUNDARY)
    model.directory=tmp_path/'live/trial-2';model.directory.mkdir(parents=True)
    original_counts=[]
    def count_original(req,timeout):original_counts.append(req);return len(req['contents'])*100000
    model.approximate_counter.native.count_native_request=count_original
    model.ask('Fresh trial two observation',schema=SCHEMA)
    expected=json.dumps(requests[-1],ensure_ascii=False,separators=(',',':')).encode()
    clone=restore(receipt['checkpoint'],checkpoint_sha256=receipt['sha256'],destination=tmp_path/'restored/trial-2',key='offline',expected_participant=config,expected_instructions='Original instructions'+INSTRUCTION,expected_boundary=BOUNDARY)
    clone_counts=[]
    def count_clone(req,timeout):clone_counts.append(req);return len(req['contents'])*100000
    clone.approximate_counter.native.count_native_request=count_clone
    clone.ask('Fresh trial two observation',schema=SCHEMA)
    assert json.dumps(requests[-1],ensure_ascii=False,separators=(',',':')).encode()==expected
    assert original_counts==clone_counts
    assert model.history_store.slid and clone.history_store.slid
    assert model.history_store.search('Recorded')==clone.history_store.search('Recorded')


def test_checkpoint_cross_binding_uses_exact_original_receipts(tmp_path,monkeypatch):
    from pairing.openrouter_checkpoint import reconstruct_from_receipts, validate_against_receipts
    _,_,parent,attempt=reconstruction_fixture(tmp_path,monkeypatch)
    receipt=reconstruct_from_receipts(parent,attempt,tmp_path/'imported',boundary=BOUNDARY)
    before={p:p.read_bytes() for p in [parent/'job.json',attempt/'history.sqlite',tmp_path/'ledger.json']}
    result=validate_against_receipts(receipt['checkpoint'],parent,attempt,boundary=BOUNDARY)
    assert result['certified_against_original_receipts'] is True
    assert all(p.read_bytes()==b for p,b in before.items())


@pytest.mark.parametrize('field',['state','active','counter','ledger','history_rows'])
def test_wrong_checkpoint_cannot_bind_only_by_self_consistent_digest(tmp_path,monkeypatch,field):
    from pairing.openrouter_checkpoint import reconstruct_from_receipts, validate_against_receipts, sha
    _,_,parent,attempt=reconstruction_fixture(tmp_path,monkeypatch)
    receipt=reconstruct_from_receipts(parent,attempt,tmp_path/'imported',boundary=BOUNDARY)
    path=Path(receipt['checkpoint']);payload=json.loads(path.read_text())
    if field=='state':payload['state']['thread_id']='substituted-thread'
    elif field=='active':payload['active']=payload['active'][:-1]
    elif field=='counter':payload['counter']['multiplier']=1.3
    elif field=='ledger':payload['ledger']['limit_usd']=999
    else:
        db=path.parent/'history.sqlite'
        with sqlite3.connect(db) as connection:connection.execute("UPDATE messages SET trial='different-trial'")
        payload['history']['sha256']=sha(db)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):validate_against_receipts(path,parent,attempt,boundary=BOUNDARY)


def multiple_completed_fixture(tmp_path,monkeypatch,count):
    model,config,parent,first=reconstruction_fixture(tmp_path,monkeypatch)
    # Match the real series-scoped live archive; finalized trial backups remain immutable.
    live=parent/'history.sqlite'
    with sqlite3.connect(model.history_store.path) as source,sqlite3.connect(live) as target:source.backup(target)
    model.history_store.path=live
    directories=[first]
    verification=json.loads((first/'verification.json').read_text())
    for number in range(2,count+1):
        directory=tmp_path/f'trial-{number}';directory.mkdir();model.directory=directory
        model.save_context(directory/'context-start.json.gz')
        model.ask(f'Trial {number} input',schema=SCHEMA)
        model.save_context(directory/'context-end.json.gz')
        (directory/'job.json').write_text(json.dumps({'id':directory.name,'campaign_id':parent.name,'attempt_number':number,'status':'verified_fail','finished_at':'now'}))
        (directory/'verification.json').write_text(json.dumps(verification))
        (directory/'continuity.json').write_text(json.dumps({'thread_id':model.thread_id}))
        directories.append(directory)
    (parent/'job.json').write_text(json.dumps({'current_job':directories[-1].name}))
    boundary={'parent_job_id':parent.name,'completed_trials':count,'next_trial_number':count+1,'completed_attempt_ids':[d.name for d in directories]}
    return model,config,parent,directories,boundary


@pytest.mark.parametrize('count',[2,3,4])
def test_multiple_completed_trials_preserve_cumulative_client_state(tmp_path,monkeypatch,count):
    from pairing.openrouter_checkpoint import reconstruct_from_receipts,validate_against_receipts
    model,config,parent,directories,boundary=multiple_completed_fixture(tmp_path,monkeypatch,count)
    receipt=reconstruct_from_receipts(parent,directories[-1],tmp_path/'imported',boundary=boundary)
    clone=restore(receipt['checkpoint'],checkpoint_sha256=receipt['sha256'],destination=tmp_path/'restored',key='offline',expected_participant=config,expected_instructions='Original instructions'+INSTRUCTION,expected_boundary=boundary)
    assert clone.calls==count and clone.usage==model.usage and clone.messages==model.messages
    assert clone.thread_id==model.thread_id
    assert validate_against_receipts(receipt['checkpoint'],parent,directories[-1],boundary=boundary)['certified_against_original_receipts']


@pytest.mark.parametrize('damage',['history_start','calls','calibration','thread','replay','missing_trial'])
def test_completed_prefix_reconstruction_rejects_any_gap(tmp_path,monkeypatch,damage):
    import gzip
    from pairing.openrouter_checkpoint import reconstruct_from_receipts
    _,_,parent,dirs,boundary=multiple_completed_fixture(tmp_path,monkeypatch,2)
    if damage=='history_start':(dirs[1]/'context-start.json.gz').write_bytes(gzip.compress(b'[]'))
    if damage=='calls':
        p=dirs[1]/'model-usage.jsonl';r=json.loads(p.read_text());r['call']=1;p.write_text(json.dumps(r)+'\n')
    if damage=='calibration':
        p=dirs[1]/'context-estimates.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()];rows=[r for r in rows if r.get('event')!='calibration'];p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if damage=='thread':(dirs[1]/'continuity.json').write_text(json.dumps({'thread_id':'new-thread'}))
    if damage=='replay':
        p=dirs[0]/'verification.json';r=json.loads(p.read_text());r['trials'][0]['infrastructure_valid']=False;p.write_text(json.dumps(r))
    if damage=='missing_trial':(dirs[0]/'context-end.json.gz').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        reconstruct_from_receipts(parent,dirs[-1],tmp_path/'imported',boundary=boundary)
    assert not (tmp_path/'imported').exists()


def test_external_visual_reservation_survives_exact_restore_unchanged(tmp_path,monkeypatch):
    model,config,_=client(tmp_path,monkeypatch)
    identity=model.ledger.reserve(2.179072,label='Gemini visual calibration historical-positive')
    before=Path(config['ledger_path']).read_bytes()
    receipt=capture(model,tmp_path/'checkpoint',boundary=BOUNDARY)
    clone=resumed(receipt['checkpoint'],receipt,config,tmp_path)
    assert Path(config['ledger_path']).read_bytes()==before
    ledger=json.loads(before)
    assert ledger['reserved'][identity]['amount_usd']==2.179072
    assert clone.ledger.path==model.ledger.path and clone.ledger.limit_usd==150
    from pairing.adapt_resume import _check_reserved_spend
    _check_reserved_spend(ledger,{identity:ledger['reserved'][identity]})
    assert Path(config['ledger_path']).read_bytes()==before


def completed_timeout_fixture(tmp_path,monkeypatch):
    import time
    model,config,parent,attempt=reconstruction_fixture(tmp_path,monkeypatch)
    normal_post=OpenRouterLearningModel._post
    def timed_out(*args,**kwargs):raise TimeoutError('offline post timeout')
    monkeypatch.setattr(OpenRouterLearningModel,'_post',staticmethod(timed_out))
    model.approximate_counter.native.count_native_request=lambda req,timeout:len(req['contents'])*10
    before=list(model.messages);started=time.time()
    with pytest.raises(TimeoutError):model.ask('Unanswered final trial input',schema=SCHEMA)
    ended=time.time()
    assert model.messages==before and model.calls==1 and len(model.usage)==1
    model.save_context(attempt/'context-end.json.gz')
    (attempt/'report.json').write_text(json.dumps({'termination_reason':'model_deadline','observed_success':False}))
    (attempt/'timing.jsonl').write_text(json.dumps({'event':'model_start','turn':1,'epoch':started})+'\n'+json.dumps({'event':'model_timeout','turn':1,'epoch':ended})+'\n')
    held=model.ledger.snapshot()['reserved'];assert len(held)==1
    ident,row=next(iter(held.items()))
    approval={'reservation_id':ident,**row,'attempt_id':attempt.name,'pending_call':2}
    boundary={**BOUNDARY,'completed_timeout_reservations':[approval]}
    monkeypatch.setattr(OpenRouterLearningModel,'_post',staticmethod(normal_post))
    return model,config,parent,attempt,boundary,ident,row


def test_completed_timeout_exact_continuation_matches_uninterrupted_no_retry(tmp_path,monkeypatch):
    from pairing.openrouter_checkpoint import reconstruct_from_receipts
    model,config,parent,attempt,boundary,ident,hold=completed_timeout_fixture(tmp_path,monkeypatch)
    prospective=capture(model,tmp_path/'prospective',boundary=boundary)
    receipt=reconstruct_from_receipts(parent,attempt,tmp_path/'imported',boundary=boundary)
    original=json.loads(Path(prospective['checkpoint']).read_text());derived=json.loads(Path(receipt['checkpoint']).read_text())
    for field in ('state','active','counter','ledger','boundary'):assert original[field]==derived[field]
    requests=[];counts=[]
    def post(url,body,timeout):
        requests.append(body)
        return {'model':'google/gemini-4-argon','choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps({'operation':'finish','path':'','content':'','message':'Next scheduled trial'})}}],'usage':{'prompt_tokens':20,'completion_tokens':4,'cost':0.01}}
    monkeypatch.setattr(OpenRouterLearningModel,'_post',staticmethod(post))
    model.directory=tmp_path/'live/trial-2';model.directory.mkdir(parents=True)
    model.approximate_counter.native.count_native_request=lambda req,timeout:(counts.append(req) or len(req['contents'])*10)
    model.ask('Next scheduled trial reset',schema=SCHEMA)
    expected=requests[-1];expected_counts=list(counts)
    clone=restore(receipt['checkpoint'],checkpoint_sha256=receipt['sha256'],destination=tmp_path/'restored/trial-2',key='offline',expected_participant=config,expected_instructions='Original instructions'+INSTRUCTION,expected_boundary=boundary)
    clone_counts=[];clone.approximate_counter.native.count_native_request=lambda req,timeout:(clone_counts.append(req) or len(req['contents'])*10)
    clone.ask('Next scheduled trial reset',schema=SCHEMA)
    assert requests[-1]==expected and clone_counts==expected_counts
    assert clone.messages==model.messages and clone.calls==model.calls==2
    assert 'Unanswered final trial input' not in json.dumps(requests[-1])
    assert clone.ledger.snapshot()['reserved'][ident]==hold
    assert original['counter']['last_request_sha256']!=json.loads((attempt/'context-estimates.jsonl').read_text().splitlines()[2])['transformed_request_sha256']


@pytest.mark.parametrize('damage',['termination','hold_changed','hold_removed','missing_usage','wrong_call','arbitrary_error','missing_timeout'])
def test_timeout_exception_does_not_admit_arbitrary_unknown_state(tmp_path,monkeypatch,damage):
    from pairing.openrouter_checkpoint import reconstruct_from_receipts
    model,config,parent,attempt,boundary,ident,hold=completed_timeout_fixture(tmp_path,monkeypatch)
    if damage=='termination':(attempt/'report.json').write_text(json.dumps({'termination_reason':'agent_finish'}))
    if damage in {'hold_changed','hold_removed'}:
        path=Path(config['ledger_path']);ledger=json.loads(path.read_text())
        if damage=='hold_removed':ledger['reserved'].pop(ident)
        else:ledger['reserved'][ident]['amount_usd']=0.1
        path.write_text(json.dumps(ledger))
    if damage=='missing_usage':(attempt/'model-usage.jsonl').write_text('')
    if damage=='wrong_call':boundary['completed_timeout_reservations'][0]['pending_call']=3
    if damage=='arbitrary_error':(attempt/'provider-http-errors.jsonl').write_text('{"status":402}\n')
    if damage=='missing_timeout':(attempt/'timing.jsonl').write_text('')
    with pytest.raises(ValueError):reconstruct_from_receipts(parent,attempt,tmp_path/'imported',boundary=boundary)
