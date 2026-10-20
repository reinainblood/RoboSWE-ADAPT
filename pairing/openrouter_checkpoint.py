"""Explicit private checkpoints for certified between-trial OpenRouter continuation.

No provider/simulator calls. Orchestration must independently certify the trial
boundary, exclusive ownership, complete workspace restore, and unchanged plan.
Never use the sanitized public episode export as a continuation checkpoint.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

from pairing.context_history import HistoryStore, encoded
from pairing.openrouter_model import OpenRouterLearningModel
from pairing.openrouter_google_count import GoogleMappedCounter
from pairing.anthropic_model import SpendLedger

POLICY = 'sqlite-sliding-or-google-count-v2'
IMPLEMENTATIONS = ('pairing/openrouter_model.py','pairing/openai_model.py','pairing/openrouter_google_count.py','pairing/gemini_token_count.py','pairing/context_history.py')
PROJECT = Path(__file__).resolve().parents[1]
FIELDS = ('input_price','output_price','provider_label','model','instructions','context_window',
          'max_output_tokens','effort','provider','calls','compactions','messages','usage',
          'thread_id','metadata','history_policy','history_attribute')
TRANSIENT = {'key','directory','ledger','history_store','approximate_counter','_history_deadline'}
OPTIONAL = {'last_provider_thinking_trace'}
COUNTER_FIELDS = ('max_output_tokens','effort','multiplier','last_native_count','last_request_sha256')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _boundary(value):
    if not isinstance(value,dict) or not value.get('parent_job_id'):
        raise ValueError('Explicit original series boundary required')
    completed=value.get('completed_trials');nxt=value.get('next_trial_number')
    if type(completed) is not int or not 1<=completed<5 or nxt!=completed+1:
        raise ValueError('Resume only between original five numbered trials')
    ids=value.get('completed_attempt_ids')
    if not isinstance(ids,list) or len(ids)!=completed or len(set(ids))!=completed:
        raise ValueError('Every completed attempt must be identified')
    for ident in [value['parent_job_id'],*ids,*([value['zero_input_prefix_job_id']] if value.get('zero_input_prefix_job_id') else [])]:
        if not isinstance(ident,str) or Path(ident).name!=ident or ident in {'.','..'}:
            raise ValueError('Boundary job identities must be plain directory names')
    return copy.deepcopy(value)


def _validate_db(path, active):
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
            raise ValueError('History SQLite integrity failure')
        rows=db.execute('SELECT id,body,sha256 FROM messages ORDER BY id').fetchall()
        for _,body,digest in rows:
            if hashlib.sha256(body.encode()).hexdigest()!=digest:
                raise ValueError('History message digest mismatch')
        # Every active message must be an order-preserving archived message.
        pos=0
        for body in active:
            while pos<len(rows) and rows[pos][1]!=body:pos+=1
            if pos==len(rows):raise ValueError('Active window is not archived in order')
            pos+=1
        if not active or [row[1] for row in rows[-len(active):]]!=active:
            raise ValueError('OpenRouter active window must be the exact archive suffix')
        return {'message_count':len(rows),'slide_count':db.execute('SELECT COUNT(*) FROM slides').fetchone()[0]}



def _validate_state(state):
    if state['model']!='google/gemini-4-argon' or state['history_policy']!=POLICY or state['history_attribute']!='messages':
        raise ValueError('Only pinned Gemini mapped-count v2 continuation supported')
    if state['compactions']!=0 or type(state['calls']) is not int or state['calls']<1 or len(state['usage'])!=state['calls']:
        raise ValueError('Incomplete client accounting or unsupported compaction')
    metadata=state['metadata']
    for recorded,attribute in [('thread_id','thread_id'),('model','model'),('context_window','context_window'),('max_output_tokens','max_output_tokens'),('reasoning_effort','effort'),('provider_pin','provider')]:
        if metadata.get(recorded)!=state[attribute]:raise ValueError('Client metadata disagrees: '+recorded)
    if metadata.get('context_policy',{}).get('method')!=state['history_policy']:
        raise ValueError('History policy metadata mismatch')


def _validate_counter(counter, state):
    if set(counter)!=set(COUNTER_FIELDS):raise ValueError('Missing or unsupported counter state')
    if counter['max_output_tokens']!=state['max_output_tokens'] or counter['effort']!=state['effort']:
        raise ValueError('Counter request settings changed')
    factor=counter['multiplier']
    if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or factor<1.3:
        raise ValueError('Invalid retained calibration multiplier')
    if type(counter['last_native_count']) is not int or counter['last_native_count']<0:
        raise ValueError('Last native count unavailable')
    value=counter['last_request_sha256']
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Last count request digest unavailable')


def _certified_timeout_holds(boundary, ledger, jobs_root, state):
    """Known local timeout state is distinct from unresolved provider billing."""
    from roboswe.adaptive_results import _attempt
    approvals=boundary.get('completed_timeout_reservations',[])
    if not isinstance(approvals,list):raise ValueError('Timeout reservations must be an exact list')
    allowed={}
    approved_attempts=set()
    for item in approvals:
        fields={'reservation_id','amount_usd','label','created_epoch','attempt_id','pending_call'}
        if not isinstance(item,dict) or set(item)!=fields:
            raise ValueError('Incomplete completed-timeout reservation identity')
        ident=item['reservation_id'];attempt_id=item['attempt_id'];call=item['pending_call']
        if ident in allowed or attempt_id in approved_attempts or attempt_id not in boundary['completed_attempt_ids'] or type(call) is not int:
            raise ValueError('Duplicate or noncompleted timeout reservation')
        expected={k:item[k] for k in ('amount_usd','label','created_epoch')}
        if ledger.get('reserved',{}).get(ident)!=expected:
            raise ValueError('Completed-timeout ledger hold changed or disappeared')
        if item['label']!=f"{state['model']}:{attempt_id}:call{call}":
            raise ValueError('Timeout hold label does not identify the completed trial request')
        maximum=(state['context_window']*state['input_price']+state['max_output_tokens']*state['output_price'])/1_000_000
        if item['amount_usd']!=maximum:
            raise ValueError('Timeout hold must retain the entire original maximum charge')
        from zipfile import ZipFile
        with ZipFile(Path(jobs_root)/boundary['parent_job_id']/'source.zip') as archive:
            if any(archive.read(name)!=(PROJECT/name).read_bytes() for name in IMPLEMENTATIONS):
                raise ValueError('Timeout mutation ordering differs from pinned original client source')
        directory=Path(jobs_root)/attempt_id
        job=json.loads((directory/'job.json').read_text());report=json.loads((directory/'report.json').read_text())
        if job.get('campaign_id')!=boundary['parent_job_id'] or not job.get('finished_at') or job.get('status')!='verified_fail' or report.get('termination_reason')!='model_deadline':
            raise ValueError('Only a completed replay-verified model deadline permits this hold')
        if _attempt(directory,job)['category']!='verified_fail':
            raise ValueError('Timeout trial replay not valid')
        response_rows=[json.loads(line) for line in (directory/'provider-responses.jsonl').read_text().splitlines()]
        usage_rows=[json.loads(line) for line in (directory/'model-usage.jsonl').read_text().splitlines()]
        if not response_rows or [r['call'] for r in response_rows]!=[r['call'] for r in usage_rows] or response_rows[-1]['call']+1!=call:
            raise ValueError('Timeout must follow the last complete accepted response, with no missing usage')
        timing=[json.loads(line) for line in (directory/'timing.jsonl').read_text().splitlines()]
        model_events=[r for r in timing if r.get('event') in {'model_start','model_end','model_timeout','model_error'}]
        if len(model_events)<2 or [r['event'] for r in model_events[-2:]]!=['model_start','model_timeout'] or model_events[-1].get('turn')!=model_events[-2].get('turn'):
            raise ValueError('No terminal pending-request model timeout evidence')
        begin,end=model_events[-2:]
        if not begin.get('epoch',float('inf'))<=item['created_epoch']<=end.get('epoch',float('-inf')):
            raise ValueError('Reservation does not fall within the final timed-out request')
        counts=[json.loads(line) for line in (directory/'context-estimates.jsonl').read_text().splitlines()]
        if not counts or counts[-1].get('event')!='count':
            raise ValueError('Timeout has no completed final count receipt')
        for name in ('provider-http-errors.jsonl','unsettled-spend.jsonl','provider-unsettled-responses.jsonl'):
            path=directory/name
            if path.exists() and path.stat().st_size:raise ValueError('Arbitrary provider error is not a completed timeout')
        allowed[ident]=expected
        approved_attempts.add(attempt_id)
    for ident,row in ledger.get('reserved',{}).items():
        if any(job in row.get('label','') for job in boundary['completed_attempt_ids']) and ident not in allowed:
            raise ValueError('Uncertified actor reservation prevents continuation')
    return allowed


def capture(model, destination, *, boundary):
    """Capture an idle live client; caller must hold the certified boundary lock."""
    if type(model) is not OpenRouterLearningModel or type(model.approximate_counter) is not GoogleMappedCounter:
        raise ValueError('Unsupported client/counter type')
    unknown=set(vars(model))-set(FIELDS)-TRANSIENT-OPTIONAL
    if unknown:raise ValueError('Uncheckpointed client fields: '+','.join(sorted(unknown)))
    state={key:copy.deepcopy(getattr(model,key)) for key in FIELDS}
    for key in OPTIONAL:
        if hasattr(model,key):state[key]=copy.deepcopy(getattr(model,key))
    _validate_state(state)
    active=[encoded(message) for message in model.messages]
    if model.history_store.active!=active:
        raise ValueError('History is not synchronized at an idle boundary')
    if model.ledger is None or not model.ledger.path.is_file():
        raise ValueError('Existing shared spend ledger required')
    boundary=_boundary(boundary)
    ledger=model.ledger.snapshot()
    _certified_timeout_holds(boundary,ledger,Path(model.directory).parent,state)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False,mode=0o700)
    model.history_store.export(destination)
    db=destination/'history.sqlite';os.chmod(db,0o600)
    verification=_validate_db(db,active)
    payload={'version':1,'kind':'gemini_between_trial_client_checkpoint','boundary':boundary,
        'state':state,'original_directory':str(model.directory),'active':active,
        'history':{'sha256':sha(db),'policy':model.history_store.policy,'window':model.history_store.window,
                   'reserve':model.history_store.reserve,'chunk':model.history_store.chunk,**verification},
        'counter':{key:copy.deepcopy(getattr(model.approximate_counter,key)) for key in COUNTER_FIELDS},
        'ledger':{'path':str(model.ledger.path.resolve()),'limit_usd':model.ledger.limit_usd},
        'implementation_sha256':{name:sha(PROJECT/name) for name in IMPLEMENTATIONS},
        'scope':'Exact private client state; orchestration/workspace/fresh-reset certification separate. No credentials serialized.'}
    _validate_counter(payload['counter'],state)
    raw=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    path=destination/'checkpoint.json'
    with path.open('xb') as stream:stream.write(raw)
    os.chmod(path,0o600)
    return {'checkpoint':str(path),'sha256':sha(path),'history_sha256':sha(db)}


def restore(checkpoint, *, checkpoint_sha256, destination, key, expected_participant,
            expected_instructions, expected_boundary):
    """Restore into a new private directory without initializing or summarizing history."""
    checkpoint=Path(checkpoint)
    if sha(checkpoint)!=checkpoint_sha256:raise ValueError('Checkpoint digest mismatch')
    payload=json.loads(checkpoint.read_text())
    if payload.get('kind')!='gemini_between_trial_client_checkpoint' or payload.get('version')!=1:
        raise ValueError('Unsupported checkpoint format')
    if payload.get('implementation_sha256')!={name:sha(PROJECT/name) for name in IMPLEMENTATIONS}:
        raise ValueError('Client/history/counter implementation changed since checkpoint')
    if payload['boundary']!=_boundary(expected_boundary):raise ValueError('Boundary mismatch')
    state=payload['state'];_validate_state(state)
    expected={'model':expected_participant['model'],'context_window':expected_participant['context_window'],
        'max_output_tokens':expected_participant['max_output_tokens'],'effort':expected_participant.get('effort','high'),
        'provider':expected_participant.get('upstream_provider'),'history_policy':expected_participant['context_policy'],
        'input_price':expected_participant.get('input_price_per_million',2.0),
        'output_price':expected_participant.get('output_price_per_million',10.0),'instructions':expected_instructions}
    if expected_participant.get('provider')!='openrouter' or any(state.get(k)!=v for k,v in expected.items()):
        raise ValueError('Provider-visible settings or effective instructions changed')
    ledger=payload['ledger']
    if Path(expected_participant['ledger_path']).resolve()!=Path(ledger['path']) or float(expected_participant['spend_limit_usd'])!=ledger['limit_usd'] or not Path(ledger['path']).is_file():
        raise ValueError('Existing shared ledger path/cap must be retained')
    history=payload['history'];counter=payload['counter']
    _validate_counter(counter,state)
    if history['policy']!=state['history_policy'] or history['window']!=state['context_window'] or history['reserve']!=state['max_output_tokens'] or counter['max_output_tokens']!=state['max_output_tokens'] or counter['effort']!=state['effort']:
        raise ValueError('History/counter configuration disagrees with client')
    _certified_timeout_holds(payload['boundary'],json.loads(Path(ledger['path']).read_text()),Path(payload['original_directory']).parent,state)
    source_db=checkpoint.parent/'history.sqlite'
    if sha(source_db)!=payload['history']['sha256']:raise ValueError('History snapshot digest mismatch')
    if payload['active']!=[encoded(m) for m in state['messages']]:raise ValueError('Active window mismatch')
    _validate_db(source_db,payload['active'])
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False,mode=0o700)
    new_db=destination/'history.sqlite'
    with sqlite3.connect(source_db.resolve().as_uri()+'?mode=ro',uri=True) as original, sqlite3.connect(new_db) as target:
        original.backup(target)
    os.chmod(new_db,0o600)
    # Deliberately bypass ordinary constructors: no new UUID, appended
    # instructions, fresh archive, calibration reset, or duplicate history rows.
    model=OpenRouterLearningModel.__new__(OpenRouterLearningModel)
    for name,value in state.items():setattr(model,name,copy.deepcopy(value))
    model.key=key;model.directory=destination
    model.ledger=SpendLedger(ledger['path'],limit_usd=ledger['limit_usd'])
    model.approximate_counter=GoogleMappedCounter(destination,max_output_tokens=state['max_output_tokens'],effort=state['effort'],deadline=lambda:getattr(model,'_history_deadline',time.monotonic()+25))
    for name,value in payload['counter'].items():setattr(model.approximate_counter,name,value)
    store=HistoryStore.__new__(HistoryStore)
    history=payload['history'];store.path=new_db;store.policy=history['policy'];store.window=history['window'];store.reserve=history['reserve'];store.chunk=history['chunk'];store.active=list(payload['active']);store.token_counter=model.approximate_counter
    model.history_store=store
    return model


def reconstruct_from_receipts(parent, attempt, destination, *, boundary):
    """Strict completed-prefix legacy capture import; no guessing missing client fields.

    Uses the original source archive to certify constructor behavior and the
    final raw receipts to reconstruct each field. Supports one to four completed
    trials at a boundary with an optional completely zero-input next-trial prefix.
    """
    import gzip
    from zipfile import ZipFile
    parent,attempt=Path(parent),Path(attempt)
    boundary=_boundary(boundary)
    if boundary['completed_attempt_ids'][-1]!=attempt.name or boundary['parent_job_id']!=parent.name:
        raise ValueError('Final completed attempt must match the original series boundary')
    parent_job=json.loads((parent/'job.json').read_text())
    directories=[parent.parent/ident for ident in boundary['completed_attempt_ids']]
    metadata=json.loads((parent/'model.json').read_text())
    prior_end=[]
    archive_prefixes=[]
    for number,directory in enumerate(directories,1):
        job=json.loads((directory/'job.json').read_text())
        verification=json.loads((directory/'verification.json').read_text())
        if job.get('campaign_id')!=parent.name or job.get('attempt_number')!=number or job.get('status') not in {'verified_pass','verified_fail'} or not job.get('finished_at') or job.get('artifact_errors'):
            raise ValueError('Completed attempt is not finalized, ordered and verified')
        trials=verification.get('trials',[])
        if verification.get('reproduced') is not True or verification.get('mismatches') or len(trials)!=1:
            raise ValueError('Completed attempt replay is not certified')
        trial=trials[0];expected=job['status']=='verified_pass'
        if trial.get('reproduced') is not True or trial.get('infrastructure_valid') is not True or trial.get('precondition_valid') is not True or trial.get('replay',{}).get('precondition_valid') is not True or any(v is not expected for v in [verification.get('verified_success'),trial.get('verified_success'),trial.get('observed_success'),trial.get('replay',{}).get('observed_success')]):
            raise ValueError('Completed replay verdict/precondition disagreement')
        continuity=json.loads((directory/'continuity.json').read_text())
        if continuity.get('thread_id')!=metadata.get('thread_id'):
            raise ValueError('Completed trial changed the original client thread')
        start=json.loads(gzip.decompress((directory/'context-start.json.gz').read_bytes()))
        end=json.loads(gzip.decompress((directory/'context-end.json.gz').read_bytes()))
        if start!=prior_end:raise ValueError('Between-trial provider history continuity broken')
        prior_end=end
        _validate_db(directory/'history.sqlite',[encoded(m) for m in end])
        manifest=json.loads((directory/'history-manifest.json').read_text())
        if manifest.get('sha256')!=sha(directory/'history.sqlite'):
            raise ValueError('Completed trial history manifest digest mismatch')
        with sqlite3.connect((directory/'history.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
            archive_prefixes.append((db.execute('SELECT id,trial,epoch,role,body,sha256 FROM messages ORDER BY id').fetchall(),db.execute('SELECT id,trial,epoch,removed_messages,estimated_removed_tokens,estimated_before_tokens,estimated_after_tokens FROM slides ORDER BY id').fetchall()))
    final_messages,final_slides=archive_prefixes[-1]
    for old_messages,old_slides in archive_prefixes:
        if final_messages[:len(old_messages)]!=old_messages or final_slides[:len(old_slides)]!=old_slides:
            raise ValueError('Final archive does not preserve completed trial history/slide prefix')
    prefix=boundary.get('zero_input_prefix_job_id')
    if parent_job.get('current_job')!=attempt.name:
        if not prefix or parent_job.get('current_job')!=prefix:
            raise ValueError('Unaccounted later trial exists')
        prefix_dir=parent.parent/prefix;prefix_job=json.loads((prefix_dir/'job.json').read_text())
        if prefix_job.get('campaign_id')!=parent.name or prefix_job.get('attempt_number')!=boundary['next_trial_number']:
            raise ValueError('Zero-input prefix identity mismatch')
        for name in ('model-transcript.jsonl','provider-responses.jsonl','model-usage.jsonl','provider-request-events.jsonl','token-counts.jsonl','context-estimates.jsonl','actions.jsonl'):
            path=prefix_dir/name
            if path.exists() and path.stat().st_size:
                raise ValueError('Next-trial prefix has input/activity evidence: '+name)
        if (prefix_dir/'report.json').exists() or (prefix_dir/'verification.json').exists():
            raise ValueError('Next-trial prefix already has result evidence')
    plan=json.loads((parent/'plan.json').read_text());config=plan['participant']
    if config.get('provider')!='openrouter' or config.get('context_policy')!=POLICY:
        raise ValueError('Unsupported original provider policy')
    # Byte-identical relevant implementation is required: no inference about
    # what an older constructor, serializer, or counter might have done.
    project=Path(__file__).resolve().parents[1]
    implementations=IMPLEMENTATIONS
    with ZipFile(parent/'source.zip') as archive:
        for name in implementations:
            if archive.read(name)!=(project/name).read_bytes():
                raise ValueError('Original implementation differs; reconstruction not certified: '+name)
    messages=json.loads(gzip.decompress((attempt/'context-end.json.gz').read_bytes()))
    active=[encoded(m) for m in messages]
    db_path=attempt/'history.sqlite';_validate_db(db_path,active)
    history_manifest=json.loads((attempt/'history-manifest.json').read_text())
    if history_manifest.get('sha256')!=sha(db_path):raise ValueError('Final history manifest digest mismatch')
    def rows(name):
        return [json.loads(line) for directory in directories for line in (directory/name).read_text().splitlines()]
    usage_rows=rows('model-usage.jsonl');responses=rows('provider-responses.jsonl')
    calls=len(usage_rows)
    if not calls or [r.get('call') for r in usage_rows]!=list(range(1,calls+1)) or [r.get('call') for r in responses]!=list(range(1,calls+1)):
        raise ValueError('Complete unique cumulative provider call sequence required')
    for usage,response in zip(usage_rows,responses):
        payload=response['response']
        if payload.get('model')!='google/gemini-4-argon' or payload.get('usage')!=usage['usage'] or payload.get('error'):
            raise ValueError('Provider accounting receipts disagree')
    estimates=rows('context-estimates.jsonl')
    calibrations=[r for r in estimates if r.get('event')=='calibration']
    if [r.get('call') for r in calibrations]!=list(range(1,calls+1)):
        raise ValueError('Complete calibration sequence required')
    multiplier=1.3
    for row,usage in zip(calibrations,usage_rows):
        actual=row.get('actual_prompt_tokens');native=row.get('native_input_tokens')
        if actual!=usage['usage'].get('prompt_tokens') or type(native) is not int or native<0:
            raise ValueError('Calibration does not match provider token receipt')
        if row.get('multiplier_before')!=multiplier:
            raise ValueError('Calibration continuity missing')
        expected=max(multiplier,1.1*actual/native) if native else multiplier
        if row.get('multiplier_after')!=expected:raise ValueError('Calibration update cannot be exactly reconstructed')
        multiplier=row['multiplier_after']
    final_count=next((r for r in reversed(estimates) if r.get('event')=='count'),None)
    if final_count is None:raise ValueError('Last native count unavailable')
    latest_completed_timeout=any(item.get('attempt_id')==attempt.name for item in boundary.get('completed_timeout_reservations',[]))
    if latest_completed_timeout:
        if estimates[-1].get('event')!='count' or final_count.get('multiplier')!=multiplier:
            raise ValueError('Timed-out final count must preserve last accepted calibration')
    elif calibrations[-1].get('transformed_request_sha256')!=final_count.get('transformed_request_sha256') or calibrations[-1].get('native_input_tokens')!=final_count.get('native_input_tokens'):
        raise ValueError('Last native count/calibration state unavailable')
    # Do not permit an unknown provider outcome after the last accounted call.
    for directory in directories:
        for name in ('unsettled-usage.jsonl','unsettled-spend.jsonl','provider-unsettled-responses.jsonl'):
            path=directory/name
            if path.exists() and path.stat().st_size:raise ValueError('Unsettled provider evidence prevents reconstruction')
    with sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        archived=[json.loads(row[0]) for row in db.execute("SELECT body FROM messages WHERE role='assistant' ORDER BY id")]
    expected_assistants=[]
    for response in responses:
        choices=response['response'].get('choices',[])
        if len(choices)!=1:raise ValueError('Ambiguous provider response')
        message=choices[0]['message']
        history={key:message[key] for key in ('role','content','reasoning','reasoning_details','refusal') if key in message}
        if history.get('refusal') is None:history.pop('refusal',None)
        history.setdefault('role','assistant');expected_assistants.append(history)
    if archived!=expected_assistants:
        raise ValueError('SQLite assistant history disagrees with complete provider responses')
    metadata=json.loads((parent/'model.json').read_text())
    from pairing.context_history import INSTRUCTION
    model=OpenRouterLearningModel.__new__(OpenRouterLearningModel)
    model.input_price=float(config.get('input_price_per_million',2.0));model.output_price=float(config.get('output_price_per_million',10.0))
    model.provider_label='OpenRouter';model.key='';model.model=config['model'];model.instructions=plan['prompt']+INSTRUCTION
    model.directory=attempt;model.context_window=config['context_window'];model.max_output_tokens=config['max_output_tokens'];model.effort=config.get('effort','high');model.provider=config.get('upstream_provider')
    model.calls=calls;model.compactions=0;model.messages=messages;model.usage=[r['usage'] for r in usage_rows]
    model.thread_id=metadata['thread_id'];model.metadata=metadata;model.history_policy=POLICY;model.history_attribute='messages'
    last=responses[-1]['response']['choices'][0]['message']
    model.last_provider_thinking_trace={'call':calls,**{k:last[k] for k in ('reasoning','reasoning_details') if k in last},'policy':'Verbatim provider-exposed fields; encrypted blocks are opaque, not readable thoughts'}
    ledger_path=Path(config['ledger_path'])
    if not ledger_path.is_file():raise ValueError('Original ledger missing')
    model.ledger=SpendLedger(ledger_path,limit_usd=config['spend_limit_usd'])
    counter=GoogleMappedCounter.__new__(GoogleMappedCounter)
    counter.max_output_tokens=model.max_output_tokens;counter.effort=model.effort;counter.multiplier=multiplier
    counter.last_native_count=final_count['native_input_tokens'];counter.last_request_sha256=final_count['transformed_request_sha256']
    model.approximate_counter=counter
    store=HistoryStore.__new__(HistoryStore);store.path=db_path;store.policy=POLICY;store.window=model.context_window;store.reserve=model.max_output_tokens
    store.chunk=metadata['context_policy']['eviction_chunk'];store.active=active;store.token_counter=counter
    model.history_store=store
    receipt=capture(model,destination,boundary=boundary)
    receipt['reconstruction']='strict original-source-and-final-receipt reconstruction; no model request'
    return receipt


def validate_against_receipts(checkpoint, parent, attempt, *, boundary):
    """Cross-bind a supplied checkpoint to fresh exact original-receipt derivation.

    Private temporary SQLite/output files only. No model construction, provider
    call, simulator, original history modification, or ledger update occurs.
    Caller still verifies the certificate's digest of the supplied checkpoint.
    """
    import tempfile
    checkpoint=Path(checkpoint)
    supplied=json.loads(checkpoint.read_text())
    with tempfile.TemporaryDirectory(prefix='roboswe-checkpoint-certification-') as temporary:
        receipt=reconstruct_from_receipts(parent,attempt,Path(temporary)/'derived',boundary=boundary)
        derived=json.loads(Path(receipt['checkpoint']).read_text())
        for field in ('version','kind','boundary','state','original_directory','active','counter','ledger','implementation_sha256'):
            if supplied.get(field)!=derived.get(field):
                raise ValueError('Checkpoint differs from exact original receipts: '+field)
        # SQLite backup page layout is not semantic state. Compare all logical
        # archive tables exactly, including IDs, epochs, trial labels and policy.
        def logical(path):
            with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as db:
                if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
                    raise ValueError('Checkpoint SQLite integrity failure')
                return {
                    'messages':db.execute('SELECT id,trial,epoch,role,body,sha256 FROM messages ORDER BY id').fetchall(),
                    'slides':db.execute('SELECT id,trial,epoch,removed_messages,estimated_removed_tokens,estimated_before_tokens,estimated_after_tokens FROM slides ORDER BY id').fetchall(),
                    'metadata':db.execute('SELECT key,value FROM metadata ORDER BY key').fetchall(),
                }
        supplied_db=checkpoint.parent/'history.sqlite'
        if sha(supplied_db)!=supplied['history']['sha256']:
            raise ValueError('Checkpoint SQLite differs from bound snapshot digest')
        if logical(supplied_db)!=logical(Path(receipt['checkpoint']).parent/'history.sqlite'):
            raise ValueError('Checkpoint history differs from exact original archive')
        for key in ('policy','window','reserve','chunk','message_count','slide_count'):
            if supplied['history'].get(key)!=derived['history'].get(key):
                raise ValueError('Checkpoint history settings differ: '+key)
    return {'certified_against_original_receipts':True,'checkpoint_sha256':sha(checkpoint),
            'parent_job_id':boundary['parent_job_id'],'completed_attempt_ids':boundary['completed_attempt_ids'],
            'comparison':'Exact client state, active window, calibration, accounting, thread identity, source pins, and all logical SQLite rows; no approximations'}
