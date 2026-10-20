"""Explicit continuation of a certified zero-input boundary; never repeat a trial."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def continuation_artifacts(certificate):
    """Exact private files requiring separate immutable remote readback."""
    certificate=Path(certificate).resolve();cert=read(certificate)
    checkpoint=Path(cert['model_checkpoint']).resolve()
    paths={certificate,Path(cert['workspace']).resolve(),checkpoint,
           checkpoint.parent/'history.sqlite',checkpoint.parent/'history-manifest.json'}
    paths.update(Path(p).resolve() for p in cert['evidence_sha256'])
    return [{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(paths)]


def _durability(certificate, cert):
    receipt=read(cert['continuation_archive_receipt'])
    if (receipt.get('schema_version')!='continuation-durability-v1'
        or receipt.get('environment')!='FDR' or receipt.get('modal_profile')!='snorkelai'):
        raise ValueError('Separate continuation bundle durability receipt required')
    rows=receipt.get('files',[])
    if len({r.get('path') for r in rows})!=len(rows):raise ValueError('Duplicate continuation durability entry')
    indexed={r['path']:r for r in rows}
    for expected in continuation_artifacts(certificate):
        row=indexed.get(expected['path'],{})
        if (any(row.get(k)!=v for k,v in expected.items())
            or row.get('remote_readback_verified') is not True or not row.get('remote_path')):
            raise ValueError('Continuation artifact durable readback missing or mismatched')


def _stop_check(parent, campaign_root):
    if any(p.exists() for p in (Path(campaign_root)/'STOP',Path(parent)/'STOP',Path(parent)/'resume-history/STOP')):
        raise ValueError('Explicit STOP blocks continuation')


def validate(parent, certificate):
    from roboswe.adaptive_results import _attempt
    from roboswe.execution.workspace_checkpoint import inspect_archive
    from pairing.adapt_modal_pair import pid_live
    parent=Path(parent).resolve();certificate=Path(certificate).resolve();cert=read(certificate)
    if cert.get('schema_version')!='certified-adaptive-boundary-v1':raise ValueError('Unsupported resume certificate')
    _stop_check(parent,cert['campaign_root'])
    boundary=cert['boundary'];job=read(parent/'job.json');plan=read(parent/'plan.json')
    count=boundary.get('completed_trials')
    ids=boundary.get('completed_attempt_ids',[])
    if (boundary.get('parent_job_id')!=parent.name or job.get('id')!=parent.name
        or plan.get('protocol')!='adaptive-public-sqlite-v1' or plan.get('trials')!=5
        or plan['participant'].get('provider')!='openrouter'
        or plan['participant'].get('context_policy')!='sqlite-sliding-or-google-count-v2'
        or type(count) is not int or not 1<=count<5 or boundary.get('next_trial_number')!=count+1
        or not isinstance(ids,list) or not all(isinstance(i,str) for i in ids) or len(ids)!=count or len(set(ids))!=count
        or job.get('status')!='infrastructure_error'):
        raise ValueError('Only certified between-trial Gemini boundaries supported')
    if sha(parent/'plan.json')!=cert['plan_sha256']:raise ValueError('Original plan changed')
    for path,digest in cert['evidence_sha256'].items():
        if sha(path)!=digest:raise ValueError('Certified boundary evidence changed')
    watcher=Path(cert['watcher_directory']).resolve()
    required=[parent/'job.json',parent/'plan.json',watcher/'active.json',watcher/'checkpoint-provenance.json',watcher/'pre-signal-boundary.json']
    state=read(watcher/'active.json');provenance=read(watcher/'checkpoint-provenance.json');proof=read(watcher/'pre-signal-boundary.json')
    if state.get('status')!='preserved_and_stopped_at_boundary' or not all(state.get(k) is True for k in ('signal_sent','queue_terminal','worker_terminal','pending_no_model_input_after_stop')):
        raise ValueError('Owned boundary stop not certified')
    if state.get('parent')!=parent.name or proof.get('parent',{}).get('id')!=parent.name or proof.get('no_new_model_input') is not True:
        raise ValueError('Boundary ownership mismatch')
    for key in ('queue_pid','worker_pid'):
        if type(state.get(key)) is not int or state[key]<=0 or pid_live(state[key]):raise ValueError('Original provider processes must be terminal')
    proofs=proof.get('completed_trials')
    if proofs is None and count==1:
        proofs=[{'job_id':ids[0],'job_sha256':proof.get('trial1_job_sha256'),
                 'verification_sha256':proof.get('trial1_verification_sha256')}]
    if not isinstance(proofs,list) or [p.get('job_id') for p in proofs]!=ids:
        raise ValueError('Every completed trial needs ordered boundary proof')
    archives=cert.get('completed_archive_receipts')
    if archives is None and count==1:
        archives={ids[0]:cert.get('completed_archive_receipt')}
    if not isinstance(archives,dict) or set(archives)!=set(ids):
        raise ValueError('Every completed trial needs a durable archive')
    completed_directories=[]
    for number,(completed_id,trial_proof) in enumerate(zip(ids,proofs),1):
        if not isinstance(completed_id,str) or Path(completed_id).name!=completed_id:
            raise ValueError('Invalid completed trial ID')
        completed=parent.parent/completed_id
        completed_directories.append(str(completed))
        completed_job=read(completed/'job.json')
        if completed_job.get('campaign_id')!=parent.name or completed_job.get('attempt_number')!=number:
            raise ValueError('Completed trial ownership/order mismatch')
        result=_attempt(completed,completed_job)
        if result['official_success'] is None or result['category'] not in ('verified_pass','verified_fail'):
            raise ValueError('Completed trial requires valid independent replay')
        if sha(completed/'job.json')!=trial_proof['job_sha256'] or sha(completed/'verification.json')!=trial_proof['verification_sha256']:
            raise ValueError('Verified trial changed after boundary stop')
        continuity=read(completed/'continuity.json')
        if not plan.get('initial_fingerprint') or plan['initial_fingerprint']!=continuity.get('initial_fingerprint'):
            raise ValueError('Resume requires original pinned verified reset fingerprint')
        required.extend([completed/'job.json',completed/'report.json',completed/'verification.json',completed/'continuity.json'])
        for name in ('context-end.json.gz','history.sqlite','history-manifest.json','provider-responses.jsonl','model-usage.jsonl'):
            required.append(completed/name)
        archive=read(archives[completed_id])
        if archive.get('scope')!='finalized-job' or archive.get('job_id')!=completed.name or not archive.get('remote_readback_verified') or not archive.get('restore_test_passed'):
            raise ValueError('Completed trial durability not verified')
        archived={entry['path']:entry['sha256'] for entry in archive.get('files',[])}
        if any(archived.get(name)!=sha(completed/name) for name in ('job.json','report.json','verification.json')):
            raise ValueError('Completed archive does not bind verified trial evidence')
        required.append(Path(archives[completed_id]).resolve())
        for name in ('timing.jsonl','token-counts.jsonl','context-estimates.jsonl','provider-request-events.jsonl'):
            if (completed/name).exists():required.append(completed/name)
    required.extend([parent/'model.json',parent/'source.zip'])
    if (parent/'context-estimates.jsonl').exists():required.append(parent/'context-estimates.jsonl')
    workspace=Path(cert['workspace']).resolve()
    if sha(workspace)!=cert['workspace_sha256'] or provenance.get('checkpoint_sha256')!=cert['workspace_sha256'] or provenance.get('code_image')!=plan['code_image']:
        raise ValueError('Full workspace checkpoint provenance mismatch')
    manifest=inspect_archive(workspace)
    if manifest.get('schema_version')!='workspace-checkpoint-v2':raise ValueError('Full regular-file checkpoint required')
    if count>1 and provenance.get('completed_attempt_id')!=ids[-1]:
        raise ValueError('Workspace checkpoint must follow latest completed trial')
    for name,digest in provenance['evidence_sha256'].items():
        if sha(completed/name)!=digest:raise ValueError('Captured completed-trial evidence changed')
    prefix_id=boundary.get('zero_input_prefix_job_id')
    if not prefix_id or Path(prefix_id).name!=prefix_id:raise ValueError('Explicit original zero-input prefix required')
    prefix=parent.parent/prefix_id;prefix_job=read(prefix/'job.json')
    if (prefix_job.get('campaign_id')!=parent.name or prefix_job.get('attempt_number')!=count+1
        or prefix_job.get('status')!='infrastructure_error' or job.get('current_job')!=prefix_id
        or state.get('pending_trial_job',state.get('pending_trial2'))!=prefix_id or proof.get('pending_child',{}).get('id')!=prefix_id):
        raise ValueError('Original next-trial prefix mismatch')
    if (prefix/'report.json').exists() or (prefix/'verification.json').exists():
        raise ValueError('Prefix has a result; not a zero-input boundary')
    for name in ('context-estimates.jsonl','timing.jsonl','provider-responses.jsonl','provider-request-events.jsonl','model-usage.jsonl','token-counts.jsonl','model-transcript.jsonl','actions.jsonl','submissions.jsonl'):
        p=prefix/name
        if p.exists() and p.stat().st_size:raise ValueError('Prefix contains model input, operations or actions; cannot resume boundary')
    allowed_children=set(ids)|{prefix_id}
    for candidate in parent.parent.glob('*/job.json'):
        child_record=read(candidate)
        if child_record.get('campaign_id')==parent.name and child_record.get('id') not in allowed_children:
            raise ValueError('Unaccounted trial exists outside certified boundary')
    required.append(prefix/'job.json')
    for p in required:
        if str(p.resolve()) not in cert['evidence_sha256']:raise ValueError('Certificate omits required source binding: '+str(p))
    checkpoint=Path(cert['model_checkpoint']).resolve()
    if sha(checkpoint)!=cert['model_checkpoint_sha256']:raise ValueError('Model checkpoint changed')
    _durability(certificate,cert)
    from pairing.openrouter_checkpoint import validate_against_receipts
    validate_against_receipts(checkpoint,parent,completed,boundary=boundary)
    _check_reserved_spend(read(plan['participant']['ledger_path']),cert.get('preserved_external_reservations',{}),
        completed_timeouts=boundary.get('completed_timeout_reservations',[]),completed_attempt_ids=ids)
    return {'parent':str(parent),'certificate':str(certificate),'certificate_sha256':sha(certificate),
            'boundary':boundary,'plan':plan,'record':job,'completed':str(completed),'completed_directories':completed_directories,
            'prefix':str(prefix),'prefix_record':prefix_job,'workspace':str(workspace),
            'model_checkpoint':str(checkpoint),'model_checkpoint_sha256':cert['model_checkpoint_sha256'],
            'preserved_external_reservations':cert.get('preserved_external_reservations',{}),
            'campaign_root':str(Path(cert['campaign_root']).resolve()),'source_scope':'no provider or simulator calls'}


def _cleanup_check(info):
    """Read-only poll of every original recorded owned sandbox, including prefix."""
    parent=Path(info['parent']);ids=set()
    for directory in [*(Path(p) for p in info['completed_directories']),Path(info['prefix'])]:
        for p in directory.rglob('modal-session.json'):
            session=read(p)
            if session.get('environment')!='FDR':raise ValueError('Unexpected original environment')
            ids.add(session['sandbox_id'])
    if not ids:raise ValueError('Missing original simulator identities')
    execution=info['plan']['execution']
    code='import modal,json; ids='+repr(sorted(ids))+'; print(json.dumps({i:modal.Sandbox.from_id(i).poll() for i in ids}))'
    result=subprocess.run([execution.get('modal_sdk_python',sys.executable),'-c',code],
        capture_output=True,text=True,timeout=60,check=True,
        env={**os.environ,'MODAL_PROFILE':execution['modal_profile'],'MODAL_ENVIRONMENT':execution['modal_environment']})
    polls=json.loads(result.stdout)
    if set(polls)!=ids or any(v is None for v in polls.values()):raise ValueError('Original simulator cleanup unconfirmed')
    return polls


def _check_reserved_spend(ledger, approved, *, completed_timeouts=(), completed_attempt_ids=()):
    """Preserve exact external and independently certified completed-timeout holds.

    This performs no ledger writes. Caller must first validate the checkpoint
    against original receipts, including the completed model-deadline proof.
    """
    if not isinstance(approved,dict):raise ValueError('External reservation review must be a mapping')
    timeout_holds={}
    required={'reservation_id','amount_usd','label','created_epoch','attempt_id','pending_call'}
    for row in completed_timeouts:
        if not isinstance(row,dict) or set(row)!=required:
            raise ValueError('Malformed completed timeout reservation')
        if (row['attempt_id'] not in completed_attempt_ids or type(row['pending_call']) is not int
            or row['pending_call']<=0 or row['label']!=f"google/gemini-4-argon:{row['attempt_id']}:call{row['pending_call']}"
            or not isinstance(row['reservation_id'],str) or not row['reservation_id']
            or type(row['amount_usd']) not in (int,float) or not math.isfinite(row['amount_usd']) or row['amount_usd']<=0
            or type(row['created_epoch']) not in (int,float) or not math.isfinite(row['created_epoch'])):
            raise ValueError('Timeout hold must identify a completed trial request exactly')
        identity=row['reservation_id']
        if identity in timeout_holds or identity in approved:
            raise ValueError('Timeout holds cannot be duplicated or classified as external')
        timeout_holds[identity]={k:row[k] for k in ('amount_usd','label','created_epoch')}
    held=ledger.get('reserved',{})
    for identity,receipt in held.items():
        if identity in timeout_holds:
            if receipt!=timeout_holds[identity]:raise ValueError('Completed timeout hold changed')
        else:
            if receipt.get('label','').startswith('google/gemini-4-argon:'):
                raise ValueError('Actor request reservation cannot be reviewed as an external hold')
            if identity not in approved or receipt!=approved[identity]:
                raise ValueError('Unsettled unreviewed provider charges prevent continuation')
    if set(approved)|set(timeout_holds)!=set(held):
        raise ValueError('Reviewed reservation state changed')


def resume(parent, certificate):
    info=validate(parent,certificate)
    with (Path(info['campaign_root'])/'provider-openrouter.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        info=validate(parent,certificate)
        polls=_cleanup_check(info)
        plan=info['plan'];parent=Path(info['parent'])
        ledger=read(plan['participant']['ledger_path'])
        _check_reserved_spend(ledger,info['preserved_external_reservations'],
            completed_timeouts=info['boundary'].get('completed_timeout_reservations',[]),
            completed_attempt_ids=info['boundary']['completed_attempt_ids'])
        maintenance=parent/'resume-history'
        maintenance.mkdir(exist_ok=False)  # one explicit continuation; never silently retry
        receipt=maintenance/'resume.json'
        for original in parent.iterdir():
            if original.is_file():
                shutil.copy2(original,maintenance/original.name)
        from pairing.openrouter_checkpoint import restore
        from pairing.context_history import INSTRUCTION
        from pairing.muse_model import credential
        model=restore(info['model_checkpoint'],checkpoint_sha256=info['model_checkpoint_sha256'],
            destination=maintenance/'restored-client',key=credential(),expected_participant=plan['participant'],
            expected_instructions=plan['prompt']+INSTRUCTION,expected_boundary=info['boundary'])
        _stop_check(parent,info['campaign_root'])
        record={'kind':'certified-between-trial-continuation','pid':os.getpid(),'started_epoch':time.time(),
                'certificate':info['certificate'],'certificate_sha256':info['certificate_sha256'],
                'boundary':info['boundary'],'original_owned_sandbox_polls':polls,
                'preserved_external_reservations':info['preserved_external_reservations'],
                'original_plan_preserved':True,'original_parent_and_trial_numbers_preserved':True,
                'no_completed_trial_rerun':True,'checkpoint_workspace_schema':'workspace-checkpoint-v2'}
        with receipt.open('x') as f:json.dump(record,f,indent=2)
        reports=[]
        for completed in info['completed_directories']:
            report=read(Path(completed)/'report.json')
            reports.append({key:report[key] for key in ('attempt_number','observed_success','termination_reason','control_steps','elapsed_seconds')})
        from pairing.adaptive_run import run
        from roboswe.adaptive_transport import episode_factory
        return run(plan,jobs_dir=parent.parent,episode_factory=episode_factory,resume_state={
            **info,'model':model,'reports':reports,'next_trial_number':info['boundary']['next_trial_number'],'receipt':str(receipt),
            'preserved_prefix':str(maintenance/'original-zero-input-prefix'),
            'stop_check':lambda:_stop_check(parent,info['campaign_root'])})
