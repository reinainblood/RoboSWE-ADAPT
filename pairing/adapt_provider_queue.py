"""At most two fresh tasks for one pinned provider after an audited handoff.

The legacy Toast pair retains global-campaign.lock. This supervisor deliberately
uses provider locks only after a root-reviewed transition receipt binds both
legacy lanes and confirms the old future-pair waiter is stopped. Never retries a
series; one child per provider, at most two provider series across this transition.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import math
from datetime import datetime
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
from zipfile import ZipFile

from pairing.adapt_campaign import write
from pairing.adapt_modal_pair import pid_live
from pairing.adapt_next_pairs import (PROVIDERS, PINS, validate_plan, validate_series,
                                     read, digest, cleanup, child_command)


def validate_transition(path, plan_path, provider, predecessor_pid, predecessor_state):
    audit=read(path)
    if (audit.get('kind')!='provider_queue_transition_v1' or
        audit.get('lock_policy')!='provider-locks-after-audited-legacy-transition' or
        audit.get('max_parallel_provider_series')!=2 or
        audit.get('plan_sha256')!=digest(plan_path) or
        set(audit.get('provider_predecessors',{}))!=set(PROVIDERS)):
        raise ValueError('Explicit bounded legacy/provider-lock transition required')
    retired=audit.get('retired_waiter',{})
    if type(retired.get('pid')) is not int or retired['pid']<=0 or pid_live(retired['pid']):
        raise ValueError('Old future-pair waiter must be stopped')
    state=read(retired['state'])
    if state.get('pid')!=retired['pid'] or not state.get('finished_epoch') or state.get('cleanup_pending') or state.get('status') not in ('stopped','stopped_by_control') or state.get('children'):
        raise ValueError('Retired waiter needs terminal cleanup receipt')
    expected=audit['provider_predecessors'][provider]
    if expected.get('pid')!=predecessor_pid or Path(expected.get('state','')).resolve()!=Path(predecessor_state).resolve():
        raise ValueError('Provider predecessor is not the audited legacy lane')
    for p,binding in audit['provider_predecessors'].items():
        prior=read(binding['state'])
        if prior.get('pid')!=binding.get('pid') or prior.get('provider')!=p:
            raise ValueError('Both legacy provider lane identities must match')
    return {'path':str(Path(path).resolve()),'sha256':digest(path),'lock_policy':audit['lock_policy']}


def terminal_lane(path, pid, provider):
    state=read(path)
    if (state.get('pid')!=pid or pid_live(pid) or not state.get('finished_epoch') or
        state.get('cleanup_pending') or state.get('provider')!=provider or
        state.get('model')!=PINS[provider]['model']):
        raise ValueError('Provider lane must be owned, terminal, and model-pinned')
    return state


def series_cleanup(series,jobs,execution):
    """Poll only sandbox IDs recorded by this finalized series, never stop others."""
    parent=read(Path(series)/'job.json'); directories=[Path(series)]
    for path in Path(jobs).glob('*/job.json'):
        if read(path).get('campaign_id')==parent['id']:directories.append(path.parent)
    ids=set()
    for directory in directories:
        for path in directory.rglob('modal-session.json'):
            session=read(path)
            if session.get('environment')!='FDR':raise ValueError('Unexpected cleanup environment')
            if read(path.parent/'remote-artifacts.json').get('persisted_readback_verified') is not True:
                raise ValueError('Owned simulator artifact readback not verified')
            ids.add(session['sandbox_id'])
    if not ids:raise ValueError('No owned simulator receipts for completed series')
    code='import modal,json; ids='+repr(sorted(ids))+'; print(json.dumps({i:modal.Sandbox.from_id(i).poll() for i in ids}))'
    env={**os.environ,'MODAL_PROFILE':'snorkelai','MODAL_ENVIRONMENT':'FDR'}
    result=subprocess.run([execution.get('modal_sdk_python',sys.executable),'-c',code],capture_output=True,text=True,timeout=60,check=True,env=env)
    polls=json.loads(result.stdout)
    if set(polls)!=ids or any(v is None for v in polls.values()):raise ValueError('Owned simulator cleanup incomplete')
    return polls


def completed_lane(state,jobs,provider,execution,expected_task=None):
    if state.get('status')!='completed' or len(state.get('completed',[]))!=1:
        raise ValueError('Lane is not one completed series')
    series=Path(state['completed'][0]);validate_series(series,jobs)
    parent=read(series/'job.json');plan=read(series/'plan.json')
    if parent.get('provider')!=provider or (expected_task and parent.get('task')!=expected_task):
        raise ValueError('Completed series provider/task mismatch')
    # Series plan persists participant and execution settings; never infer from a label.
    from roboswe.adaptive import build_plan
    expected=build_plan(state['dataset'],state['config'])['tasks'][0]
    if any(plan.get(k)!=expected.get(k) for k in ('participant','execution','limits','trials','context_policy','cohort')):
        raise ValueError('Completed model/context/budget/task metadata mismatch')
    return {'classification':'completed','series':str(series),'cleanup':series_cleanup(series,jobs,execution)}


def reviewed_refusal(review_path,state_path,pid,provider,jobs):
    """Only an evidence-bound task refusal permits a different, fresh task."""
    from roboswe.adaptive_results import validated_failure_review
    if not review_path:raise ValueError('Stopped lane requires a strict task-refusal review')
    state=terminal_lane(state_path,pid,provider)
    review_path=Path(review_path).resolve();review=read(review_path)
    child_dir=Path(jobs).resolve()/review.get('job_id','INVALID')
    if review_path!=child_dir/'failure-review.json':
        raise ValueError('Failure review must be attached to the exact child job')
    child=read(child_dir/'job.json');series=Path(jobs).resolve()/child.get('campaign_id','INVALID')
    parent=read(series/'job.json');plan=read(series/'plan.json');config=read(state['config'])
    from roboswe.adaptive import build_plan
    normalized=build_plan(state['dataset'],state['config'])['tasks'][0]
    if (child.get('kind')!='adaptive-direct-attempt' or parent.get('kind')!='adaptive-direct-series' or
        child.get('provider')!=provider or parent.get('provider')!=provider or
        parent.get('current_job')!=child.get('id') or not parent.get('finished_at') or
        not child.get('finished_at') or child.get('artifact_errors') or parent.get('artifact_errors') or
        plan.get('cohort',{}).get('id')!=state.get('current_task') or
        plan.get('participant')!=normalized.get('participant') or
        plan.get('execution')!=normalized.get('execution')):
        raise ValueError('Reviewed failure is not the exact terminal provider lane')
    raw=read_last_response(child_dir)
    check_refusal_cost(raw,provider)
    classification=validated_failure_review(child_dir,child)
    if not classification or classification.get('classification')!='model_refusal':
        raise ValueError('Only a validated task-specific refusal can continue')
    if state.get('error') and state.get('error')!='OpenRouter incomplete decision; no operation executed':
        if state.get('error_type')!='ModelRefusal':
            raise ValueError('Funding/auth/transport/unknown lane error stops this provider')
    if (child_dir/'verification.json').exists():raise ValueError('Cannot replace a completed verification with a refusal')
    return {'classification':'reviewed_task_refusal','series':str(series),
            'review_sha256':digest(review_path),'review_path':str(review_path),
            'cleanup':series_cleanup(series,jobs,config['execution']),
            'scope':'Fresh different task only; original incomplete trial retained'}


def read_last_response(directory):
    data=(Path(directory)/'provider-responses.jsonl').read_text()
    if not data.endswith('\n'):raise ValueError('Partial provider response file')
    rows=[json.loads(line) for line in data.splitlines() if line.strip()]
    if not rows:raise ValueError('Missing provider response')
    return rows[-1]


def check_refusal_cost(row,provider):
    response=row.get('response',{})
    cost=response.get('usage',{}).get('cost')
    if (provider!='openrouter' or response.get('model')!=PINS[provider]['model'] or response.get('error') or
        type(cost) not in (int,float) or not math.isfinite(cost) or cost<0):
        raise ValueError('Refusal lacks pinned model and settled provider cost evidence')


def future_refusal_review(state,jobs,provider):
    """Create only an additive bound receipt for a new typed OpenRouter refusal."""
    if provider!='openrouter' or state.get('status')!='model_refusal':
        raise ValueError('Only a newly typed recognized refusal permits automatic review')
    task=read(state['dataset'])['tasks'][0]['task'];candidates=[]
    for path in Path(jobs).glob('*/job.json'):
        job=read(path)
        if job.get('kind')=='adaptive-direct-series' and job.get('provider')==provider and job.get('task')==task:
            if datetime.fromisoformat(job['created_at']).timestamp()>=state['started_epoch']-2:
                candidates.append((path.parent,job))
    if len(candidates)!=1:raise ValueError('Refusal series identity is not unique')
    series,parent=candidates[0];directory=Path(jobs)/parent['current_job'];child=read(directory/'job.json')
    if child.get('status')!='model_refusal' or not child.get('finished_at') or not parent.get('finished_at'):
        raise ValueError('Auto review requires finalized typed refusal')
    row=read_last_response(directory);check_refusal_cost(row,provider)
    choices=row['response'].get('choices',[])
    if len(choices)!=1:raise ValueError('Refusal requires one provider choice')
    choice=choices[0]
    if not (choice.get('finish_reason')=='content_filter' or choice.get('native_finish_reason')=='SAFETY' or (choice.get('message') or {}).get('refusal')):
        raise ValueError('Raw response does not establish task refusal')
    path=directory/'failure-review.json'
    receipt={'schema_version':1,'job_id':child['id'],'job_sha256':digest(directory/'job.json'),
             'response_file_sha256':digest(directory/'provider-responses.jsonl'),'response_call':row['call'],
             'choice_index':0,'finish_reason':choice.get('finish_reason'),'native_finish_reason':choice.get('native_finish_reason'),
             'classification':'model_refusal'}
    if path.exists():
        if read(path)!=receipt:raise ValueError('Existing review differs; never overwrite')
    else:
        with path.open('x') as stream:json.dump(receipt,stream,indent=2)
    from roboswe.adaptive_results import validated_failure_review
    validated_failure_review(directory,child)
    return path


RECOVERY_PARENT='20261020T122023Z-6cbecc8d70'
RECOVERY_CHILD='20261020T122023Z-c47fe278d1'
RECOVERY_ERROR='Unsupported message role or fields; no approximate fallback'
RECOVERY_FILES={
    RECOVERY_PARENT:('job.json','plan.json','report.json','stderr.txt'),
    RECOVERY_CHILD:('job.json','stderr.txt','provider-responses.jsonl','actions.jsonl','submissions.jsonl','initial-snapshot.zip','final-snapshot.zip'),
}


def validate_infrastructure_recovery(path,state_path,pid,provider,jobs):
    """One named zero-action harness incident; no general infrastructure retry."""
    review=read(path);jobs=Path(jobs).resolve();state=terminal_lane(state_path,pid,provider)
    expected={'kind':'openrouter_null_refusal_counter_recovery_v1','action':'replace_zero_action_BeverageSorting_once',
              'fix_verified':True,'predecessor_pid':28686,'parent_job_id':RECOVERY_PARENT,'child_job_id':RECOVERY_CHILD}
    if provider!='openrouter' or pid!=28686 or any(review.get(k)!=v for k,v in expected.items()):
        raise ValueError('Recovery is restricted to the named zero-action counter incident')
    if (review.get('fix_verified') is not True or state.get('status')!='stopped_error' or state.get('completed') or state.get('results') or
        state.get('error')!=RECOVERY_ERROR or state.get('error_type')!='ValueError'):
        raise ValueError('Counter repair or exact infrastructure error not established')
    expected_paths={str(Path(job)/name) for job,names in RECOVERY_FILES.items() for name in names}
    if set(review.get('evidence_sha256',{}))!=expected_paths:raise ValueError('Recovery evidence inventory differs')
    for relative,sha in review['evidence_sha256'].items():
        if digest(jobs/relative)!=sha:raise ValueError('Recovery original evidence hash mismatch')
    parent_dir=jobs/RECOVERY_PARENT;child_dir=jobs/RECOVERY_CHILD
    parent=read(parent_dir/'job.json');child=read(child_dir/'job.json');plan=read(parent_dir/'plan.json')
    for job in (parent,child):
        if job.get('status')!='infrastructure_error' or not job.get('finished_at') or job.get('artifact_errors') or job.get('provider')!='openrouter' or job.get('task')!='BeverageSorting':
            raise ValueError('Only clean finalized infrastructure receipts qualify')
    if (parent.get('id')!=RECOVERY_PARENT or parent.get('current_job')!=RECOVERY_CHILD or parent.get('current_attempt')!=1 or
        child.get('id')!=RECOVERY_CHILD or child.get('campaign_id')!=RECOVERY_PARENT or child.get('attempt_number')!=1 or
        parent.get('kind')!='adaptive-direct-series' or child.get('kind')!='adaptive-direct-attempt'):
        raise ValueError('Recovery parent/child identity mismatch')
    from roboswe.adaptive import build_plan
    original=build_plan(state['dataset'],state['config'])['tasks'][0]
    if any(plan.get(k)!=original.get(k) for k in ('participant','execution','limits','trials','context_policy','cohort')):
        raise ValueError('Recovery provider/context/budget/scene settings differ')
    for directory in (parent_dir,child_dir):
        if (directory/'stderr.txt').read_text().rstrip().splitlines()[-1]!='ValueError: '+RECOVERY_ERROR:
            raise ValueError('Recovery does not match exact counter traceback')
        if (directory/'verification.json').exists():raise ValueError('Verified trials cannot be recovered')
    for filename in ('unsettled-spend.jsonl','provider-unsettled-responses.jsonl'):
        p=child_dir/filename
        if p.exists() and p.stat().st_size:raise ValueError('Unsettled provider failure blocks recovery')
    if read(parent_dir/'report.json').get('trials')!=[] or (child_dir/'report.json').exists() or (child_dir/'actions.jsonl').read_bytes():
        raise ValueError('Recovery requires zero completed trials and zero actions')
    siblings=[read(p) for p in jobs.glob('*/job.json') if read(p).get('campaign_id')==RECOVERY_PARENT]
    if len(siblings)!=1:raise ValueError('Recovery requires exactly one attempted child')
    rows=(child_dir/'provider-responses.jsonl').read_text().splitlines()
    if len(rows)!=1:raise ValueError('Recovery requires exactly one provider response')
    row=read_last_response(child_dir);check_refusal_cost(row,provider)
    choices=row['response'].get('choices',[])
    if len(choices)!=1 or choices[0].get('finish_reason')!='stop' or choices[0].get('native_finish_reason')!='STOP' or 'refusal' not in choices[0].get('message',{}) or choices[0]['message']['refusal'] is not None:
        raise ValueError('Recovery response must be normal stop with explicit null refusal')
    snapshots=[]
    for filename in ('initial-snapshot.zip','final-snapshot.zip'):
        with ZipFile(child_dir/filename) as archive:snapshots.append(json.loads(archive.read('manifest.json')))
    if any(x.get('steps')!=0 for x in snapshots) or not snapshots[0].get('current_fingerprint') or snapshots[0]['current_fingerprint']!=snapshots[1].get('current_fingerprint'):
        raise ValueError('Recovery requires unchanged zero-step physical state')
    fixes=review.get('fix_files_sha256',{})
    if not fixes or not set(fixes)<={'pairing/openrouter_google_count.py','pairing/openrouter_model.py'}:
        raise ValueError('Counter fix source inventory invalid')
    repo=Path(__file__).resolve().parents[1]
    for relative,sha in fixes.items():
        if digest(repo/relative)!=sha:raise ValueError('Verified counter fix source changed')
    test_path=Path(review['test_receipt'])
    if digest(test_path)!=review.get('test_receipt_sha256'):raise ValueError('Fix test receipt hash mismatch')
    test=read(test_path)
    if test.get('status')!='passed' or test.get('returncode')!=0 or type(test.get('tests_passed')) is not int or test['tests_passed']<1:
        raise ValueError('Counter fix test receipt must record passing tests')
    return {'classification':'audited_zero_action_counter_recovery','ignored_parent_id':RECOVERY_PARENT,
            'replacement_task':'BeverageSorting','review_path':str(Path(path).resolve()),'review_sha256':digest(path),
            'cleanup':series_cleanup(parent_dir,jobs,plan['execution']),
            'scope':'Exactly one fresh-context five-trial replacement; original prefix preserved and excluded separately'}


def ensure_fresh_task(jobs,provider,task,*,ignored_parent_id=None):
    if ignored_parent_id is not None and (ignored_parent_id!=RECOVERY_PARENT or provider!='openrouter' or task!='BeverageSorting'):
        raise ValueError('Only the reviewed BeverageSorting incident may be ignored')
    for path in Path(jobs).glob('*/job.json'):
        job=read(path)
        if ignored_parent_id is not None and ignored_parent_id!=RECOVERY_PARENT:
            raise ValueError('Only the reviewed incident may be ignored')
        if path.parent.name==ignored_parent_id and job.get('id')==ignored_parent_id:
            continue
        if job.get('kind')=='adaptive-direct-series' and job.get('provider')==provider and job.get('task')==task:
            plan=read(path.parent/'plan.json')
            if plan.get('participant',{}).get('model')==PINS[provider]['model']:
                raise ValueError('Existing series for this provider/model/task; no duplicate or retry')


def stop_requested(folder,root,lane=None):
    return any((p/'STOP').exists() for p in (folder,root,*([lane] if lane else [])))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('plan','state-dir','campaign-root','predecessor-state','transition-review'):
        parser.add_argument('--'+field,required=True)
    parser.add_argument('--provider',choices=PROVIDERS,required=True)
    parser.add_argument('--predecessor-pid',type=int,required=True)
    parser.add_argument('--failure-review')
    parser.add_argument('--infrastructure-recovery')
    parser.add_argument('--jobs-dir',default='jobs')
    parser.add_argument('--wait-hours',type=float,default=12)
    args=parser.parse_args(argv)
    if args.predecessor_pid<=0 or not 0<args.wait_hours<=48:raise ValueError('Bounded wait and positive PID required')
    root,folder,jobs=map(lambda p:Path(p).resolve(),(args.campaign_root,args.state_dir,args.jobs_dir));folder.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        def terminate_owned(signum,frame):
            raise KeyboardInterrupt('Supervisor termination requested')
        previous_handler=signal.signal(signal.SIGTERM,terminate_owned)
        stack.callback(signal.signal,signal.SIGTERM,previous_handler)
        for path in (folder/'owned.lock',root/('provider-'+args.provider+'.lock')):
            lock=stack.enter_context(path.open('a+'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'active.json').exists():raise ValueError('Existing provider queue receipt; no restart/retry')
        pairs,hashes=validate_plan(args.plan,root)
        transition=validate_transition(args.transition_review,args.plan,args.provider,args.predecessor_pid,args.predecessor_state)
        hashes[transition['path']]=transition['sha256']
        state={'pid':os.getpid(),'provider':args.provider,'status':'waiting_for_predecessor','started_epoch':time.time(),'input_sha256':hashes,'transition':transition,'predecessor_pid':args.predecessor_pid,'predecessor_state':args.predecessor_state,'children':[],'results':[],'max_fresh_tasks':len(pairs),'max_active_series_this_provider':1,'max_active_series_all_providers':2,'retry_policy':'none; reviewed task refusals may proceed to a distinct fresh task only'}
        children=[];write(folder/'active.json',state)
        try:
            deadline=time.monotonic()+args.wait_hours*3600
            while pid_live(args.predecessor_pid):
                if stop_requested(folder,root):raise RuntimeError('STOP before launch')
                if time.monotonic()>deadline:raise TimeoutError('Predecessor wait exceeded')
                time.sleep(5)
            previous=terminal_lane(args.predecessor_state,args.predecessor_pid,args.provider)
            execution=read(pairs[0]['configs'][args.provider])['execution']
            recovery=None
            if args.infrastructure_recovery:
                if args.failure_review:raise ValueError('Recovery and refusal review are mutually exclusive')
                recovery=validate_infrastructure_recovery(args.infrastructure_recovery,args.predecessor_state,args.predecessor_pid,args.provider,jobs)
                if read(pairs[0]['dataset'])['tasks'][0]['task']!='BeverageSorting':
                    raise ValueError('Recovery must begin with its one replacement BeverageSorting series')
                state['predecessor_result']=recovery
                hashes[recovery['review_path']]=recovery['review_sha256']
            elif previous.get('status')=='completed':state['predecessor_result']=completed_lane(previous,jobs,args.provider,execution)
            else:state['predecessor_result']=reviewed_refusal(args.failure_review,args.predecessor_state,args.predecessor_pid,args.provider,jobs)
            previous_tasks=read(previous['dataset']).get('tasks',[])
            used={t['task'] for t in previous_tasks}
            for pair in pairs:
                task=read(pair['dataset'])['tasks'][0]['task']
                replacement=bool(recovery and task==recovery['replacement_task'])
                if task in used and not replacement:raise ValueError('No failed/completed task may be retried in provider queue')
                used.add(task)
                ignored=recovery['ignored_parent_id'] if replacement else None
                if replacement:
                    validate_infrastructure_recovery(args.infrastructure_recovery,args.predecessor_state,args.predecessor_pid,args.provider,jobs)
                    recovery=None  # The sole exception cannot authorize a second replacement.
                ensure_fresh_task(jobs,args.provider,task,ignored_parent_id=ignored)
                if any(digest(p)!=sha for p,sha in hashes.items()):raise ValueError('Frozen queue inputs changed')
                if stop_requested(folder,root):raise RuntimeError('STOP before fresh task')
                lane=folder/pair['id']
                lane.mkdir(exist_ok=False)
                env={**os.environ,'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
                if args.provider=='openrouter' and not env.get('GEMINI_API_KEY'):
                    key=subprocess.run(['/usr/bin/security','find-generic-password','-a','GEMINI_API_KEY','-w'],capture_output=True,timeout=20)
                    if key.returncode or not key.stdout.strip():raise RuntimeError('Count credential unavailable; output suppressed')
                    env['GEMINI_API_KEY']=key.stdout.decode().strip()
                if stop_requested(folder,root,lane):raise RuntimeError('STOP immediately before paid child launch')
                with (lane/'supervisor.log').open('ab') as log:
                    process=subprocess.Popen(child_command(pair,args.provider,lane,jobs),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                children.append((args.provider,process));state['children'].append({'task':task,'pid':process.pid,'lane':str(lane)})
                state.update(status='running',current_task=task);write(folder/'active.json',state)
                while process.poll() is None:
                    if stop_requested(folder,root,lane):raise RuntimeError('STOP during owned child')
                    time.sleep(5)
                current=terminal_lane(lane/'active.json',process.pid,args.provider)
                if current.get('status')=='completed':
                    if process.returncode:raise ValueError('Completed child has nonzero exit code')
                    result=completed_lane(current,jobs,args.provider,read(pair['configs'][args.provider])['execution'],task)
                else:
                    review=future_refusal_review(current,jobs,args.provider)
                    state.setdefault('created_failure_reviews',[]).append({'path':str(review),'sha256':digest(review)})
                    write(folder/'active.json',state)
                    result=reviewed_refusal(review,lane/'active.json',process.pid,args.provider,jobs)
                state['results'].append({'task':task,**result});write(folder/'active.json',state)
            state['status']='completed'
        except BaseException as exc:state.update(status='stopped',error_type=type(exc).__name__,error=str(exc)[:1200])
        finally:
            pending=cleanup(children)
            if pending:state.update(status='cleanup_pending',cleanup_pending=pending,finished_epoch=None)
            else:state['finished_epoch']=time.time()
            write(folder/'active.json',state)
    return state


if __name__=='__main__':main()
