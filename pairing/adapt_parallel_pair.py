"""One bounded two-provider Modal pair; trials within each provider stay serial.

Prepared replacement for the serial waiting supervisor. Never starts more than
one owned campaign child per provider, never retries, never schedules more tasks.
"""
import argparse
import fcntl
import json
import hashlib
import re
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from pairing.adapt_campaign import write
from pairing.adapt_followon import live_simulators
from pairing.adapt_modal_pair import classify,pid_live


def child_command(root,provider,python):
    if provider not in ('openrouter','anthropic'):raise ValueError('Unsupported provider')
    return [python,'-u','-m','pairing.adapt_campaign','--dataset',str(root/'toast-modal-dataset.json'),
        '--config',str(root/(provider+'-toast-modal.json')),'--state-dir',str(root/(provider+'-toast-modal')),'--max-series','1']


def child_result(root,provider,returncode):
    receipt=root/(provider+'-toast-modal')/'active.json'
    if not receipt.exists():return {'provider':provider,'returncode':returncode,'classification':'blocked_infrastructure','status':'missing_receipt'}
    value=json.loads(receipt.read_text());status=value.get('child_status',value.get('status'))
    result={'provider':provider,'pid':value.get('pid'),'returncode':returncode,'status':status,
        'classification':classify(status,value.get('error','')),'receipt':str(receipt),'completed':value.get('completed',[]),
        'context_policy':value.get('context_policy')}
    if result['classification']=='completed':
        if returncode or len(result['completed'])!=1:result['classification']='blocked_infrastructure'
        else:
            series=Path(result['completed'][0]);parent=json.loads((series/'job.json').read_text());report=json.loads((series/'report.json').read_text())
            if parent.get('status')!='completed_unscored' or len(report.get('trials',[]))!=5 or any(r.get('termination_reason')=='precondition_invalid' for r in report.get('trials',[])):
                result['classification']='blocked_infrastructure'
                result['validation_error']='Series not a complete valid five-trial result'
    return result


def validate_pair_configs(root):
    configs={p:json.loads((root/(p+'-toast-modal.json')).read_text()) for p in ('openrouter','anthropic')}
    left,right=configs.values()
    if left['execution']!=right['execution'] or left['limits']!=right['limits'] or left['trials']!=right['trials']:
        raise ValueError('Pair simulator, code image, budgets and trial counts must match')
    if left['execution'].get('backend')!='modal' or left['execution'].get('modal_environment')!='FDR':raise ValueError('Pair requires explicit Modal FDR')
    if left['trials']!=5:raise ValueError('Pair requires exactly five serial trials per provider')
    if left['participant'].get('provider')!='openrouter' or right['participant'].get('provider')!='anthropic':raise ValueError('Provider config mismatch')
    return configs


COUNT_TRANSPORT_ERROR = 'Anthropic token counting transport failed; no inference attempted'


def validate_predecessor_recovery(audit_path, previous):
    """Allow only an audited count-only transport incident before a fresh task.

    This never changes the failed predecessor or treats its incomplete trial as
    verified. An audit author must verify the repair separately before setting
    fix_verified; its exact bytes and the original stderr are bound in state.
    """
    audit_path, previous = Path(audit_path), Path(previous)
    raw = audit_path.read_bytes()
    audit = json.loads(raw)
    expected = {'kind': 'anthropic_count_transport_recovery_v1',
                'status': 'infrastructure_error', 'error': COUNT_TRANSPORT_ERROR,
                'action': 'allow_fresh_toast_only', 'fix_verified': True}
    if audit.get('fix_verified') is not True or any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError('Recovery audit does not authorize this exact repaired incident')
    parent = json.loads((previous / 'job.json').read_text())
    stderr = (previous / 'stderr.txt').read_bytes()
    if (parent.get('id') != audit.get('predecessor_job_id') or
        parent.get('id') != previous.name or
        parent.get('status') != 'infrastructure_error' or
        parent.get('provider') != 'anthropic' or
        not parent.get('finished_at') or parent.get('artifact_errors')):
        raise ValueError('Recovery predecessor identity, status, or finalization mismatch')
    if hashlib.sha256(stderr).hexdigest() != audit.get('stderr_sha256'):
        raise ValueError('Recovery stderr hash mismatch')
    if stderr.decode().rstrip().splitlines()[-1] != 'RuntimeError: ' + COUNT_TRANSPORT_ERROR:
        raise ValueError('Recovery is restricted to the exact count-only transport exception')
    child_id = audit.get('child_job_id', '')
    if not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-[a-f0-9]{10}', child_id) or parent.get('current_job') != child_id:
        raise ValueError('Recovery child does not match predecessor current job')
    child_dir = previous.parent / child_id
    child = json.loads((child_dir / 'job.json').read_text())
    if (child.get('id') != child_id or child.get('campaign_id') != parent['id'] or
        child.get('kind') != 'adaptive-direct-attempt' or child.get('provider') != 'anthropic' or
        child.get('status') != 'infrastructure_error' or not child.get('finished_at') or
        child.get('artifact_errors')):
        raise ValueError('Recovery requires a finalized incomplete Anthropic child')
    child_report = child_dir / 'report.json'
    if child_report.exists():
        report = json.loads(child_report.read_text())
        if report.get('verified_success') is not None or report.get('observed_success') is not None:
            raise ValueError('Recovery must not bypass a completed trial result')
    if (child_dir / 'verification.json').exists():
        raise ValueError('Recovery must not bypass an existing verification receipt')
    return {'audit_path': str(audit_path.resolve()), 'audit_sha256': hashlib.sha256(raw).hexdigest(),
            'predecessor_job_id': parent['id'], 'child_job_id': child_id,
            'stderr_sha256': audit['stderr_sha256'], 'classification': 'audited_count_transport_recovery',
            'scope': 'Fresh Toast task only; predecessor incomplete trial remains incomplete'}


def validate_fresh_toast(root):
    data = json.loads((root / 'toast-modal-dataset.json').read_text())
    tasks = data.get('tasks', [])
    if len(tasks) != 1 or tasks[0].get('task') != 'ToastBagel' or tasks[0].get('config', {}).get('task') != 'robocasa/ToastBagel':
        raise ValueError('Recovery may launch only the separately planned fresh ToastBagel task')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--predecessor-pid',required=True,type=int);parser.add_argument('--predecessor-job',required=True)
    parser.add_argument('--wait-hours',type=float,default=8)
    parser.add_argument('--predecessor-recovery',type=Path)
    parser.add_argument('--state-dir',type=Path)
    args=parser.parse_args(argv)
    root=Path(args.root).resolve();default_folder=root/'modal-toast-parallel-pair'
    folder=args.state_dir.resolve() if args.state_dir else default_folder
    if args.predecessor_recovery and (args.state_dir is None or folder==default_folder):
        raise ValueError('Recovery requires a new separate --state-dir; preserve original supervisor state')
    recovery = None
    if args.predecessor_recovery:
        validate_fresh_toast(root)
        recovery=validate_predecessor_recovery(args.predecessor_recovery,args.predecessor_job)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'owned.lock').open('a+') as owned:
        fcntl.flock(owned,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'active.json').exists():raise RuntimeError('Existing parallel pair receipt; no implicit relaunch')
        validate_pair_configs(root)
        state={'pid':os.getpid(),'status':'waiting_for_local_predecessor','started_epoch':time.time(),
            'predecessor_pid':args.predecessor_pid,'predecessor_job':args.predecessor_job,'children':[],'results':[],
            'task_pair_limit':1,'max_parallel_provider_series':2,'trials_per_provider':5,'trials_within_provider':'sequential',
            'compute_concurrency':'At most two independent FDR L4 simulator sandboxes and two local code execution lanes; no local simulator',
            'retry_policy':'none','environment':'FDR','modal_profile':'snorkelai'}
        if recovery:state['predecessor_recovery']=recovery
        write(folder/'active.json',state)
        children=[]
        try:
            deadline=time.monotonic()+args.wait_hours*3600
            while pid_live(args.predecessor_pid) or live_simulators():
                if time.monotonic()>deadline:raise TimeoutError('Bounded predecessor wait expired')
                if (folder/'STOP').exists():raise RuntimeError('Stopped by control before paid launch')
                time.sleep(10)
            previous=Path(args.predecessor_job);job=json.loads((previous/'job.json').read_text())
            if not job.get('finished_at'):raise RuntimeError('Predecessor not finalized')
            trace=(previous/'stderr.txt').read_text() if (previous/'stderr.txt').exists() else ''
            kind=classify(job.get('status'),trace);state['predecessor_status']=job.get('status')
            report=previous/'report.json'
            if report.exists() and any(x.get('termination_reason')=='precondition_invalid' for x in json.loads(report.read_text()).get('trials',[])):
                raise RuntimeError('Predecessor precondition invalid')
            if kind=='blocked_infrastructure':
                if not recovery:raise RuntimeError('Predecessor infrastructure/verifier/storage failure blocks launch')
                current_recovery=validate_predecessor_recovery(args.predecessor_recovery,previous)
                if current_recovery!=recovery:raise RuntimeError('Recovery audit changed after startup')
                kind='audited_count_transport_recovery'
            providers=['openrouter'] if kind=='provider_blocked' else ['openrouter','anthropic']
            if kind=='provider_blocked':state['results'].append({'provider':'anthropic','status':'skipped_provider_blocked','classification':'provider_blocked'})
            with (root/'global-campaign.lock').open('a+') as global_guard:
                fcntl.flock(global_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if live_simulators():raise RuntimeError('Local simulator present before parallel remote launch')
                if (folder/'STOP').exists():raise RuntimeError('Stopped by control before paid launch')
                validate_pair_configs(root)
                if recovery:
                    validate_fresh_toast(root)
                    if validate_predecessor_recovery(args.predecessor_recovery,previous)!=recovery:
                        raise RuntimeError('Recovery audit changed before launch')
                state['config_sha256']={p:hashlib.sha256((root/(p+'-toast-modal.json')).read_bytes()).hexdigest() for p in providers}
                # Refuse any previous lane receipt, including failed/completed ones.
                for provider in providers:
                    if (root/(provider+'-toast-modal')/'active.json').exists():raise RuntimeError('Provider lane already has receipt; refusing duplicate')
                key=os.environ.get('GEMINI_API_KEY')
                if not key:
                    result=subprocess.run(['/usr/bin/security','find-generic-password','-a','GEMINI_API_KEY','-w'],capture_output=True,timeout=20)
                    if result.returncode or not result.stdout.strip():raise RuntimeError('Google count-only key missing; output suppressed')
                    key=result.stdout.decode().strip()
                for provider in providers:
                    if (folder/'STOP').exists():raise RuntimeError('Stopped by control before provider launch')
                    if recovery and validate_predecessor_recovery(args.predecessor_recovery,previous)!=recovery:
                        raise RuntimeError('Recovery audit changed before provider launch')
                    lane=root/(provider+'-toast-modal');lane.mkdir(exist_ok=True)
                    log=(lane/'supervisor.log').open('ab')
                    env={**os.environ,'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
                    if provider=='openrouter':env['GEMINI_API_KEY']=key
                    try:process=subprocess.Popen(child_command(root,provider,sys.executable),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                    finally:log.close()
                    children.append((provider,process));state['children'].append({'provider':provider,'pid':process.pid,'log':str(lane/'supervisor.log')})
                    write(folder/'active.json',state)
                key=None
                state['status']='running_parallel_provider_series';write(folder/'active.json',state)
                remaining=list(children)
                while remaining:
                    for provider,process in list(remaining):
                        code=process.poll()
                        if code is None:continue
                        state['results'].append(child_result(root,provider,code));remaining.remove((provider,process))
                        write(folder/'active.json',state)
                    if remaining:time.sleep(10)
                state['status']='completed' if all(r['classification']=='completed' for r in state['results']) else 'partial_stopped'
        except BaseException as exc:
            state.update(status='stopped',error_type=type(exc).__name__,error=str(exc)[:2000]);(folder/'error.txt').write_text(traceback.format_exc())
            # Never abandon an owned child: request its normal finally cleanup.
            import signal
            for provider,process in children:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        state.setdefault('cleanup_pending',[]).append({'provider':provider,'pid':process.pid})
        finally:
            if state.get('cleanup_pending'):
                state['status']='cleanup_pending'
                state['supervisor_finished_epoch']=time.time()
                state['finished_epoch']=None
            else:state['finished_epoch']=time.time()
            write(folder/'active.json',state)


if __name__=='__main__':main()
