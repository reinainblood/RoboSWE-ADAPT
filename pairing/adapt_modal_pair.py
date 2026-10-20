"""One staged Modal task pair; wait for owned local predecessor, never retry."""
import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from pairing.adapt_campaign import main as campaign_main,write
from pairing.adapt_followon import live_simulators


def classify(status,error=''):
    if status in ('completed_unscored','completed'):return 'completed'
    if status=='model_refusal':return 'provider_blocked'
    if status in ('artifact_durability_error','verification_invalid','invalid_precondition'):return 'blocked_infrastructure'
    if any(marker in error for marker in ('Independent replay invalid','Reset differs','ArtifactDurabilityError','readback mismatch','Owned sandbox cleanup failed','No space left on device')):return 'blocked_infrastructure'
    if any(marker in error for marker in ('OpenRouter HTTP ', 'Anthropic HTTP ',
            'OpenRouter transport failed;', 'campaign spend guard reached',
            'OpenRouter cost/usage receipt missing;', 'Anthropic explicitly refused')):
        return 'provider_blocked'
    return 'blocked_infrastructure'


def pid_live(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True,timeout=10)
    return result.returncode==0 and bool(result.stdout.strip()) and not result.stdout.strip().startswith('Z')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--predecessor-pid',required=True,type=int);p.add_argument('--predecessor-job',required=True)
    p.add_argument('--wait-hours',type=float,default=8);args=p.parse_args(argv)
    root=Path(args.root).resolve();folder=root/'modal-toast-pair';folder.mkdir(exist_ok=True)
    with (folder/'owned.lock').open('a+') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'active.json').exists():raise RuntimeError('Existing task-pair receipt; no implicit relaunch')
        state={'pid':os.getpid(),'status':'waiting_for_local_predecessor','predecessor_pid':args.predecessor_pid,
            'predecessor_job':args.predecessor_job,'started_epoch':time.time(),'results':[],'task_pair_limit':1,
            'environment':'FDR','modal_profile':'snorkelai','retry_policy':'none'}
        write(folder/'active.json',state);previous_key=os.environ.get('GEMINI_API_KEY')
        try:
            deadline=time.monotonic()+args.wait_hours*3600
            while pid_live(args.predecessor_pid) or live_simulators():
                if time.monotonic()>deadline:raise TimeoutError('Bounded predecessor wait expired')
                if (folder/'STOP').exists():raise RuntimeError('Stopped by control before provider launch')
                time.sleep(10)
            predecessor=Path(args.predecessor_job);job=json.loads((predecessor/'job.json').read_text())
            if not job.get('finished_at'):raise RuntimeError('Predecessor not finalized')
            trace=(predecessor/'stderr.txt').read_text() if (predecessor/'stderr.txt').exists() else ''
            kind=classify(job.get('status'),trace);state['predecessor_status']=job.get('status')
            report=predecessor/'report.json'
            if report.exists() and any(r.get('termination_reason')=='precondition_invalid' for r in json.loads(report.read_text()).get('trials',[])):
                raise RuntimeError('Predecessor invalid initial state blocks follow-on')
            if kind=='blocked_infrastructure':raise RuntimeError('Predecessor infrastructure/verifier/storage failure requires repair')
            native_blocked=kind=='provider_blocked'
            with (root/'global-campaign.lock').open('a+') as guard:
                fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if live_simulators():raise RuntimeError('Local simulator appeared before remote pair')
                if (folder/'STOP').exists():raise RuntimeError('Stopped by control before provider launch')
                # Native Google is used ONLY for countTokens. Never write or print key.
                if not previous_key:
                    key=subprocess.run(['/usr/bin/security','find-generic-password','-a','GEMINI_API_KEY','-w'],capture_output=True,timeout=20)
                    if key.returncode or not key.stdout.strip():raise RuntimeError('Google count-only credential unavailable; output suppressed')
                    os.environ['GEMINI_API_KEY']=key.stdout.decode().strip()
                for provider in ('openrouter','anthropic'):
                    if provider=='anthropic' and native_blocked:
                        state['results'].append({'provider':provider,'status':'skipped_provider_blocked','reason':'Native predecessor provider refused or funding/transport blocked; no retry'})
                        continue
                    if (folder/'STOP').exists():raise RuntimeError('Stopped by control at series boundary')
                    state.update(status='running_'+provider,current_provider=provider);write(folder/'active.json',state)
                    lane=root/(provider+'-toast-modal')
                    error=''
                    try:
                        campaign_main(['--dataset',str(root/'toast-modal-dataset.json'),'--config',str(root/(provider+'-toast-modal.json')),
                            '--state-dir',str(lane),'--max-series','1'])
                    except Exception as exc:error=str(exc)
                    if not (lane/'active.json').exists():raise RuntimeError('Provider lane failed before owned receipt: '+error)
                    result=json.loads((lane/'active.json').read_text());status=result.get('child_status',result['status'])
                    classification=classify(status,error or result.get('error',''))
                    state['results'].append({'provider':provider,'status':status,'classification':classification,'receipt':str(lane/'active.json'),
                        'context_policy':result.get('context_policy'),'completed':result.get('completed',[])})
                    write(folder/'active.json',state)
                    if classification=='blocked_infrastructure':raise RuntimeError('Task-pair infrastructure/verifier/storage error; no later lane launched')
                state['status']='completed' if all(r.get('classification')=='completed' for r in state['results']) else 'partial_provider_blocked'
        except BaseException as exc:
            state.update(status='stopped',error_type=type(exc).__name__,error=str(exc)[:2000])
            (folder/'error.txt').write_text(traceback.format_exc())
        finally:
            if previous_key is None:os.environ.pop('GEMINI_API_KEY',None)
            else:os.environ['GEMINI_API_KEY']=previous_key
            state['finished_epoch']=time.time();write(folder/'active.json',state)


if __name__=='__main__':main()
