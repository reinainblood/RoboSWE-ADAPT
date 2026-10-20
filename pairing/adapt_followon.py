"""Owned single follow-on: wait for first series, Opus once, then scene admission.

No model retries, no overlapping simulator work, no success-based task filtering.
"""
import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from pairing.adapt_campaign import main as campaign_main,write


def safe_predecessor(job,trace=''):
    status=job.get('status')
    if status in ('completed_unscored','model_refusal'):return True
    if status=='infrastructure_error':
        return any(marker in trace for marker in ('OpenRouter HTTP ', 'OpenRouter transport failed;',
            'OpenRouter cost/usage receipt missing;', 'OpenRouter incomplete decision;',
            'OpenRouter error or model substitution;', 'campaign spend guard reached'))
    return False


def live_simulators():
    result=subprocess.run(['docker','ps','--filter','name=roboswe-simulator-','--format','{{.Names}}'],capture_output=True,text=True,timeout=30,check=True)
    return result.stdout.splitlines()


def admit(root,candidates,execution):
    from roboswe.adaptive_transport import episode_factory
    from roboswe.library.serial import serial_spec
    from pairing.overnight_suite import suite_worker
    admitted=[]
    for row in candidates:
        receipts=[]
        for number in (1,2):
            if live_simulators():raise RuntimeError('Unexpected running simulator before scene admission')
            folder=root/'scene-admission'/row['id']/f'reset-{number}'
            folder.mkdir(parents=True,exist_ok=False)
            spec=serial_spec(row['task'],timeout=300,max_steps=10)
            spec.update(execution_backend='docker',docker_simulator_image=execution['simulator_image'],simulator_startup_timeout_seconds=300)
            runtime=None;receipt={'config':row['config'],'model_calls':0};began=time.monotonic()
            try:
                runtime=episode_factory(row['config'],spec,folder,number,[],worker=suite_worker)
                receipt['fingerprint']=runtime.initial['fingerprint']
                initial=runtime.start_task(0)
                if not initial['precondition_valid']:raise RuntimeError('Invalid initial precondition')
                dimension=initial['observation']['action_schema']['shape'][0]
                for _ in range(5):runtime.bridge({'operation':'step','action':[0.0]*dimension})
                receipt.update(ok=True,control_steps=5)
                runtime.finish_task(code_errors=0)
            except Exception as exc:
                receipt.update(ok=False,error_type=type(exc).__name__,error=str(exc)[:2000]);raise
            finally:
                if runtime:
                    try:receipt['close']=runtime.close()
                    except Exception:
                        receipt['ok']=False;receipt['close_error']=traceback.format_exc()
                receipt['seconds']=time.monotonic()-began;write(folder/'receipt.json',receipt)
            if not receipt['ok']:raise RuntimeError('Admission close failed')
            receipts.append(receipt)
        if receipts[0]['fingerprint']!=receipts[1]['fingerprint']:
            write(root/'scene-admission'/row['id']/'result.json',{'status':'reset_mismatch','receipts':receipts})
            continue
        admitted.append({**row,'admission_status':'two-reset-five-step-docker-preflight','initial_fingerprint':receipts[0]['fingerprint']})
        write(root/'scene-admission'/row['id']/'result.json',{'status':'admitted','initial_fingerprint':receipts[0]['fingerprint'],'model_calls':0})
    write(root/'admitted-bread-dataset.json',{'tasks':admitted,'status':'admitted-not-launched','model_calls':0})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--predecessor-job',required=True)
    p.add_argument('--deadline-hours',type=float,default=8)
    args=p.parse_args(argv);root=Path(args.root).resolve();state_root=root/'followon';state_root.mkdir(exist_ok=True)
    with (state_root/'owned.lock').open('a+') as owned:
        fcntl.flock(owned,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (state_root/'active.json').exists():raise RuntimeError('Existing follow-on receipt; refuse implicit restart')
        state={'pid':os.getpid(),'status':'waiting_for_predecessor','predecessor_job':args.predecessor_job,'started_epoch':time.time(),'model_retry_policy':'none'}
        write(state_root/'active.json',state)
        try:
            deadline=time.monotonic()+args.deadline_hours*3600
            with (root/'openrouter-pilot'/'campaign.lock').open('a+') as preceding:
                while True:
                    if (state_root/'STOP').exists():raise RuntimeError('Stopped by control before launch')
                    if time.monotonic()>deadline:raise TimeoutError('Bounded predecessor wait expired')
                    try:fcntl.flock(preceding,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:time.sleep(10)
                previous=Path(args.predecessor_job);job=json.loads((previous/'job.json').read_text())
                trace=(previous/'stderr.txt').read_text() if (previous/'stderr.txt').exists() else ''
                state['predecessor_status']=job.get('status')
                report_path=previous/'report.json'
                if report_path.exists() and any(r.get('termination_reason')=='precondition_invalid' for r in json.loads(report_path.read_text()).get('trials',[])):
                    raise RuntimeError('Predecessor invalid task precondition blocks follow-on')
                if not safe_predecessor(job,trace):raise RuntimeError('Predecessor infrastructure/verifier/precondition status blocks follow-on')
                with (root/'global-campaign.lock').open('a+') as global_lock:
                    fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if live_simulators():raise RuntimeError('Unexpected live simulator blocks follow-on')
                    state['status']='running_anthropic';write(state_root/'active.json',state)
                    campaign_main(['--dataset',str(root/'pilot-dataset.json'),'--config',str(root/'anthropic-run.json'),
                        '--state-dir',str(root/'anthropic-pilot'),'--max-series','1'])
                    opus=json.loads((root/'anthropic-pilot'/'active.json').read_text())
                    state['anthropic_status']=opus['status']
                    if opus['status']!='completed':raise RuntimeError('Anthropic lane stopped; no admission runs after failure')
                    state['status']='qualifying_bread_scenes';write(state_root/'active.json',state)
                    candidates=json.loads((root/'next-candidate-scenes.json').read_text())['tasks']
                    execution=json.loads((root/'anthropic-run.json').read_text())['execution']
                    admit(root,candidates,execution)
                    state['status']='completed'
        except BaseException as exc:
            state.update(status='stopped',error_type=type(exc).__name__,error=str(exc)[:2000])
            (state_root/'error.txt').write_text(traceback.format_exc())
        finally:
            state['finished_epoch']=time.time();write(state_root/'active.json',state)


if __name__=='__main__':main()
