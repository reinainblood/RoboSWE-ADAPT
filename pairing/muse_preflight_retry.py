"""One additional exact-target reset admission after a mismatched preflight pair."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from pairing import cohort50 as c
from pairing.muse_cohort import ROOT, task_row, spec_for
from pairing.modal_episode import RemoteEpisode


def retry(task):
    folder=ROOT/'preflight'/task
    original=json.loads((folder/'receipt.json').read_text())
    if original['same_reset'] and original['passes_execution_margin']:
        raise ValueError('Preflight already passes reset and throughput checks')
    before=folder/'receipt-before-exact-retry.json'
    if before.exists():raise ValueError('Additional reset already attempted; inspect evidence')
    c.emit(before,original)
    directory=folder/'3';directory.mkdir(exist_ok=False)
    eligible=[i for i,(fp,duration) in enumerate(zip(original['fingerprints'],original['50_steps_seconds']))
              if fp==original['fingerprints'][0] and duration<75]
    prior_index=eligible[-1] if eligible else 0
    runtime=None;started=time.time();receipt={'task':task,'expected_fingerprint':original['fingerprints'][0],'policy':'One additional independent reset, exact original-first target, no model input or outcome selection','started_epoch':started,'passed':False}
    try:
        row=task_row(task);runtime=RemoteEpisode(row['config'],spec_for(row),directory,0,[],worker=c.suite_worker)
        actual=runtime.initial['fingerprint'];receipt['actual_fingerprint']=actual
        if actual!=receipt['expected_fingerprint']:raise ValueError('Third reset differs from original exact target')
        feedback=runtime.start_task(0)
        if not feedback['precondition_valid']:raise ValueError('Invalid initial task')
        obs=feedback['observation'];action=obs['whole_body_ik']['current_pose_hold_action'] if row['config']['robot']=='GR1' else [0.]*obs['action_schema']['shape'][0]
        began=time.monotonic()
        for _ in range(50):runtime.bridge({'operation':'step','action':action})
        duration=time.monotonic()-began;receipt['50_steps_seconds']=duration
        if max(original['50_steps_seconds'][prior_index],duration)>=75:raise ValueError('Execution margin failed')
        receipt['admitted_prior_reset_index']=prior_index+1
        receipt['passed']=True
    except Exception as exc:
        receipt['error_type']=type(exc).__name__;receipt['error']=str(exc)
    finally:
        try:
            if runtime:runtime.close()
        except Exception as exc:
            receipt['passed']=False;receipt['artifact_error']=str(exc)
        receipt['finished_epoch']=time.time();c.emit(folder/'exact-retry-receipt.json',receipt)
    if receipt['passed']:
        # Retain every rejected sample and the original paired receipt.
        updated={**original,'fingerprints':[original['fingerprints'][0],receipt['actual_fingerprint']],
                 '50_steps_seconds':[original['50_steps_seconds'][prior_index],receipt['50_steps_seconds']],
                 'same_reset':True,'passes_execution_margin':True,'admitted_reset_indices':[prior_index+1,3],
                 'rejected_reset_indices':[i for i in (1,2) if i!=prior_index+1],'admission_policy':receipt['policy']}
        c.emit(folder/'receipt.json',updated)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('task');parser.add_argument('--continue-queue',action='store_true');args=parser.parse_args()
    receipt=retry(args.task);print(json.dumps(receipt),flush=True)
    if not args.continue_queue:return
    quarantine=json.loads((ROOT/'quarantine.json').read_text())
    if receipt['passed']:quarantine.pop(args.task,None)
    else:quarantine[args.task]={'classification':'reset_preflight_infrastructure_blocked','reason':'Original pair and one bounded exact-target retry did not pass; no model input','evidence':f'preflight/{args.task}/exact-retry-receipt.json','repair_pending':True,'epoch':time.time()}
    c.emit(ROOT/'quarantine.json',quarantine)
    # Main queue takes its own exclusive flock and will refuse duplicate ownership.
    subprocess.run([sys.executable,'-u','-m','pairing.muse_queue'],check=True)


if __name__=='__main__':main()
