"""Muse contributor: frozen same-50 cohort, three retained Modal episodes."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from zipfile import ZipFile
from pairing import cohort50 as c
from pairing.modal_episode import RemoteEpisode
from pairing.muse_model import MuseModel
from pairing.run_qwen_icl import save_context
from roboswe.jobs import create,save,now

ROOT=Path('artifacts/muse-learning-20261006')

def task_row(task_id):
    return c.effective_row(next(r for r in c.freeze()['tasks'] if r['id']==task_id))

def spec_for(row):
    spec=c.serial_spec(row['task'],timeout=1800,max_steps=6000)
    image_receipt=Path('artifacts/modal-muse-region-tolerance-image.json')
    if image_receipt.exists():
        spec.update(modal_image_id=json.loads(image_receipt.read_text())['image'],counter_region_match_atol=1e-12)
    spec.update(artifact_transfer='remote-video',learning_trials=3,cohort_protocol=c.VERSION,transport_optimizations=c.TRANSPORT_VERSION,sensor_documentation=c.SENSOR_DOCUMENTATION,hand_action_contract=c.HAND_ACTION_CONTRACT if row['config']['robot']=='GR1' else 'upstream-standard',environment_documentation=c.ENVIRONMENT_DOCUMENTATION,execution_limit_documentation=c.EXECUTION_LIMIT_DOCUMENTATION,observation_feedback_version=c.OBSERVATION_FEEDBACK_VERSION,budget_feedback_version=c.BUDGET_FEEDBACK_VERSION)
    region=c.preflight_directory(row)/'counter-region-order.json'
    if region.exists():spec.update(counter_region_reference=json.loads(region.read_text()),reset_protocol='counter-region-order-v1')
    spec['sha256']=hashlib.sha256(json.dumps({k:v for k,v in spec.items() if k!='sha256'},sort_keys=True).encode()).hexdigest()
    return spec

def preflight(task_id):
    row=task_row(task_id);folder=ROOT/'preflight'/task_id;folder.mkdir(parents=True,exist_ok=True)
    fingerprints=[];times=[]
    for n in (1,2):
        directory=folder/str(n);directory.mkdir(exist_ok=False)
        runtime=None
        try:
            runtime=RemoteEpisode(row['config'],spec_for(row),directory,0,[],worker=c.suite_worker)
            fingerprints.append(runtime.initial['fingerprint'])
            feedback=runtime.start_task(0)
            if not feedback['precondition_valid']:raise RuntimeError('Initially successful or invalid scene')
            obs=feedback['observation'];action=obs['whole_body_ik']['current_pose_hold_action'] if row['config']['robot']=='GR1' else [0.]*obs['action_schema']['shape'][0]
            began=time.monotonic()
            for _ in range(50):runtime.bridge({'operation':'step','action':action})
            times.append(time.monotonic()-began)
        finally:
            if runtime:runtime.close()
    receipt={'config':row['config'],'fingerprints':fingerprints,'50_steps_seconds':times,'same_reset':len(set(fingerprints))==1,'passes_execution_margin':max(times)<75,'execution_timeout_seconds':90,'margin_seconds':15,'backend':'Modal FDR L4 Linux; separate local Docker authored-code sandbox','historical_comparison':'Same task/config/seed; cross-platform numerical equivalence is not assumed.'}
    c.emit(folder/'receipt.json',receipt)
    if not receipt['same_reset'] or not receipt['passes_execution_margin']:raise RuntimeError('Modal reset/throughput preflight failed; do not spend model budget')
    return receipt

def admit_episode(row,spec,directory,number,reports,expected,*,max_resets=3):
    """Retry only simulator admission, before the model receives any input.

    Require the original exact fingerprint on every accepted attempt. Rejected
    startup artifacts are retained; no model episode or outcome is retried here.
    """
    admission=[]
    for reset in range(1,max_resets+1):
        existing={p.name for p in directory.iterdir()}
        runtime=RemoteEpisode(row['config'],spec,directory,number,reports,worker=c.suite_worker)
        actual=runtime.initial['fingerprint']
        admission.append({'reset':reset,'expected':expected,'actual':actual,'accepted':actual==expected,'model_input_delivered_before_admission':False})
        if actual==expected:
            c.emit(directory/'reset-admission.json',{'resets':admission,'policy':'Exact fingerprint only; bounded infrastructure admission retries before episode start.'})
            return runtime
        runtime.close()
        rejected=directory/'rejected-resets'/str(reset);rejected.mkdir(parents=True)
        for path in list(directory.iterdir()):
            if path.name not in existing and path.name!='rejected-resets':path.rename(rejected/path.name)
        c.emit(directory/'reset-admission.json',{'resets':admission,'policy':'Rejected reset never shown to model; private artifacts preserved.'})
    raise RuntimeError('Exact initial fingerprint unavailable after bounded reset admission; no model episode started')


def series(task_id,resume_series=None):
    row=task_row(task_id);ROOT.mkdir(parents=True,exist_ok=True)
    recovery=None
    if resume_series:
        from pairing.muse_checkpoint import checkpoint
        recovery=checkpoint(resume_series)
        if recovery['plan']['cohort']!=row:raise ValueError('Resume cohort differs')
    frozen_spec=recovery['spec'] if recovery else spec_for(row)
    pre=ROOT/'preflight'/task_id/'receipt.json'
    receipt=recovery['plan']['preflight'] if recovery else (json.loads(pre.read_text()) if pre.exists() else preflight(task_id))
    if not receipt['same_reset'] or max(receipt['50_steps_seconds'])>=75:raise RuntimeError('Preflight failed')
    c.emit(ROOT/'preflight'/task_id/'execution-budget-review.json',{'original_receipt':receipt,'approved_execution_seconds':90,'minimum_margin_seconds':15,'reason':'Remote transport needs uniform extra script wall-clock margin; all three trials use identical limits; episode remains 1800 seconds.'})
    prompt=c.prompt_for(row['task']).replace('2-attempt','3-attempt').replace('50-step/60-second','50-step/90-second').replace('50-step, 60-second','50-step, 90-second')
    prompt=prompt.replace('You have no built-in tools. Each response must be exactly one JSON operation', 'Your only tool is robot_decision, which submits one operation to this harness. Call it exactly once per response with arguments')
    prompt+='\nAcross all three attempts, maintain a reusable reset-ready entry point or a short ordered list of your tested scripts in NOTES.md. On a reset, use your verified code when the observed starting conditions match, and focus new probes on unresolved steps. Do not claim physical progress until execution feedback confirms it.\n'
    if recovery:prompt=recovery['plan']['prompt']
    directory,record=create('jobs',f'Muse contributor · {task_id} {row["task"]} · 3×30m retained ICL · Modal','muse-cohort-series')
    record.update(cohort_id=task_id,provider='muse',status='running',benchmark_score=None)
    if recovery:record.update(continuation_of=resume_series,original_attempt_jobs=recovery['completed_directories'])
    c.emit(ROOT/(task_id+'.json'),{'job_id':record['id'],'directory':str(directory)})
    image=recovery['plan']['code_image'] if recovery else Path('artifacts/toaster-learning-code-image.txt').read_text().strip()
    c.emit(directory/'plan.json',{'cohort':row,'prompt':prompt,'attempts':3,'spec':frozen_spec,'code_image':image,'modal_image':recovery['plan']['modal_image'] if recovery else json.loads(Path('artifacts/modal-muse-region-tolerance-image.json' if Path('artifacts/modal-muse-region-tolerance-image.json').exists() else 'artifacts/modal-trials-image.json').read_text()),'preflight':receipt,'resume_from':resume_series,'original_attempt_jobs':recovery['completed_directories'] if recovery else [],'control_arm':'pending user preference','budgets':{'active_seconds':1800,'control_steps':6000,'per_execution_steps':50,'per_execution_seconds':90}})
    with ZipFile(directory/'source.zip','w') as archive:
        for folder in ('pairing','roboswe','robot_session'):
            for path in Path(folder).rglob('*.py'):archive.write(path)
    record['artifacts'].update(plan='plan.json',source='source.zip');save(directory,record)
    reports=list(recovery['prior']) if recovery else []
    model=MuseModel(prompt,directory,context_policy="legacy",decision_transport=recovery['model']['decision_transport'] if recovery else 'native-tool-results-v2')
    c.emit(directory/'model.json',model.metadata)
    try:
        with c.CodeSandbox(image,lifetime_seconds=10000) as sandbox:
            if recovery:
                from pairing.muse_checkpoint import restore
                restore(recovery,model,sandbox,directory)
            for number in range(len(reports)+1,4):
                attempt,job=create('jobs',f'Muse · {task_id} {row["task"]} · ICL {number}/3','muse-cohort-attempt')
                job.update(campaign_id=record['id'],cohort_id=task_id,provider='muse',attempt_number=number,status='starting',benchmark_score=None)
                record.update(current_job=job['id'],current_attempt=number);save(directory,record);save(attempt,job)
                model.directory=attempt;runtime=journal=None
                try:
                    runtime=admit_episode(row,frozen_spec,attempt,number,reports,receipt['fingerprints'][0])
                    c.emit(attempt/'continuity.json',{'initial_fingerprint':runtime.initial['fingerprint'],'workspace_keeper':sandbox.keeper,'retained':number>1,'context':save_context(attempt/'context-start.json.gz',model.messages),'library_start':sandbox.archive(attempt/'library-start.zip')})
                    journal=c.Journal(attempt/'submissions.jsonl');job['status']='running';save(attempt,job)
                    report=c.drive(c.EfficientRunner(runtime,sandbox,journal),model,attempt,execution_seconds=90)
                    report.update(attempt_number=number,provider='muse',cohort_id=task_id,retained_context=number>1,retained_library=number>1,library=sandbox.archive(attempt/'library.zip'))
                    c.emit(attempt/'report.json',report)
                    reports.append({k:report[k] for k in ('attempt_number','observed_success','termination_reason','control_steps','elapsed_seconds')})
                    # Persist completed task results before artifact-finalization I/O.
                    c.emit(directory/'report.json',{'trials':reports,'benchmark_score':None})
                    job['status']='observed_pass' if report['observed_success'] else 'observed_fail'
                    job['artifacts'].update(report='report.json',library='library.zip',timing='timing.jsonl')
                except BaseException:
                    job['status']='infrastructure_error';(attempt/'stderr.txt').write_text(traceback.format_exc())
                    if not (attempt/'library.zip').exists():
                        try:
                            sandbox.archive(attempt/'library.zip');job['artifacts']['library']='library.zip'
                        except Exception as archive_error:(attempt/'archive-error.txt').write_text(str(archive_error))
                    raise
                finally:
                    save_context(attempt/'context-end.json.gz',model.messages)
                    if journal:journal.close()
                    try:
                        if runtime:
                            close=runtime.close()
                            job['remote_artifacts']=close['remote_artifacts']
                            video=json.loads((Path(close['recording_dir'])/'video.json').read_text())
                            job['video']=video
                            job['artifacts']['remote_artifacts']='remote-artifacts.json'
                    finally:job['finished_at']=now();save(attempt,job)
                    c.emit(directory/'report.json',{'trials':reports,'benchmark_score':None})
        record['status']='completed_unscored'
    except BaseException:
        record['status']='infrastructure_error';(directory/'stderr.txt').write_text(traceback.format_exc());raise
    finally:record['finished_at']=now();save(directory,record)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','series']);p.add_argument('task_id');p.add_argument('--resume-series');a=p.parse_args()
    preflight(a.task_id) if a.mode=='preflight' else series(a.task_id,resume_series=a.resume_series)
