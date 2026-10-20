"""Continue the completed GR1 pilot as attempt 1 of a three-attempt ICL sequence."""
import json,time,traceback
from pathlib import Path
from zipfile import ZipFile
from pairing.run_toaster_learning import FreshEpisode,LearningModel,instructions,write_human_outputs,usage_summary,EXPECTED_ACCOUNT
from pairing.run_hand_microwave import trial_slot
from pairing.simulation_wait import WaitingAttemptRunner
from roboswe.execution.sandbox import CodeSandbox
from roboswe.library.attempts import Journal
from roboswe.library.serial import serial_spec
from roboswe.campaign import drive_task,publish_video
from roboswe.jobs import create,save,now

FIRST=Path('jobs/20261005T030326Z-e4fe49175d')


def gr1_worker(pipe,config,spec,directory,number,previous):
    import pairing.run_gr1_trial as gr1
    from pairing.gr1_compat import install_stock_gr1_placement
    directory=Path(directory);gr1.ROOT=directory
    install_stock_gr1_placement()
    session=runtime=evidence=None
    try:
        session=gr1.GR1Session(config);session.recording_root=directory/'recordings'
        session.start_recording({'recording_id':'attempt','width':640,'height':480})
        evidence=(directory/'actions.jsonl').open('x')
        runtime=gr1.RetrievalRuntime(session,spec,timeout=1800,max_steps=6000,evidence=evidence)
        pipe.send({'ok':runtime.metadata(),'fingerprint':session.initial})
        while True:
            request=pipe.recv();op=request['operation']
            try:
                if op=='start_task':
                    result=runtime.start_task(0)
                    reason=previous[-1]['termination_reason']
                    ended='reached its 30-minute active-work timeout' if reason in ('task_deadline','model_deadline') else 'ended with '+reason
                    result.update(trial_number=number,total_trials=3,previous_attempts=previous,
                        reset_notice=f'Attempt {number-1} {ended}. You are now starting attempt {number} of 3 with a new 30-minute active-work allowance. The toaster, bread, robot body, both arms, and both hands have been reset to the exact starting state. Your existing conversation context, any compaction summary, all Python library files, and NOTES.md have been maintained. Continue learning from your previous attempts; do not assume the last physical pose is still present.')
                elif op=='observe':result=runtime.observe()
                elif op=='bridge':result=runtime.bridge(request['message'])
                elif op=='wait':result=runtime.wait(request['steps'])
                elif op=='finish_task':result=runtime.finish_task(code_errors=request['code_errors'])
                elif op=='digest':result=runtime.action_digest.hexdigest()
                elif op=='close':
                    session.stop_recording({'recording_id':'attempt'});(directory/'final-snapshot.zip').write_bytes(session.snapshot())
                    pipe.send({'ok':{'recording_dir':str(session.recording.directory)}});break
                else:raise ValueError('Unknown coordinator operation')
                pipe.send({'ok':result})
            except Exception as exc:pipe.send({'error':traceback.format_exc(),'error_type':type(exc).__name__})
    except BaseException:
        try:pipe.send({'error':traceback.format_exc()})
        except (BrokenPipeError,EOFError):pass
    finally:
        if runtime:runtime.diagnostic.close()
        if evidence:evidence.close()
        if session:
            if session.recording and session.recording.active:session.stop_recording({'recording_id':'attempt'})
            session.close()
        pipe.close()


def restore_library(sandbox,archive_path,receipt_path):
    with ZipFile(archive_path) as archive:
        manifest=json.loads(archive.read('library-manifest.json'))
        for item in manifest['files']:
            name=item['path'];data=archive.read(name).decode()
            result=sandbox.request('write',name,content=data,timeout=60)
            if result['returncode'] or result['error']:raise RuntimeError('Could not restore library file '+name)
    restored=sandbox.archive(receipt_path)
    expected={x['path']:x['sha256'] for x in manifest['files']}
    actual={x['path']:x['sha256'] for x in restored['files']}
    if actual!=expected:raise RuntimeError('Restored library hashes differ from saved checkpoint')
    return restored


def main():
    first_plan=json.loads((FIRST/'plan.json').read_text());first_model=json.loads((FIRST/'model.json').read_text());first_report=json.loads((FIRST/'report.json').read_text())
    directory,record=create('jobs','Astra high · GR1 toast retrieval · 3× ICL · 30 min','gr1-learning-sequence')
    record.update(status='queued',task='GetToastedBread · retrieval only',benchmark_score=None,attempt_jobs=[FIRST.name])
    Path('artifacts/gr1-toast-trial/icl-job.json').write_text(json.dumps({'id':record['id'],'directory':str(directory)}));save(directory,record)
    prompt=first_plan['instructions'].replace('1-attempt learning sequence','3-attempt learning sequence').replace('Your starting library is empty.','Your existing code library and NOTES.md have been restored from attempt 1.')
    prompt+='\nKirsten has extended this SAME experiment to three attempts total. The completed pilot counts as attempt 1. Attempts 2 and 3 reset physics, but preserve your actual conversation and library. Each reset notice describes what ended and the new budget. Keep updating NOTES.md and refine your existing code.\n'
    image=Path('artifacts/toaster-learning-code-image.txt').read_text().strip()
    plan={**first_plan,'instructions':prompt,'total_trials':3,'first_attempt':FIRST.name,'resume_thread_id':first_model['thread_id'],'restore_library_from':str(FIRST/'library.zip'),'context_window':1000000,'compact_at':800000}
    (directory/'plan.json').write_text(json.dumps(plan,indent=2));record['artifacts']['plan']='plan.json';save(directory,record)
    first_report={**first_report,'attempt_number':1,'job_id':FIRST.name};reports=[first_report]
    try:
        with trial_slot({'execution':{'memory_mb':6144,'cpu':4,'min_free_disk_gb':10},'jobs_dir':'jobs'},True), CodeSandbox(image,lifetime_seconds=6500) as sandbox, LearningModel(model='gpt-6-astra',effort='high',instructions=prompt,context_window=1000000,compact_at=800000,persistent=True,resume_thread_id=first_model['thread_id'],reasoning_summary='detailed',verbosity='high',diagnostics=directory/'model-events.jsonl',summary_log=directory/'reasoning-summaries.jsonl') as model:
            if model._rpc('account/rateLimits/read',{}).get('accountId')!=EXPECTED_ACCOUNT:raise RuntimeError('Active account differs from selected RoboSWE account')
            restored=restore_library(sandbox,FIRST/'library.zip',directory/'restored-library.zip')
            (directory/'continuity.json').write_text(json.dumps({'original_thread_id':first_model['thread_id'],'resumed_thread_id':model.thread_id,'last_completed_turn':model.last_completed_turn,'restored_library':restored,'workspace_keeper':sandbox.keeper},indent=2))
            (directory/'model.json').write_text(json.dumps(model.metadata,indent=2))
            record.update(status='running',model_thread_id=model.thread_id,workspace_keeper=sandbox.keeper);save(directory,record)
            turn=0
            for number in (2,3):
                if model._rpc('account/rateLimits/read',{}).get('accountId')!=EXPECTED_ACCOUNT:raise RuntimeError('RoboSWE account changed')
                attempt,job=create('jobs',f'Astra high · GR1 toast retrieval · ICL {number}/3','gr1-retrieval-diagnostic')
                job.update(campaign_id=record['id'],attempt_number=number,task='GetToastedBread · retrieval only',benchmark_score=None,verifier_status='diagnostic_only')
                save(attempt,job);record['attempt_jobs'].append(job['id']);record.update(current_attempt=number,current_job=job['id']);save(directory,record)
                runtime=journal=None
                try:
                    spec=serial_spec('GetToastedBread',timeout=1800,max_steps=6000)
                    previous=[{k:r[k] for k in ['attempt_number','termination_reason','observed_success','control_steps']} for r in reports]
                    runtime=FreshEpisode(first_plan['config'],spec,attempt,number,previous,worker=gr1_worker)
                    if runtime.initial['fingerprint']!=first_report['start_fingerprint']:raise RuntimeError('GR1 reset fingerprint differs from attempt 1')
                    journal=Journal(attempt/'submissions.jsonl');runner=WaitingAttemptRunner(runtime,sandbox,journal)
                    with (attempt/'model-transcript.jsonl').open('x') as transcript:
                        report,turn=drive_task(runner,model,0,{'max_control_steps':6000,'max_attempt_steps':50,'attempt_timeout_seconds':60},attempt,transcript,time.monotonic()+2700,turn)
                    report.update(attempt_number=number,job_id=job['id'],model_thread_id=model.thread_id,retained_context=True,retained_library=True,usage_after=usage_summary(directory/'model-events.jsonl'))
                    report['library']=sandbox.archive(attempt/'library.zip')
                    with ZipFile(attempt/'library.zip') as archive:
                        if 'NOTES.md' in archive.namelist():(attempt/'NOTES.md').write_bytes(archive.read('NOTES.md'));job['artifacts']['notes']='NOTES.md'
                    (attempt/'report.json').write_text(json.dumps(report,indent=2));job['artifacts'].update(report='report.json',library='library.zip')
                    write_human_outputs(attempt);job['artifacts']['stdout']='stdout.txt'
                    job['status']='diagnostic_pass' if report['observed_success'] else 'diagnostic_incomplete';reports.append(report)
                except BaseException:
                    job['status']='infrastructure_error';(attempt/'stderr.txt').write_text(traceback.format_exc());job['artifacts']['stderr']='stderr.txt';raise
                finally:
                    if journal:journal.close()
                    if runtime:
                        result=runtime.close();publish_video(Path(result['recording_dir']),attempt,job,phase='original')
                    job['finished_at']=now();save(attempt,job)
                    (directory/'report.json').write_text(json.dumps({'trials':reports,'benchmark_score':None,'scope':'GR1 retrieval diagnostic, not full delivery'},indent=2));record['artifacts']['report']='report.json';save(directory,record)
            record.update(status='completed_unscored',exit_code=0)
    except BaseException:
        record.update(status='error',exit_code=1);(directory/'stderr.txt').write_text(traceback.format_exc());record['artifacts']['stderr']='stderr.txt';raise
    finally:record['finished_at']=now();save(directory,record)
if __name__=='__main__':main()
