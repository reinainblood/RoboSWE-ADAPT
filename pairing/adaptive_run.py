"""New direct-API adaptive series; isolated from frozen campaign queues.

Plan-only is default. --run is an explicit launch action. No background queue,
legacy resume, paid preflight, or model-error retry is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import traceback
from pathlib import Path
from zipfile import ZipFile

from pairing import cohort50 as c
from pairing.context_history import POLICY
from pairing.direct_models import create_participant
from roboswe.jobs import create,save,now

PROTOCOL='adaptive-direct-sqlite-v1'
LIMITS=dict(active_seconds=1800,control_steps=6000,per_script_steps=50,per_script_seconds=60)


def build_plan(task_id,config,*,trials=5,code_image,cohort_path=None,video_judge=None):
    if video_judge not in (None,'glimpse'):raise ValueError('Unsupported video judge')
    if type(trials) is not int or not 1<=trials<=20:raise ValueError('Trials must be 1..20')
    if config.get('provider') not in ('openai','anthropic','openrouter','google'):raise ValueError('Unsupported direct provider')
    if not config.get('model'):raise ValueError('Explicit model required')
    if type(config.get('context_window')) is not int or type(config.get('max_output_tokens')) is not int:
        raise ValueError('Explicit integer context and output limits required')
    if config['max_output_tokens']<=0 or config['context_window']<=config['max_output_tokens']+4096:
        raise ValueError('Invalid context/output limits')
    cohort_path=Path(cohort_path or c.ROOT/'cohort.json')
    manifest=json.loads(cohort_path.read_text())
    candidates=[r for r in manifest['tasks'] if r['id']==task_id]
    if len(candidates)!=1:raise ValueError('Task ID must occur exactly once in frozen cohort')
    row=c.effective_row(candidates[0])
    preflight_path=c.preflight_directory(row)/'receipt.json'
    preflight=json.loads(preflight_path.read_text())
    if preflight.get('config')!=row['config'] or not preflight.get('fingerprint'):
        raise ValueError('Missing or mismatched frozen preflight')
    region_path=c.preflight_directory(row)/'counter-region-order.json'
    region=json.loads(region_path.read_text()) if region_path.exists() else None
    prompt=c.prompt_for(row['task']).replace('2-attempt',f'{trials}-attempt')
    prompt=prompt.replace('any compaction summary, ','')
    prompt+='\nThis new series uses sqlite-sliding-v1: exact history is archived, never summarized; retrieval is provided by the client.\n'
    return {'protocol':PROTOCOL,'context_policy':POLICY,'cohort':row,'participant':config,
        'trials':trials,'limits':LIMITS.copy(),'code_image':code_image,'prompt':prompt,
        'initial_fingerprint':preflight['fingerprint'],'counter_region_reference':region,
        'cohort_manifest_sha256':hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        'preflight_sha256':hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        'reset_policy':'fresh local worker; exact frozen preflight fingerprint every trial',
        'runtime':'local simulator worker; isolated Docker authored-code executor',
        'budget_source':'explicit RoboSWE protocol, NOT claimed upstream task defaults',
        'interpretation':'descriptive retained adaptation; no fresh-context control or causal ICL claim',
        'new_campaign':True,'legacy_resume':False,'score':None,
        'video_judge':video_judge,'mechanical_grade':'observed upstream sustained predicate only; no independent replay verification'}


def run(plan,*,jobs_dir='jobs',token_counter=None,episode_factory=None,resume_state=None):
    config=plan['participant']
    protocol=plan['protocol']
    public=plan['protocol']=='adaptive-public-sqlite-v1'
    if not public and (plan['protocol']!=PROTOCOL or plan['limits']!=LIMITS):raise ValueError('Unsupported protocol limits')
    factory=episode_factory or c.FreshEpisode
    limits=plan['limits']
    fingerprint=plan.get('initial_fingerprint')
    if config['provider']=='openrouter' and token_counter is None and config.get('context_policy') not in ('sqlite-sliding-or-conservative-v1','sqlite-sliding-or-google-count-v2'):
        raise ValueError('OpenRouter adaptive run requires verified model-aware counter; no paid run started')
    c.require_disk_headroom(Path(jobs_dir),5)
    if resume_state is None:
        directory,record=create(jobs_dir,f'Adaptive direct · {plan["cohort"]["task"]} · {plan["trials"]} trials','adaptive-direct-series')
        record.update(task=plan['cohort']['task'],provider=config['provider'],protocol=protocol,benchmark_score=None)
        c.emit(directory/'plan.json',plan);record['artifacts']['plan']='plan.json'
        with ZipFile(directory/'source.zip','w') as archive:
            for folder in ('pairing','roboswe','robot_session'):
                for path in (Path(c.__file__).resolve().parent.parent/folder).rglob('*.py'):
                    archive.write(path,arcname=str(Path(folder)/path.relative_to(Path(c.__file__).resolve().parent.parent/folder)))
        record['artifacts']['source']='source.zip';save(directory,record)
        reports=[]
    else:
        if not public:
            raise ValueError('Certified continuation requires public adaptive protocol')
        directory=Path(resume_state['parent'])
        record=dict(resume_state['record'])
        record.pop('finished_at',None)
        record.update(status='running',resume_receipt=resume_state['receipt'])
        reports=list(resume_state['reports'])
        save(directory,record)
    try:
        model=resume_state['model'] if resume_state else create_participant(plan['prompt'],directory,config,token_counter=token_counter)
        if resume_state is None:
            c.emit(directory/'model.json',model.metadata);record['artifacts']['model']='model.json'
        with model, c.CodeSandbox(plan['code_image'],lifetime_seconds=plan['trials']*(limits['active_seconds']*2+1200)+600) as sandbox:
            if resume_state:
                sandbox.restore_checkpoint(resume_state['workspace'])
            for number in range(resume_state['next_trial_number'] if resume_state else 1,plan['trials']+1):
                if resume_state and number==resume_state['next_trial_number']:
                    attempt=Path(resume_state['prefix'])
                    attempt.rename(Path(resume_state['preserved_prefix']))
                    attempt.mkdir()
                    job=dict(resume_state['prefix_record'])
                    job.pop('finished_at',None)
                    job.update(status='created',artifacts={},artifact_errors=[],
                               maintenance_resume_receipt=resume_state['receipt'])
                    c.emit(attempt/'maintenance-resume.json',{'receipt':resume_state['receipt'],
                        'preserved_zero_input_prefix':resume_state['preserved_prefix'],
                        'same_original_trial_number':number,'model_input_before_boundary':False})
                else:
                    attempt,job=create(jobs_dir,f'{config["model"]} · {plan["cohort"]["task"]} · {number}/{plan["trials"]}','adaptive-direct-attempt')
                job.update(campaign_id=record['id'],task=plan['cohort']['task'],provider=config['provider'],
                           attempt_number=number,protocol=protocol,benchmark_score=None)
                record.update(current_job=job['id'],current_attempt=number);save(directory,record)
                model.directory=attempt
                spec=c.serial_spec(plan['cohort']['task'],timeout=limits['active_seconds'],max_steps=limits['control_steps'])
                if public:
                    execution=plan['execution']
                    spec.update(execution_backend=execution['backend'],
                        docker_simulator_image=execution['simulator_image'] if execution['backend']=='docker' else None,
                        modal_image_id=execution['simulator_image'] if execution['backend']=='modal' else None)
                    spec.update({key:value for key,value in execution.items() if key.startswith('modal_')})
                    if execution.get('artifact_transfer'):spec['artifact_transfer']=execution['artifact_transfer']
                spec.update(learning_trials=plan['trials'],cohort_protocol=protocol,
                    transport_optimizations=c.TRANSPORT_VERSION,sensor_documentation=c.SENSOR_DOCUMENTATION,
                    hand_action_contract=c.HAND_ACTION_CONTRACT if plan['cohort']['config']['robot']=='GR1' else 'upstream-standard',
                    environment_documentation=c.ENVIRONMENT_DOCUMENTATION,execution_limit_documentation=c.EXECUTION_LIMIT_DOCUMENTATION,
                    observation_feedback_version=c.OBSERVATION_FEEDBACK_VERSION,budget_feedback_version=c.BUDGET_FEEDBACK_VERSION,
                    reset_protocol='counter-region-order-v1' if plan['counter_region_reference'] else 'seeded-rng-plus-hashseed-v2')
                if plan['counter_region_reference']:spec['counter_region_reference']=plan['counter_region_reference']
                spec['sha256']=hashlib.sha256(json.dumps({k:v for k,v in spec.items() if k!='sha256'},sort_keys=True).encode()).hexdigest()
                runtime=journal=None
                attempt_failed=False
                try:
                    model.save_context(attempt/'context-start.json.gz')
                    runtime=factory(plan['cohort']['config'],spec,attempt,number,reports,worker=c.suite_worker)
                    if fingerprint is None:fingerprint=runtime.initial['fingerprint']
                    if runtime.initial['fingerprint']!=fingerprint:raise RuntimeError('Reset differs from reference fingerprint; no model input sent')
                    c.emit(attempt/'continuity.json',dict(initial_fingerprint=runtime.initial['fingerprint'],retained=number>1,
                        thread_id=model.thread_id,workspace_keeper=sandbox.keeper,library_start=(sandbox.checkpoint if public else sandbox.archive)(attempt/'library-start.zip')))
                    journal=c.Journal(attempt/'submissions.jsonl');job['status']='running';save(attempt,job)
                    drive_options={'execution_seconds':limits['per_script_seconds']}
                    if public:drive_options.update(episode_steps=limits['control_steps'],script_steps=limits['per_script_steps'])
                    if resume_state and resume_state.get('stop_check'):
                        resume_state['stop_check']()
                    report=c.drive(c.EfficientRunner(runtime,sandbox,journal),model,attempt,**drive_options)
                    report.update(attempt_number=number,provider=config['provider'],protocol=protocol,
                                  retained_context=number>1,retained_library=number>1,library=(sandbox.checkpoint if public else sandbox.archive)(attempt/'library.zip'))
                    c.emit(attempt/'report.json',report)
                    reports.append({key:report[key] for key in ('attempt_number','observed_success','termination_reason','control_steps','elapsed_seconds')})
                    job['status']='observed_pass' if report['observed_success'] else 'observed_fail'
                    if report['termination_reason']=='precondition_invalid':job['status']='invalid_precondition'
                    job['artifacts'].update(report='report.json',library='library.zip',timing='timing.jsonl',transcript='model-transcript.jsonl')
                except BaseException as exc:
                    attempt_failed=True
                    job['status']=getattr(exc,'status','infrastructure_error');(attempt/'stderr.txt').write_text(traceback.format_exc());job['artifacts']['stderr']='stderr.txt'
                    try:
                        (sandbox.checkpoint if public else sandbox.archive)(attempt/'library-interrupted.zip')
                        job['artifacts']['interrupted_library']='library-interrupted.zip'
                    except Exception as archive_error:
                        job['artifact_errors'].append({'stage':'interrupted_library','error':str(archive_error)[:500]})
                    raise
                finally:
                    try:
                        model.save_context(attempt/'context-end.json.gz')
                        model.history_store.export(attempt)
                        if resume_state:
                            model.history_store.export(directory)
                        for name in ('history.sqlite','history-manifest.json','provider-thinking.jsonl','provider-responses.jsonl','model-usage.jsonl','token-counts.jsonl','provider-request-events.jsonl','provider-unsettled-responses.jsonl'):
                            if (attempt/name).exists():job['artifacts'][name]=name
                    finally:
                        if journal:journal.close()
                        try:
                            if runtime:
                                receipt=runtime.close();c.publish_video(Path(receipt['recording_dir']),attempt,job,phase='original')
                        finally:
                            job['finished_at']=now();save(attempt,job)
                            c.emit(directory/'report.json',{'trials':reports,'benchmark_score':None,'protocol':protocol})
                            if attempt_failed and public and plan['execution']['backend']=='modal':
                                try:
                                    from roboswe.adaptive_transport import archive_finalized_attempt
                                    archive_finalized_attempt(attempt,plan['execution'])
                                except Exception as archive_error:
                                    job['artifact_errors'].append({'stage':'failed_attempt_archive','error':str(archive_error)[:500]})
                                    save(attempt,job)
                if public:
                    verification=verify_attempt(plan,spec,attempt,number,reports[:-1],factory)
                    c.emit(attempt/'verification.json',verification)
                    job['artifacts']['verification']='verification.json'
                    job['status']='verified_pass' if verification.get('verified_success') else 'verified_fail' if verification.get('reproduced') else 'verification_invalid'
                    save(attempt,job)
                    if verification.get('artifact_durability_error'):
                        from roboswe.adaptive_transport import ArtifactDurabilityError
                        raise ArtifactDurabilityError(verification['artifact_durability_error'])
                    if not verification.get('reproduced'):
                        raise RuntimeError('Independent replay invalid; stopping series without retry')
                if plan.get('video_judge'):
                    try:
                        from roboswe.library.predicate_audit import audit_job
                        judge = plan['video_judge']
                        options = {}
                        if isinstance(judge, dict):
                            from roboswe.library.judge_budget import budgeted_request
                            options = {
                                'judge_backend': judge['backend'],
                                'reasoning_effort': 'low' if judge['backend'] == 'gemini' else None,
                                'max_tokens': 8192,
                                'request_fn': budgeted_request(
                                    ledger_path=judge['ledger_path'],
                                    limit_usd=judge['spend_limit_usd'],
                                    receipts=attempt/'judge-provider-receipts',
                                    judge_backend=judge['backend']),
                            }
                        audit=audit_job(attempt, **options)
                        c.emit(attempt/'visual-audit.json',audit)
                        job['artifacts']['visual_audit']='visual-audit.json'
                    except Exception as exc:
                        job['artifact_errors'].append({'stage':'predicate_video_audit','error':str(exc)[:500]})
                    save(attempt,job)
                if public and plan['execution']['backend']=='modal':
                    from roboswe.adaptive_transport import archive_finalized_attempt
                    archive_finalized_attempt(attempt,plan['execution'])
            record['status']='completed_unscored'
    except BaseException as exc:
        record['status']=getattr(exc,'status','infrastructure_error');(directory/'stderr.txt').write_text(traceback.format_exc());record['artifacts']['stderr']='stderr.txt'
        raise
    finally:
        record['finished_at']=now()
        if (directory/'report.json').exists():record['artifacts']['report']='report.json'
        if (directory/'history.sqlite').exists():record['artifacts']['history']='history.sqlite'
        save(directory,record)
    return directory


def verify_attempt(plan,spec,attempt,number,previous,factory):
    """Replay authored operations in a fresh scene/workspace; never ask a model."""
    from roboswe.library.attempts import replay_journal
    destination=attempt/'verification';destination.mkdir()
    runtime=None
    result=None
    try:
        with c.CodeSandbox(plan['code_image'],lifetime_seconds=plan['limits']['active_seconds']+600) as sandbox:
            sandbox.restore_checkpoint(attempt/'library-start.zip')
            runtime=factory(plan['cohort']['config'],spec,destination,number,previous,worker=c.suite_worker)
            result=replay_journal(attempt/'submissions.jsonl',runtime,sandbox,destination/'submissions.jsonl',runner_factory=c.EfficientRunner)
            result['verified_success']=bool(result.get('reproduced') and len(result.get('trials',[]))==1 and result['trials'][0].get('verified_success'))
            return result
    except Exception as exc:
        return {'verified_success':False,'reproduced':False,'infrastructure_valid':False,'error_type':type(exc).__name__,'error':str(exc)[:500]}
    finally:
        if runtime:
            try:runtime.close()
            except Exception:
                (destination/'close-error.txt').write_text(traceback.format_exc())
                if isinstance(result,dict) and plan.get('execution',{}).get('backend')=='modal':
                    result['artifact_durability_error']='Replay artifact readback/finalization failed; local evidence preserved; no model retry'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id',required=True)
    parser.add_argument('--provider',required=True,choices=['openai','anthropic','openrouter','google'])
    parser.add_argument('--model',required=True)
    parser.add_argument('--context-window',required=True,type=int)
    parser.add_argument('--max-output-tokens',required=True,type=int)
    parser.add_argument('--code-image',required=True)
    parser.add_argument('--trials',type=int,default=5)
    parser.add_argument('--effort',default='high')
    parser.add_argument('--workspace-id')
    parser.add_argument('--video-judge',choices=['glimpse'],help='Opt-in paid visual predicate cross-check after each finalized trial')
    parser.add_argument('--episode-seconds',type=int,choices=[1800],default=1800)
    parser.add_argument('--control-steps',type=int,choices=[6000],default=6000)
    parser.add_argument('--script-seconds',type=int,choices=[60],default=60)
    parser.add_argument('--script-steps',type=int,choices=[50],default=50)
    parser.add_argument('--run',action='store_true',help='Explicitly launch paid inference and local simulation; otherwise print plan only')
    args=parser.parse_args(argv)
    config={key:getattr(args,key) for key in ('provider','model','context_window','max_output_tokens','effort')}
    if args.workspace_id:config['workspace_id']=args.workspace_id
    plan=build_plan(args.task_id,config,trials=args.trials,code_image=args.code_image,video_judge=args.video_judge)
    if args.run:print(run(plan))
    else:print(json.dumps(plan,indent=2))


if __name__=='__main__':main()
