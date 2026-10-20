"""Frozen 50-task, paired-provider retained-learning campaign (unscored pilot)."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZipFile

from roboswe.catalog import load_catalog
from roboswe.jobs import create, save, now
from roboswe.library.serial import serial_spec
from roboswe.library.attempts import Journal
from roboswe.execution.sandbox import CodeSandbox
from roboswe.campaign import publish_video, model_input
from roboswe.storage import require_disk_headroom
from pairing.overnight_suite import suite_worker
from pairing.run_toaster_learning import FreshEpisode, instructions, LearningModel, EXPECTED_ACCOUNT, write_human_outputs
from pairing.run_qwen_icl import LearningQwen, save_context
from pairing.simulation_wait import WaitingAttemptRunner, WAIT_SCHEMA

ROOT = Path('artifacts/cohort50-20261005')
VERSION = 'cohort50-v1'
TRANSPORT_VERSION = 'sensor-serialization-cache-v1'
OBSERVATION_FEEDBACK_VERSION = 'explicit-observe-refresh-v3'
SENSOR_DOCUMENTATION = 'nested-rgb-v1'
HAND_ACTION_CONTRACT = 'native-raw-position-v2'
ENVIRONMENT_DOCUMENTATION = 'installed-numerics-v1'
EXECUTION_LIMIT_DOCUMENTATION = 'per-script-limit-v2'
BUDGET_FEEDBACK_VERSION = 'remaining-work-clock-v1'
SCHEMA = copy.deepcopy(WAIT_SCHEMA)
SCHEMA['properties']['operation']['enum'].append('write_execute')
EASY = 'CloseMicrowave TurnOnSinkFaucet TurnOffSinkFaucet OpenMicrowave CloseCabinet OpenCabinet CloseDrawer OpenDrawer TurnOnMicrowave TurnOffMicrowave StartCoffeeMachine TurnOnBlender TurnOnToaster TurnOnElectricKettle CloseElectricKettleLid OpenElectricKettleLid CloseToasterOvenDoor OpenToasterOvenDoor CloseBlenderLid OpenBlenderLid'.split()
MEDIUM = 'PickPlaceCounterToBlender PickPlaceCounterToCabinet PickPlaceCounterToDrawer PickPlaceCounterToMicrowave PickPlaceCounterToSink PickPlaceCabinetToCounter PickPlaceDrawerToCounter PickPlaceMicrowaveToCounter PickPlaceSinkToCounter CoffeeSetupMug CoffeeServeMug AdjustWaterTemperature TurnSinkSpout SlideToasterOvenRack NavigateKitchen'.split()
HARD = 'BlendIngredients ToastBagel GetToastedBread PrepareCoffee FillKettle KettleBoiling PrepareSmoothie BlendMarinade FillBlenderJug LoadDishwasher ReheatMeal RinseBowls WashFruitColander MultistepSteaming SteamInMicrowave'.split()
GR1_TASKS = set('OpenMicrowave CloseCabinet OpenCabinet CloseDrawer OpenDrawer TurnOnBlender TurnOnToaster CloseBlenderLid OpenBlenderLid CloseElectricKettleLid OpenElectricKettleLid TurnOffMicrowave'.split())


def emit(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def effective_row(row):
    """Explicit pre-evaluation scene revisions; never mutate the frozen cohort."""
    if 'config' not in row:return row
    path=ROOT/'case-revisions'/f'{row["id"]}.json'
    if not path.exists():return row
    import re
    revision=json.loads(path.read_text())
    if revision['original_config']!=row['config']:raise ValueError('Case revision has a different base configuration')
    if not re.fullmatch('[a-z0-9-]+',revision['revision_id']):raise ValueError('Invalid case revision ID')
    updated=revision['config'];original=row['config']
    if {k:v for k,v in updated.items() if k!='env_kwargs'}!={k:v for k,v in original.items() if k!='env_kwargs'}:
        raise ValueError('Scene revisions cannot silently change task, robot, or seed')
    changed={k for k in set(updated['env_kwargs'])|set(original['env_kwargs']) if updated['env_kwargs'].get(k)!=original['env_kwargs'].get(k)}
    if not changed.issubset({'layout_ids','style_ids'}):raise ValueError('Unsupported scene revision')
    return {**row,'config':copy.deepcopy(updated),'original_config':copy.deepcopy(original),
        'case_revision':revision['revision_id'],'case_revision_reason':revision['reason']}


def preflight_directory(row):
    path=ROOT/'preflight'/row['id']
    return path/row['case_revision'] if row.get('case_revision') else path


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/'cohort.json'
    if path.exists():
        return json.loads(path.read_text())
    catalog = load_catalog()
    rows = []
    # Interleave difficulty before seeing outcomes; required composite appears second.
    groups = [list(EASY), list(HARD), list(MEDIUM)]
    while any(groups):
        for group in groups:
            if not group:
                continue
            task = group.pop(0)
            index = len(rows)
            assert task in catalog['tasks'], task
            rows.append({'id':f't{index+1:02d}', 'task':task,
                'difficulty_prior':'easy' if task in EASY else 'medium' if task in MEDIUM else 'hard',
                'difficulty_basis':'Authored operation complexity; not calibrated model difficulty',
                'config':{'robot':'GR1' if task in GR1_TASKS else 'PandaOmron', 'task':'robocasa/'+task,
                    'seed':0, 'split':None, 'env_kwargs':{'layout_ids':[2,3,4,5,6][index%5],
                    'style_ids':[1,3,5][index%3], 'camera_widths':640, 'camera_heights':480, 'camera_depths':True}}})
    assert len(rows)==50 and len({r['task'] for r in rows})==50
    manifest = {'version':VERSION, 'created_at':now(), 'tasks':rows,
        'registry_sha256':catalog['registry_sha256'], 'upstream_commit':catalog['checkout_commit'],
        'attempts_per_series':2, 'active_seconds_per_attempt':1800, 'max_control_steps':6000,
        'models':['gpt-6-astra','Qwen/Qwen3.8-Max-VL-Thinking'], 'effort':'high',
        'selection_policy':'Outcome-blind fixed 20 easy / 15 medium / 15 hard unique task names, five layouts, three styles, two robots. GR1 restricted to local fixture manipulation because its base is fixed. Preflight failures remain visible; no silent replacement.',
        'interpretation':'Descriptive adaptation across two matched resets, retaining conversation, code and notes. No cross-task code transfer, no privileged hints, no causal ICL claim, no certified benchmark score.',
        'harness':'write_execute composes journaled write+execute under one execution budget; unchanged observations not resent; sensor PNGs encoded JPEG quality 90 for both providers; per-phase timing; Qwen model-generated compaction on tokens or request bytes.',
        'verifier':'Unmodified upstream whole-task predicate sustained 10 steps; initially-successful or invalid scenes excluded as infrastructure/precondition issues.'}
    manifest['sha256'] = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    emit(path,manifest)
    return manifest


class EfficientRunner(WaitingAttemptRunner):
    def apply(self, decision, **kwargs):
        if not isinstance(decision,dict) or decision.get('operation')!='write_execute':
            return super().apply(decision,**kwargs)
        if not str(decision.get('path','')).endswith('.py'):
            raise ValueError('write_execute requires a Python file')
        began=time.monotonic()
        saved=super().apply({**decision,'operation':'write'},**kwargs)
        if saved.get('returncode') or saved.get('error'):
            return saved
        remaining=kwargs.get('timeout',60)-(time.monotonic()-began)
        if remaining<=0:
            return {**saved,'error':'combined_operation_deadline','message':'File saved, execution did not start.'}
        return super().apply({**decision,'operation':'execute','content':''},**{**kwargs,'timeout':remaining})


class EfficientAstra(LearningModel):
    def ask(self,text,**kwargs):
        kwargs['schema']=SCHEMA
        return super().ask(text,**kwargs)


class EfficientQwen(LearningQwen):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.metadata['protocol_recovery']='final-json-retry-v1: one corrective request within original budget; never execute reasoning_content'
        self.metadata['context_policy']='Retain conversation across the two configured attempts; model-generated compaction near 800000 tokens or above 8000000 serialized history bytes. Archive pre-compaction context; retain code and notes separately.'
    def ask(self,text,**kwargs):
        from pairing.run_qwen_trials import ModelProtocolError
        began=time.monotonic()
        limit=kwargs.get('timeout',180)
        size=len(json.dumps(self.messages).encode())
        if size > 8_000_000:
            self.compact(limit)
        kwargs.update(schema=SCHEMA,timeout=limit-(time.monotonic()-began))
        if kwargs['timeout']<=0:
            raise TimeoutError('Compaction used remaining budget')
        try:
            return super().ask(text,**kwargs)
        except ModelProtocolError as exc:
            with (self.directory/'protocol-recovery.jsonl').open('a') as stream:
                stream.write(json.dumps({'event':'invalid_final_decision','call':self.calls,'error':str(exc),'executed':False,'epoch':time.time()})+'\n')
            remaining=limit-(time.monotonic()-began)
            if remaining<=0:raise TimeoutError('No budget remains for protocol correction') from None
            original=json.loads(text)
            correction=json.dumps({'observation':{'control_step':original.get('observation',{}).get('control_step',0),'unchanged_since_previous_observation':True},
                'protocol_error':str(exc),
                'notice':'Your last final answer was not a valid structured decision. No proposed operation executed, and physics did not advance. Return one valid JSON decision in FINAL content using the supplied schema. Do not put the final decision only in reasoning. The prior sensor images and state remain current.'})
            kwargs.update(timeout=remaining,images=())
            return super().ask(correction,**kwargs)


def compact_observation_feedback(feedback, observation):
    """Keep operation receipts; sensors are delivered through model_input only."""
    if isinstance(feedback,dict) and 'sensors' in feedback:
        return {k:v for k,v in feedback.items() if k not in observation and k not in {
            'sensors','images_png_base64','depth_npz_base64','depth_arrays'}}
    return feedback


def drive(runner,model,directory,*,execution_seconds=60,episode_steps=6000,script_steps=50):
    """Paired protocol: preserve all state after motion, omit unchanged repeats."""
    from PIL import Image
    runtime=runner.runtime
    feedback=runner.start_task(0)
    observation=feedback.pop('observation')
    feedback['execution_limits']={'max_control_steps':script_steps,'wall_seconds':execution_seconds,
        'scope':f'Each script execution; separate from the {episode_steps}-step, {runtime.timeout}-second episode budget.'}
    first=True
    previous_step=None
    explicit_observation=False
    turn=0
    stop='precondition_invalid'
    with (directory/'model-transcript.jsonl').open('x') as transcript, (directory/'timing.jsonl').open('x') as timings:
        def log(event,**data):
            timings.write(json.dumps({'event':event,'epoch':time.time(),'monotonic':time.monotonic(),'turn':turn,**data})+'\n');timings.flush()
        if feedback['precondition_valid']:
            while runner.task_active:
                require_disk_headroom(directory,5)
                remaining=runtime.timeout-(time.monotonic()-runtime.started)
                if remaining<=0:
                    stop='task_deadline';break
                if observation['control_step']>=episode_steps:
                    stop='control_step_budget';break
                begin=time.monotonic()
                step=observation['control_step']
                compact_feedback=compact_observation_feedback(feedback,observation)
                if first or step!=previous_step or explicit_observation:
                    prompt,images=model_input(compact_feedback,observation,directory/'model-images',turn)
                    converted=[]
                    for path in images:
                        jpg=path.with_suffix('.jpg')
                        with Image.open(path) as im: im.convert('RGB').save(jpg,quality=90)
                        path.unlink() # Lossless source also in recorded simulator evidence; model receives JPEG.
                        converted.append(jpg)
                    images=converted
                    value=json.loads(prompt)
                    # Observe/wait replies otherwise duplicate the complete sensor object.
                    if isinstance(value['feedback'],dict) and 'sensors' in value['feedback']:
                        value['feedback']={k:v for k,v in value['feedback'].items() if k not in value['observation']}
                    prompt=json.dumps(value,separators=(',',':'))
                else:
                    prompt=json.dumps({'feedback':compact_feedback,'observation':{'control_step':step,'unchanged_since_previous_observation':True},
                        'notice':'Physics has not advanced; the previous sensor state and camera images remain current. File operations do not move the robot.'},separators=(',',':'))
                    images=[]
                previous_step=step
                value=json.loads(prompt)
                value['episode_budget']={
                    'active_work_seconds_remaining':round(max(0.,runtime.timeout-(time.monotonic()-runtime.started)),3),
                    'control_steps_remaining':max(0,episode_steps-step),
                    'passive_wait_steps_remaining':max(0,1200-getattr(runtime,'passive_wait_steps',0)),
                    'scope':f'Remaining episode budget; passive-wait execution time is excluded from active work. Each script remains limited to {script_steps} control steps and {execution_seconds} seconds.'}
                prompt=json.dumps(value,separators=(',',':'))
                log('input_ready',seconds=time.monotonic()-begin,prompt_bytes=len(prompt.encode()),images=len(images),image_bytes=sum(p.stat().st_size for p in images),control_step=step)
                transcript.write(json.dumps({'event':'request','turn':turn,'prompt':prompt,'images':[p.name for p in images]})+'\n');transcript.flush()
                begin=time.monotonic()
                log('model_start',control_step=step)
                try: decision=model.ask(prompt,images=images,timeout=max(.1,runtime.timeout-(time.monotonic()-runtime.started)),as_tool_output=not first)
                except TimeoutError:
                    log('model_timeout',seconds=time.monotonic()-begin);stop='model_deadline';break
                except Exception as exc:
                    log('model_error',seconds=time.monotonic()-begin,error_type=type(exc).__name__,error=str(exc)[:500])
                    raise
                log('model_end',seconds=time.monotonic()-begin,operation=decision.get('operation'),control_step=step)
                first=False
                thinking_trace=getattr(model,'last_provider_thinking_trace',None)
                if thinking_trace is not None:
                    transcript.write(json.dumps({'event':'provider_thinking','turn':turn,'trace':thinking_trace},ensure_ascii=False)+'\n');transcript.flush()
                transcript.write(json.dumps({'event':'decision','turn':turn,'decision':decision})+'\n');transcript.flush()
                begin=time.monotonic()
                try: feedback=runner.apply(decision,timeout=execution_seconds,max_control_steps=script_steps)
                except ValueError as exc: feedback={'operation':decision.get('operation'),'protocol_error':str(exc)}
                except TimeoutError:
                    stop='task_deadline';break
                log('operation_end',seconds=time.monotonic()-begin,operation=decision.get('operation'),accepted_control_steps=feedback.get('accepted_control_steps'),error=feedback.get('error'))
                explicit_observation=decision.get('operation')=='observe'
                if not runner.task_active:return feedback
                # Keep prior observation across file-only operations; no need to render again.
                if decision.get('operation') not in ('write','read'):
                    begin=time.monotonic();observation=runtime.observe();log('observe_end',seconds=time.monotonic()-begin,control_step=observation['control_step'])
                if observation.get('simulator_phase')!='active': raise RuntimeError('Simulator faulted')
                if observation.get('task_success'):
                    stop='sustained_success';break
                turn+=1
        return runner.finish_task(reason=stop)


def prompt_for(task):
    from pairing.run_toaster_learning import SCOOP
    prompt=instructions(task,timeout=1800,trials=2).replace('\n'+SCOOP,'')
    prompt+='\nThe isolated execution environment has Python 3.11, NumPy, SciPy, and Pillow installed. SciPy Rotation and Slerp are available for numerical rotation calculations. These standard numerical/image libraries are permitted; the initial workspace contains no robot motion or grasp library. Write your own bounded robot control functions using the supplied observations and native actions.\n'
    return prompt+'\nYou may use operation="write_execute" with a relative .py path and source in content to save then run it in ONE turn. It uses the same 50-step, 60-second execution limits. A failed save never executes. Prefer this for a prepared bounded probe or movement to avoid a redundant model turn. Use Python to calculate transforms and numeric geometry instead of doing long arithmetic in prose. Keep public rationale brief (1–3 sentences). After a bounded probe, use its measurements to choose the next action. No task solution library is supplied. If the observation says unchanged, use the previously supplied images and state. Sensor images are JPEG quality 90. Success still requires the full official task predicate.\n'


def series(row,provider,parent,image):
    hand_contract=HAND_ACTION_CONTRACT if row['config']['robot']=='GR1' else 'upstream-standard'
    region_file=preflight_directory(row)/'counter-region-order.json'
    region_reference=json.loads(region_file.read_text()) if region_file.exists() else None
    reset_protocol='counter-region-order-v1' if region_reference is not None else 'seeded-rng-plus-hashseed-v2'
    repair_label=' · native hand repair' if row.get('repair_kind')=='native_hand_contract' else ' · reset repair' if row.get('repair_kind')=='counter_region_order' else ' · protocol repair' if row.get('repair_request_id') else ''
    directory,record=create('jobs',f'{provider}{repair_label} · {row["id"]} {row["task"]} · 2×30m ICL', 'cohort50-series')
    record.update(campaign_id=parent,task=row['task'],cohort_id=row['id'],provider=provider,benchmark_score=None,status='running',hand_action_contract=hand_contract)
    for key in ('repair_request_id','retry_of_series','repair_reason','repair_kind'):
        if key in row:record[key]=row[key]
    emit(directory/'plan.json',{'cohort':row,'version':VERSION,'transport_optimizations':TRANSPORT_VERSION,'sensor_documentation':SENSOR_DOCUMENTATION,'environment_documentation':ENVIRONMENT_DOCUMENTATION,'execution_limit_documentation':EXECUTION_LIMIT_DOCUMENTATION,'observation_feedback_version':OBSERVATION_FEEDBACK_VERSION,'budget_feedback_version':BUDGET_FEEDBACK_VERSION,'hand_action_contract':hand_contract,'reset_protocol':reset_protocol,'counter_region_reference_sha256':hashlib.sha256(region_file.read_bytes()).hexdigest() if region_reference is not None else None,'prompt':prompt_for(row['task']),'code_image':image,'attempts':2,'model':'claude-opus-5-5' if provider=='opus55' else 'gpt-6-astra' if provider=='astra' else 'Qwen/Qwen3.8-Max-VL-Thinking','effort':'high','provider_transport':'anthropic-messages-direct-with-automatic-prompt-caching-and-deterministic-local-rollover-v1' if provider=='opus55' else provider});record['artifacts']['plan']='plan.json';save(directory,record)
    with ZipFile(directory/'source.zip','w') as archive:
        for folder in ('pairing','roboswe','robot_session'):
            for path in Path(folder).rglob('*.py'):archive.write(path)
    record['artifacts']['source']='source.zip';save(directory,record)
    reports=[]
    fingerprint=None
    try:
        if provider=='astra':
            model_context=EfficientAstra(model='gpt-6-astra',effort='high',instructions=prompt_for(row['task']),context_window=1000000,compact_at=800000,reasoning_summary='detailed',verbosity='medium',persistent=True,diagnostics=directory/'model-events.jsonl',summary_log=directory/'reasoning-summaries.jsonl')
        elif provider=='qwen':
            model_context=nullcontext(EfficientQwen(prompt_for(row['task']),directory))
        elif provider=='opus55':
            from pairing.anthropic_model import AnthropicLearningModel
            ledger=os.environ.get('ROBOSWE_ANTHROPIC_BUDGET_PATH')
            if not ledger:raise RuntimeError('Opus campaign requires a shared spend ledger')
            model_context=AnthropicLearningModel(prompt_for(row['task']),directory,context_policy="legacy",
                effort='high',context_window=1000000,compact_at=760000,cache_ttl='5m',
                ledger_path=ledger,spend_limit_usd=float(os.environ.get('ROBOSWE_ANTHROPIC_MAX_USD','195')))
        else:raise ValueError('Unknown cohort provider')
        with CodeSandbox(image,lifetime_seconds=6600) as sandbox, model_context as model:
            if provider=='astra' and model._rpc('account/rateLimits/read',{}).get('accountId')!=EXPECTED_ACCOUNT:
                raise RuntimeError('Active Astra account differs from the authorized account')
            emit(directory/'model.json',model.metadata)
            for number in (1,2):
                if provider=='astra' and model._rpc('account/rateLimits/read',{}).get('accountId')!=EXPECTED_ACCOUNT:
                    raise RuntimeError('Active Astra account differs from the authorized account')
                attempt,job=create('jobs',f'{provider}{repair_label} · {row["id"]} {row["task"]} · ICL {number}/2','cohort50-attempt')
                job.update(campaign_id=record['id'],task=row['task'],provider=provider,cohort_id=row['id'],attempt_number=number,status='starting',verifier_status='observed_only',benchmark_score=None)
                record.update(current_job=job['id'],current_attempt=number);save(directory,record);save(attempt,job)
                runtime=journal=None
                spec=serial_spec(row['task'],timeout=1800,max_steps=6000);spec['learning_trials']=2;spec['cohort_protocol']=VERSION
                spec['transport_optimizations']=TRANSPORT_VERSION
                spec['sensor_documentation']=SENSOR_DOCUMENTATION
                spec['hand_action_contract']=hand_contract
                spec['environment_documentation']=ENVIRONMENT_DOCUMENTATION
                spec['execution_limit_documentation']=EXECUTION_LIMIT_DOCUMENTATION
                spec['observation_feedback_version']=OBSERVATION_FEEDBACK_VERSION
                spec['budget_feedback_version']=BUDGET_FEEDBACK_VERSION
                spec['reset_protocol']=reset_protocol
                if region_reference is not None:spec['counter_region_reference']=region_reference
                spec['sha256']=hashlib.sha256(json.dumps({k:v for k,v in spec.items() if k!='sha256'},sort_keys=True).encode()).hexdigest()
                if provider=='qwen':model.directory=attempt;save_context(attempt/'context-start.json.gz',model.messages)
                elif provider=='opus55':model.save_context(attempt/'context-start.json.gz')
                try:
                    runtime=FreshEpisode(row['config'],spec,attempt,number,reports,worker=suite_worker)
                    actual=runtime.initial['fingerprint']
                    if fingerprint is None:fingerprint=actual
                    if actual!=fingerprint:raise RuntimeError('ICL reset fingerprint mismatch')
                    expected=json.loads((preflight_directory(row)/'receipt.json').read_text())['fingerprint']
                    if actual!=expected:raise RuntimeError('Scene differs from frozen preflight')
                    emit(attempt/'continuity.json',{'initial_fingerprint':actual,'thread_id':getattr(model,'thread_id',None),'workspace_keeper':sandbox.keeper,'retained':number>1,'library_start':sandbox.archive(attempt/'library-start.zip')})
                    journal=Journal(attempt/'submissions.jsonl');job['status']='running';save(attempt,job)
                    report=drive(EfficientRunner(runtime,sandbox,journal),model,attempt)
                    report.update(attempt_number=number,provider=provider,cohort_id=row['id'],retained_context=number>1,retained_library=number>1)
                    report['library']=sandbox.archive(attempt/'library.zip')
                    emit(attempt/'report.json',report)
                    with ZipFile(attempt/'library.zip') as z:
                        if 'NOTES.md' in z.namelist():(attempt/'NOTES.md').write_bytes(z.read('NOTES.md'));job['artifacts']['notes']='NOTES.md'
                    reports.append({k:report[k] for k in ('attempt_number','observed_success','termination_reason','control_steps','elapsed_seconds')})
                    job['status']='observed_pass' if report['observed_success'] else 'observed_fail'
                    if report['termination_reason']=='precondition_invalid':job['status']='invalid_precondition'
                    if provider in ('astra','opus55'):write_human_outputs(attempt)
                    job['artifacts'].update(report='report.json',library='library.zip',timing='timing.jsonl',transcript='model-transcript.jsonl')
                except BaseException as exc:
                    job['status']='budget_limit_reached' if type(exc).__name__=='BudgetLimitReached' else 'infrastructure_error';(attempt/'stderr.txt').write_text(traceback.format_exc());job['artifacts']['stderr']='stderr.txt'
                    if not (attempt/'library.zip').exists() and not sandbox.active:
                        try:sandbox.archive(attempt/'library.zip');job['artifacts']['library']='library.zip'
                        except Exception:pass
                    raise
                finally:
                    if provider=='qwen':save_context(attempt/'context-end.json.gz',model.messages)
                    elif provider=='opus55':model.save_context(attempt/'context-end.json.gz')
                    if journal:journal.close()
                    try:
                        if runtime:
                            receipt=runtime.close();publish_video(Path(receipt['recording_dir']),attempt,job,phase='original')
                    finally:
                        job['finished_at']=now();save(attempt,job)
                        try:
                            from pairing.cohort50_evidence import finalize
                            finalize(attempt,job)
                        except Exception as exc:
                            job['artifact_errors'].append({'stage':'evidence_log_registration','error':str(exc)});save(attempt,job)
                    emit(directory/'report.json',{'trials':reports,'benchmark_score':None})
            record['status']='completed_unscored'
    except BaseException as exc:
        record['status']='budget_limit_reached' if type(exc).__name__=='BudgetLimitReached' else 'infrastructure_error';(directory/'stderr.txt').write_text(traceback.format_exc());record['artifacts']['stderr']='stderr.txt';raise
    finally:
        record['finished_at']=now()
        if (directory/'report.json').exists():record['artifacts']['report']='report.json'
        for key,name in (
            ('model','model.json'),
            ('model_usage','model-usage.jsonl'),
            ('provider_responses','provider-responses.jsonl'),
            ('provider_thinking','provider-thinking.jsonl'),
        ):
            if (directory/name).exists():record['artifacts'][key]=name
        save(directory,record)


def preflight(row):
    directory=preflight_directory(row);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'receipt.json').exists():
        receipt=json.loads((directory/'receipt.json').read_text())
        if receipt['config']!=row['config']:raise ValueError('Preflight configuration changed')
        return
    runtime=None
    try:
        spec=serial_spec(row['task'],timeout=240,max_steps=6000);spec['cohort_protocol']=VERSION
        spec['capture_counter_regions']=True
        spec['sha256']=hashlib.sha256(json.dumps({k:v for k,v in spec.items() if k!='sha256'},sort_keys=True).encode()).hexdigest()
        runtime=FreshEpisode(row['config'],spec,directory,0,[],worker=suite_worker)
        result=runtime.start_task(0);obs=result['observation']
        if not result['precondition_valid']:raise RuntimeError('Task already successful at reset')
        action=obs['whole_body_ik']['current_pose_hold_action'] if row['config']['robot']=='GR1' else [0.]*obs['action_schema']['shape'][0]
        began=time.monotonic()
        for _ in range(10):runtime.bridge({'operation':'step','action':action})
        elapsed=time.monotonic()-began
        if elapsed>9:raise RuntimeError(f'10-step throughput {elapsed:.1f}s fails 50-step/60s margin')
        emit(directory/'receipt.json',{'status':'passed','fingerprint':runtime.initial['fingerprint'],'10_steps_seconds':elapsed,'config':row['config'],'instruction':result['instruction']})
    except BaseException:
        (directory/'error.txt').write_text(traceback.format_exc());raise
    finally:
        if runtime:runtime.close()


def supervise(hours, resume=False):
    from roboswe.campaign import reference_lease
    manifest=freeze()
    inherited=[]
    completed=[]
    if resume:
        saved=json.loads((ROOT/'active.json').read_text())
        directory=Path(saved['directory']);record=json.loads((directory/'job.json').read_text())
        deadline=saved['deadline']
        progress=json.loads((directory/'progress.json').read_text())
        completed=progress['completed'];inherited=progress['active']
    else:
        directory,record=create('jobs','Astra + Qwen · frozen 50-task overnight ICL','cohort50-campaign')
        deadline=time.time()+hours*3600
    record.update(status='running',benchmark_score=None);save(directory,record)
    emit(ROOT/'active.json',{'job_id':record['id'],'directory':str(directory),'pid':os.getpid(),'deadline':deadline})
    image=subprocess.check_output(['docker','image','inspect','roboswe-code:local','--format','{{.Id}}'],text=True).strip()
    (directory/'cohort.json').write_bytes((ROOT/'cohort.json').read_bytes());record['artifacts']['cohort']='cohort.json'
    source_name='supervisor-resume-source.zip' if resume else 'source.zip'
    with ZipFile(directory/source_name,'w') as z:
        for folder in ('pairing','roboswe','robot_session'):
            for path in Path(folder).rglob('*.py'):z.write(path)
    record['artifacts']['source']='source.zip'
    if resume:record['artifacts']['supervisor_resume_source']=source_name
    save(directory,record)
    try:
        with reference_lease({'jobs_dir':'jobs','execution':{'memory_mb':6144,'cpu':4,'min_free_disk_gb':5}}):
            # Adopt only already-running series from this campaign; never relaunch them.
            for item in inherited:
                while True:
                    command=subprocess.run(['ps','-p',str(item['pid']),'-o','command='],capture_output=True,text=True).stdout
                    if f'--parent {record["id"]}' not in command or 'pairing.cohort50 series' not in command:break
                    time.sleep(10)
                completed.append({**item,'status':'finished_after_supervisor_resume','exit_code':None})
            # Freeze task membership independently of queue order: exercise both
            # embodiments early even when the overnight window cannot cover all 50.
            priorities=['t01','t02','t10','t03','t04','t16','t05','t06','t07','t08','t09']
            rows=sorted(manifest['tasks'],key=lambda row:priorities.index(row['id']) if row['id'] in priorities else len(priorities)+int(row['id'][1:]))
            emit(directory/'execution-order.json',{'task_ids':[row['id'] for row in rows],'reason':'Bring GR1 forward for overnight embodiment coverage; frozen task membership and scenes unchanged.'})
            record['artifacts']['execution_order']='execution-order.json';save(directory,record)
            for row in rows:
                if row['id'] in {item['id'] for item in completed}:continue
                if time.time()>deadline-4500:break
                require_disk_headroom('jobs',7)
                control=json.loads((ROOT/'control.json').read_text()) if (ROOT/'control.json').exists() else {}
                if control.get('stop') or control.get('pause_launches'):break
                log=(ROOT/(row['id']+'-preflight.log')).open('a')
                proc=subprocess.Popen([sys.executable,'-u','-m','pairing.cohort50','preflight','--task-id',row['id']],stdout=log,stderr=subprocess.STDOUT)
                try:code=proc.wait(timeout=300)
                except subprocess.TimeoutExpired:proc.terminate();code=proc.wait(timeout=30)
                log.close()
                if code:
                    completed.append({'id':row['id'],'status':'preflight_failed'});emit(directory/'progress.json',{'completed':completed,'active':[]});continue
                active=[]
                for provider in ('astra','qwen'):
                    stream=(ROOT/f'{row["id"]}-{provider}.log').open('a')
                    proc=subprocess.Popen([sys.executable,'-u','-m','pairing.cohort50','series','--task-id',row['id'],'--provider',provider,'--parent',record['id'],'--code-image',image],stdout=stream,stderr=subprocess.STDOUT)
                    active.append((provider,proc,stream))
                emit(directory/'progress.json',{'completed':completed,'active':[{'id':row['id'],'provider':p,'pid':proc.pid} for p,proc,_ in active]})
                for provider,proc,stream in active:
                    code=proc.wait();stream.close();completed.append({'id':row['id'],'provider':provider,'exit_code':code})
                emit(directory/'progress.json',{'completed':completed,'active':[]})
        record['status']='window_complete'
    except BaseException:
        record['status']='infrastructure_error';(directory/'stderr.txt').write_text(traceback.format_exc());record['artifacts']['stderr']='stderr.txt';raise
    finally:
        record['artifacts']['progress']='progress.json';record['finished_at']=now();save(directory,record)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','preflight','series','supervise']);p.add_argument('--task-id');p.add_argument('--provider',choices=['astra','qwen','opus55']);p.add_argument('--parent');p.add_argument('--code-image');p.add_argument('--hours',type=float,default=10);p.add_argument('--resume',action='store_true');p.add_argument('--repair-request-id')
    args=p.parse_args();manifest=freeze()
    row=next((r for r in manifest['tasks'] if r['id']==args.task_id),None)
    if row is not None:row=effective_row(row)
    if args.repair_request_id:
        import re
        if not re.fullmatch('[a-z0-9-]+',args.repair_request_id):raise ValueError('Invalid repair request ID')
        request=json.loads((ROOT/'repair-requests'/f'{args.repair_request_id}.request.json').read_text())
        if request['task_id']!=args.task_id or request['provider']!=args.provider:raise ValueError('Repair request does not match series')
        row={**row,'repair_request_id':args.repair_request_id,'retry_of_series':request['source_series_id'],'repair_reason':request['reason'],'repair_kind':request.get('repair_kind','provider_protocol')}
    if args.mode=='preflight':preflight(row)
    elif args.mode=='series':series(row,args.provider,args.parent,args.code_image)
    elif args.mode=='supervise':supervise(args.hours,args.resume)
    else:print(json.dumps({'path':str(ROOT/'cohort.json'),'count':len(manifest['tasks']),'sha256':manifest['sha256']}))


if __name__=='__main__':main()
