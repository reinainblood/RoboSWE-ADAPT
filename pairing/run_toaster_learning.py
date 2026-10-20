"""Two isolated five-episode toaster learning sequences, with retained code/context.

Fresh simulator process per episode; one model conversation and persistent
workspace keeper per sequence. Diagnostic stage telemetry never enters observations.
"""
import argparse
import base64
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from roboswe.campaign import drive_task, publish_video, reference_lease
from roboswe.codex_login import INSTRUCTIONS
from roboswe.execution.sandbox import CodeSandbox
from roboswe.jobs import create, save, now
from roboswe.library.attempts import AttemptRunner, Journal
from roboswe.library.serial import serial_spec
from pairing.run_hand_microwave import CoachedCodexLogin
from pairing.simulation_wait import WAIT_SCHEMA, WAIT_INSTRUCTIONS, WaitingAttemptRunner, passive_wait

class LearningModel(CoachedCodexLogin):
    def ask(self, text, **kwargs):
        kwargs.setdefault('schema', WAIT_SCHEMA)
        return super().ask(text, **kwargs)


TASKS = ('ToastBagel', 'GetToastedBread')
EXPECTED_ACCOUNT = __import__('os').environ.get('ROBOSWE_LEGACY_ACCOUNT_ID', 'UNCONFIGURED')
SCOOP = """Human coaching from Kirsten: this is a SCOOP, not a pinch. Gently position fingers 2–5 flat on the plate, where geometry permits, so they form a little ramp under the bagel's edge. Establish actual supporting contact before attempting to secure it. Guide the bagel onto those fingers with the thumb moving down and inward, then support/curl the fingers upward. Do not lift the arm while the fingers are merely near the bagel. Inspect and adjust incrementally; a commanded hand shape or unexplained resistance is not evidence of bagel contact. Adapt this demonstrated motion to the hand's coupled joints. The photograph is a real human demonstration, not a simulator camera or exact robot pose."""


def instructions(task, *, timeout=720, trials=5):
    value = INSTRUCTIONS.replace('one; execute(path)', 'one (or NOTES.md); execute(path)')
    value = value.replace('replaces a relative .py file;', 'replaces a relative .py file or NOTES.md;')
    value = value.replace('message is a brief explanation, not\nan execution claim.', 'message is a public action rationale of 2–5 sentences, not an execution claim. Describe your observations, proposed contact/motion, expected effect, and evidence that would change your approach. Distinguish observation from hypothesis.')
    value += """\nThis is a five-attempt learning sequence for ONE task. Each episode has the same 720-second budget and resets the entire robot/appliance/object scene to the same initial seed. Your conversation, any compaction summary, all Python files, and NOTES.md persist across episodes. A reset does not erase your previous learning. Use feedback from prior episodes to revise rather than restarting your library. Maintain NOTES.md as you work: attempted approaches, verified observations, failed assumptions, useful functions, and next tests. NOTES.md is read/write data, never executable. Save notes incrementally because the timeout is enforced. Your starting library is empty. You receive no previous agent's code or privileged simulator state. Success is checked by the official task predicate. Continue toward the whole stated task, not just its first action.\n"""
    if task == 'ToastBagel':
        value += '\n' + SCOOP
    value += WAIT_INSTRUCTIONS
    if trials != 5:
        value = value.replace('five-attempt', f'{trials}-attempt')
    return value.replace('720-second', f'{timeout}-second')


def stage_state(session, initial):
    """Read-only human diagnostics, not solver observations or a new task grader."""
    raw, model, data = session.raw, session.raw.sim.model._model, session.raw.sim.data._data
    name = 'bagel' if 'bagel' in raw.objects else 'obj'
    obj = raw.objects[name]
    body_id = model.body(obj.root_body).id
    position = data.xpos[body_id]
    touching = set()
    for c in data.contact[:data.ncon]:
        a, b = model.geom(c.geom1).name, model.geom(c.geom2).name
        # Exact object collision geoms avoid confusing bagel_container (plate).
        if a in obj.contact_geoms:
            touching.add(b)
        elif b in obj.contact_geoms:
            touching.add(a)
    hand = sorted(x for x in touching if x.startswith('gripper'))
    plate = any('plate' in x or 'container' in x for x in touching)
    row = {'step': session.steps, 'object_position': position.tolist(),
           'height_gain_m': float(position[2] - initial[2]), 'hand_contacts': hand,
           'thumb_and_finger_contact': any('thumb' in x for x in hand) and any('thumb' not in x for x in hand),
           'plate_contact': plate, 'upstream_success': bool(session.info.get('success'))}
    row['supported_lift_candidate'] = bool(hand and not plate and row['height_gain_m'] >= .025)
    if hasattr(raw, 'toaster'):
        row['toaster_state'] = raw.toaster.get_state(raw)
    if hasattr(raw, 'toaster_oven'):
        row['toaster_oven_state'] = raw.toaster_oven.get_state(raw)
    return row


def simulator_worker(pipe, config, spec, directory, number, previous):
    # Import and build MuJoCo only in a fresh spawn process for a deterministic reset.
    from pairing.run_hand_microwave import HandSession, HandSerialRuntime
    from pairing.panda_inspire import register_panda_inspire_hand
    register_panda_inspire_hand()
    directory = Path(directory)
    session = None
    evidence = metrics = None
    try:
        session = HandSession(config)
        session.recording_root = directory / 'recordings'
        session.start_recording({'recording_id': 'episode', 'width': 640, 'height': 480})
        evidence = (directory / 'actions.jsonl').open('x')
        metrics = (directory / 'stage-telemetry.jsonl').open('x')
        runtime = HandSerialRuntime(session, spec, timeout=spec['task_timeout_seconds'], max_steps=6000, evidence=evidence)
        obj_name = 'bagel' if 'bagel' in session.raw.objects else 'obj'
        initial = session.raw.sim.data.body_xpos[session.raw.obj_body_id[obj_name]].copy()
        stage_error = None
        last_action = None
        def observe_stage():
            nonlocal stage_error
            if stage_error is not None:
                return
            try:
                metrics.write(json.dumps(stage_state(session, initial)) + '\n'); metrics.flush()
            except Exception:
                stage_error = traceback.format_exc()
                (directory / 'stage-telemetry-error.txt').write_text(stage_error)
        observe_stage()
        pipe.send({'ok': runtime.metadata(), 'fingerprint': session.initial})
        while True:
            request = pipe.recv()
            op = request['operation']
            try:
                if op == 'start_task':
                    result = runtime.start_task(0)
                    result['passive_wait_api'] = {'operation':'wait','content':'integer steps, 1–600','path':'','total_wait_limit':1200,'clock':'physics advances; passive wait host time excluded from agent-work budget'}
                    result.update(trial_number=number, total_trials=spec.get('learning_trials', 5), reset_notice='Fresh deterministic simulator episode; your model conversation, code library, and NOTES.md are retained.', previous_attempts=previous)
                elif op == 'observe': result = runtime.observe()
                elif op == 'bridge':
                    result = runtime.bridge(request['message'])
                    if request['message']['operation'] == 'step':
                        last_action = list(request['message']['action'])
                        observe_stage()
                elif op == 'wait':
                    result = passive_wait(runtime, last_action, request['steps'], on_step=observe_stage)
                elif op == 'finish_task':
                    result = runtime.finish_task(code_errors=request['code_errors'])
                    result.update(passive_wait_control_steps=getattr(runtime, 'passive_wait_steps', 0), passive_wait_wall_seconds=getattr(runtime, 'passive_wait_wall_seconds', 0.))
                    result['wall_elapsed_seconds'] = result['elapsed_seconds'] + result['passive_wait_wall_seconds']
                    result['budget_policy'] = f'passive_wait_v1: {runtime.timeout}s agent-work plus separately recorded bounded waiting'
                elif op == 'digest': result = runtime.action_digest.hexdigest()
                elif op == 'close':
                    session.stop_recording({'recording_id': 'episode'})
                    (directory / 'final-snapshot.zip').write_bytes(session.snapshot())
                    result = {'recording_dir': str(session.recording.directory), 'stage_error': stage_error}
                    pipe.send({'ok': result})
                    break
                else: raise ValueError('Unknown trusted coordinator operation')
                pipe.send({'ok': result})
            except Exception:
                pipe.send({'error': traceback.format_exc(), 'error_type': sys.exc_info()[0].__name__})
    except BaseException:
        try: pipe.send({'error': traceback.format_exc()})
        except (BrokenPipeError, EOFError): pass
    finally:
        if evidence: evidence.close()
        if metrics: metrics.close()
        if session:
            if session.recording and session.recording.active:
                session.stop_recording({'recording_id': 'episode'})
            session.close()
        pipe.close()


class FreshEpisode:
    def __init__(self, config, spec, directory, number, previous, *, worker=simulator_worker):
        context = mp.get_context('spawn')
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=worker, args=(child, config, spec, str(directory), number, previous))
        # RoboCasa's seeded placement logic can still traverse hash-backed
        # collections while building a fresh interpreter.  A spawned process
        # otherwise receives a new randomized Python hash seed, which can swap
        # equally valid placement regions even when the task seed is unchanged.
        # Pin only the simulator child's interpreter seed, then restore the
        # coordinator environment immediately after spawn.
        previous_hash_seed = os.environ.get('PYTHONHASHSEED')
        os.environ['PYTHONHASHSEED'] = str(config.get('seed', 0))
        try:
            self.process.start()
        finally:
            if previous_hash_seed is None:
                os.environ.pop('PYTHONHASHSEED', None)
            else:
                os.environ['PYTHONHASHSEED'] = previous_hash_seed
        child.close()
        self.timeout, self.started = spec['task_timeout_seconds'], None
        self.passive_wait_steps = 0
        self.has_stepped = False
        self.initial = self._receive(180)
        self.action_digest = SimpleNamespace(hexdigest=lambda: self.call('digest'))
    def _receive(self, timeout=90):
        if not self.pipe.poll(timeout): raise TimeoutError('Trusted simulator did not respond')
        reply = self.pipe.recv()
        if 'error' in reply:
            if reply.get('error_type') == 'ValueError': raise ValueError(reply['error'])
            if reply.get('error_type') == 'TimeoutError': raise TimeoutError(reply['error'])
            raise RuntimeError(reply['error'])
        return reply
    def call(self, operation, **args):
        self.pipe.send({'operation': operation, **args})
        return self._receive()['ok']
    def metadata(self): return self.initial['ok']
    def start_task(self, index):
        if index != 0: raise ValueError('One episode per simulator')
        self.started = time.monotonic()
        return self.call('start_task')
    def observe(self): return self.call('observe')
    def bridge(self, message):
        result = self.call('bridge', message=message)
        if message.get('operation') == 'step': self.has_stepped = True
        return result
    def wait(self, steps):
        if not self.has_stepped: raise ValueError('Execute a native step before waiting')
        began = time.monotonic()
        self.pipe.send({'operation': 'wait', 'steps': steps})
        result = self._receive(600)['ok']
        self.started += time.monotonic() - began
        self.passive_wait_steps = result['passive_wait']['episode_wait_steps']
        return result
    def finish_task(self, *, code_errors): return self.call('finish_task', code_errors=code_errors)
    def close(self):
        try: return self.call('close')
        finally:
            self.process.join(30)
            if self.process.is_alive(): self.process.terminate(); self.process.join(10)
            self.pipe.close()


def usage_summary(path):
    totals, contexts, compact, recovery = {}, [], 0, 0
    if path.exists():
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e['method'] == 'thread/tokenUsage/updated':
                u=e.get('token_usage') or {}
                totals=u.get('total', totals)
                if u.get('modelContextWindow'): contexts.append(u['modelContextWindow'])
            if e['method']=='item/completed' and e.get('item_type')=='contextCompaction': compact+=1
            if e['method']=='roboswe/recovery/forked': recovery+=1
    return {'cumulative_token_usage': totals, 'reported_context_windows': sorted(set(contexts)), 'compactions':compact, 'recovery_forks':recovery}


def write_human_outputs(directory):
    lines=['Model-authored action rationales; intended behavior is not proof of contact or success. Times use simulation playback.\n']
    step=0
    for line in (directory/'model-transcript.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['event']=='request': step=json.loads(row['prompt'])['observation']['control_step']
        elif row['event']=='decision':
            d=row['decision']; lines.append(f"Video {step/20:.2f}s · step {step} · {d['operation']} {d['path']}\n{d['message']}\n")
    lines.append('\nEXECUTION OUTPUT\n')
    for line in (directory/'submissions.jsonl').read_text().splitlines():
        e=json.loads(line)
        if e['kind']=='operation_result':
            r=e['data']['result']
            if r.get('output') or r.get('error'): lines.append(json.dumps(r,indent=2))
    (directory/'stdout.txt').write_text('\n'.join(lines))


def summarize_stages(directory):
    rows=[json.loads(l) for l in (directory/'stage-telemetry.jsonl').read_text().splitlines()]
    if not rows: return {'available':False}
    run=best=0
    for r in rows:
        run=run+1 if r['supported_lift_candidate'] else 0; best=max(best,run)
    return {'available':True,'privileged_human_diagnostics_only':True,'max_height_gain_m':max(r['height_gain_m'] for r in rows),
            'hand_contact_steps':sum(bool(r['hand_contacts']) for r in rows),
            'opposed_contact_steps':sum(r['thumb_and_finger_contact'] for r in rows),
            'longest_supported_lift_candidate_steps':best,
            'toaster_ever_on':any(any(v.get('turned_on') for v in r.get('toaster_state',{}).values()) for r in rows),
            'limitations':'Contact and height are diagnostic indicators, not a validated grasp or safety verdict.'}


def run_series(task, image, campaign_id, photo, timeout=720, trials=5, *, config_override=None, worker=simulator_worker, absolute_deadline=None):
    embodiment = (config_override or {}).get('robot', 'PandaOmron')
    directory,record=create('jobs',f'Astra high · {embodiment} · {task} · retained learning {trials}× · {timeout//60} min','learning-sequence')
    record.update(campaign_id=campaign_id,task=task,benchmark_score=None)
    save(directory,record)
    config=json.loads(Path('jobs/20261004T151002Z-f2676811da/plan.json').read_text())['config']
    if config_override is not None:
        config=json.loads(json.dumps(config_override))
    config['task']='robocasa/'+task
    spec=serial_spec(task,timeout=timeout,max_steps=6000)
    spec['learning_trials'] = trials
    if absolute_deadline is not None:
        spec['absolute_deadline'] = absolute_deadline
    spec['sha256'] = hashlib.sha256(json.dumps({k:v for k,v in spec.items() if k != 'sha256'},sort_keys=True).encode()).hexdigest()
    execution={'timeout_seconds':timeout,'max_control_steps':6000,'max_attempt_steps':50,'attempt_timeout_seconds':60}
    text=instructions(task, timeout=timeout, trials=trials)
    plan={'task':task,'trials':trials,'model':'gpt-6-astra','effort':'high','context_window':1000000,'compact_at':800000,
          'budget_policy':'passive_wait_v1; bounded passive waiting excluded from agent-work wall time; all physics ticks counted and recorded','same_container_workspace':True,'same_conversation':True,'fresh_simulator_each_attempt':True,'seed':0,'config':config,'execution':execution,
          'reset_protocol':'seeded_rng_plus_pythonhashseed_v2',
          'coaching':SCOOP if task=='ToastBagel' else None,'instructions':text,'code_image':image,
          'assessment':'Within-sequence adaptation, descriptive only; context, notes, and code are retained together, so their individual causal contributions are not separated.'}
    (directory/'plan.json').write_text(json.dumps(plan,indent=2)); record['artifacts']['plan']='plan.json';save(directory,record)
    reports=[];first_fingerprint=None;turn=0
    try:
        with CodeSandbox(image,lifetime_seconds=trials*(timeout+900)+600) as sandbox, LearningModel(
            model='gpt-6-astra',effort='high',instructions=text,context_window=1000000,compact_at=800000,
            reasoning_summary='detailed',verbosity='high',persistent=True,
            diagnostics=directory/'model-events.jsonl',summary_log=directory/'reasoning-summaries.jsonl') as model:
            account=model._rpc('account/rateLimits/read',{})
            if account.get('accountId')!=EXPECTED_ACCOUNT: raise RuntimeError('Active CLI account differs from the user-selected RoboSWE account')
            (directory/'model.json').write_text(json.dumps(model.metadata,indent=2))
            record.update(model_thread_id=model.thread_id,workspace_keeper=sandbox.keeper,workspace_volume=sandbox.volume)
            save(directory,record)
            if task=='ToastBagel':
                copied=directory/'human-demonstration.jpg';copied.write_bytes(Path(photo).read_bytes())
                model.coaching_image=copied;model.coaching_receipt=directory/'human-coaching-input.json'
            for number in range(1,trials+1):
                if absolute_deadline is not None and time.time() >= absolute_deadline-60:
                    record['status']='time_limit_reached'
                    break
                if model._rpc('account/rateLimits/read',{}).get('accountId')!=EXPECTED_ACCOUNT:
                    raise RuntimeError('Account changed during RoboSWE sequence; stopped before another attempt')
                attempt,job=create('jobs',f'Astra high · {task} · attempt {number}/{trials} · retained','learning-attempt')
                job.update(task=task,campaign_id=record['id'],attempt_number=number,verifier_status='observed_only',
                           video_provenance='Original attempt · retained context, code, notes',human_coached=task=='ToastBagel')
                save(attempt,job)
                record.update(current_attempt=number,current_job=job['id']);save(directory,record)
                print(json.dumps({'series':record['id'],'task':task,'attempt':number,'job':job['id']}),flush=True)
                previous=[{k:r[k] for k in ('attempt_number','observed_success','termination_reason','control_steps')} for r in reports]
                runtime=None;journal=None
                try:
                    runtime=FreshEpisode(config,spec,attempt,number,previous,worker=worker)
                    fp=runtime.initial['fingerprint']
                    if first_fingerprint is None: first_fingerprint=fp
                    if fp!=first_fingerprint: raise RuntimeError('Reset fingerprint differs; refusing an unmatched learning episode')
                    journal=Journal(attempt/'submissions.jsonl')
                    runner=WaitingAttemptRunner(runtime,sandbox,journal)
                    before=usage_summary(directory/'model-events.jsonl')
                    (attempt/'continuity-start.json').write_text(json.dumps({'thread_id':model.thread_id,'workspace_keeper':sandbox.keeper,'volume':sandbox.volume,'initial_fingerprint':fp,'usage':before},indent=2))
                    with (attempt/'model-transcript.jsonl').open('x') as transcript:
                        deadline=time.monotonic()+timeout+900
                        if absolute_deadline is not None:
                            deadline=min(deadline,time.monotonic()+max(0,absolute_deadline-time.time()-30))
                        report,turn=drive_task(runner,model,0,execution,attempt,transcript,deadline,turn)
                    report.update(attempt_number=number,retained_context=number>1,retained_library=number>1,retained_notes=number>1,
                                  model_thread_id=model.thread_id,workspace_keeper=sandbox.keeper,usage_before=before,usage_after=usage_summary(directory/'model-events.jsonl'))
                    report['stages']=summarize_stages(attempt)
                    report['library']=sandbox.archive(attempt/'library.zip')
                    with ZipFile(attempt/'library.zip') as archive:
                        if 'NOTES.md' in archive.namelist():
                            (attempt/'NOTES.md').write_bytes(archive.read('NOTES.md'));job['artifacts']['notes']='NOTES.md'
                    reports.append(report)
                    (attempt/'report.json').write_text(json.dumps(report,indent=2))
                    job['artifacts'].update(report='report.json',library='library.zip')
                    job['status']='observed_pass' if report['observed_success'] else 'observed_fail'
                    write_human_outputs(attempt);job['artifacts']['stdout']='stdout.txt'
                except BaseException:
                    job['status']='infrastructure_error';(attempt/'stderr.txt').write_text(traceback.format_exc());job['artifacts']['stderr']='stderr.txt'
                    if not (attempt/'library.zip').exists() and not sandbox.active:
                        try:
                            sandbox.archive(attempt/'library.zip');job['artifacts']['library']='library.zip'
                        except Exception:
                            (attempt/'library-export-error.txt').write_text(traceback.format_exc())
                    raise
                finally:
                    if journal: journal.close()
                    if runtime:
                        receipt=runtime.close()
                        publish_video(Path(receipt['recording_dir']),attempt,job,phase='original')
                    job['finished_at']=now();save(attempt,job)
                    (directory/'report.json').write_text(json.dumps({'trials':reports,'benchmark_score':None,'assessment':plan['assessment']},indent=2))
                    record['artifacts']['report']='report.json';save(directory,record)
            record.update(status='completed_unscored' if len(reports)==trials else 'time_limit_reached',exit_code=0)
    except BaseException:
        record.update(status='error',exit_code=1)
        (directory/'stderr.txt').write_text(traceback.format_exc());record['artifacts']['stderr']='stderr.txt'
        raise
    finally:
        record['finished_at']=now();save(directory,record)
    return {'task':task,'series_id':record['id'],'trials':reports}



def assessment(results):
    lines = ['Astra high: retained-learning experiment',
             'Each sequence retains conversation, compaction summaries, code, and NOTES.md. Simulator resets are matched by initial-state fingerprint.',
             'These are observed task outcomes and descriptive within-sequence comparisons; no causal attribution to context alone and no independent replay certification.\n']
    for sequence in results:
        lines.append(sequence['task'])
        previous_files = {}
        for report in sequence['trials']:
            files = {f['path']: f['sha256'] for f in report['library']['files']}
            reused = sum(previous_files.get(k) == v for k, v in files.items())
            stages = report['stages']
            lines.append(f"Attempt {report['attempt_number']}: success={report['observed_success']}, steps={report['control_steps']}, seconds={report['elapsed_seconds']:.1f}, unchanged retained files={reused}, NOTES.md={'NOTES.md' in files}, supported-lift candidate longest streak={stages.get('longest_supported_lift_candidate_steps')}, total compactions={report['usage_after']['compactions']}")
            previous_files = files
        flags = [r['observed_success'] for r in sequence['trials']]
        if flags and not flags[0] and any(flags[1:]):
            lines.append('A later attempt succeeded after an initial failure: evidence consistent with adaptation, but dependent attempts do not isolate learning from exploration or chance.')
        elif flags and not any(flags):
            lines.append('No whole-task success was observed. Inspect videos, intermediate progress, notes, and code revisions before concluding whether useful adaptation occurred.')
        else:
            lines.append('Compare later success consistency and control-step/time efficiency with the first attempt; retained state alone is not evidence of learning.')
        lines.append('')
    return '\n'.join(lines)


def preflight_one(task,directory):
    config=json.loads(Path('jobs/20261004T151002Z-f2676811da/plan.json').read_text())['config'];config['task']='robocasa/'+task
    spec=serial_spec(task,timeout=180,max_steps=6000)
    directory.mkdir(parents=True,exist_ok=False)
    runtime=FreshEpisode(config,spec,directory,0,[])
    try:
        result=runtime.start_task(0);obs=result['observation'];shape=obs['action_schema']['shape'][0]
        action=[0.]*shape;action[6:12]=[-1.]*6
        start=time.monotonic()
        for _ in range(50):runtime.bridge({'operation':'step','action':action})
        return {'task':task,'50_recorded_steps_seconds':time.monotonic()-start,'initial_fingerprint':runtime.initial['fingerprint']}
    finally:runtime.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-image',required=True)
    parser.add_argument('--photo',required=True)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--timeout-seconds', type=int, default=720)
    parser.add_argument('--trials', type=int, default=5)
    args=parser.parse_args()
    if args.timeout_seconds < 1 or args.trials < 1:
        parser.error('Timeout and number of trials must be positive')
    tasks=list(dict.fromkeys(args.tasks))
    concurrency=len(tasks)
    directory,record=create('jobs',f"Astra high · {' + '.join(tasks)} · fixed wait · {args.trials} attempts · {args.timeout_seconds//60} min",'learning-campaign')
    record.update(status='preflight',benchmark_score=None)
    save(directory,record)
    print('Campaign '+record['id'],flush=True)
    Path('artifacts/toaster-learning-campaign.json').write_text(json.dumps({'id':record['id'],'directory':str(directory)}))
    # Reserve the reference host once; child sequences are the two explicitly owned slots.
    with reference_lease({'jobs_dir':'jobs','execution':{'memory_mb':6144,'cpu':4,'min_free_disk_gb':10}}):
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                receipts=list(pool.map(lambda task:preflight_one(task,directory/('preflight-'+task)),tasks))
            (directory/'plan.json').write_text(json.dumps({'tasks':tasks,'trials_per_task':args.trials,'timeout_seconds':args.timeout_seconds,'context_window':1000000,'compact_at':800000,'reasoning_effort':'high','concurrency':concurrency,'budget_policy':'passive_wait_v1','preflight':receipts},indent=2))
            record['artifacts']['plan']='plan.json'
            if max(x['50_recorded_steps_seconds'] for x in receipts)>45:
                raise RuntimeError('Simulator preflight leaves insufficient margin inside the 60-second execution timeout')
            record['status']='running';save(directory,record)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures=[pool.submit(run_series,task,args.code_image,record['id'],args.photo,timeout=args.timeout_seconds,trials=args.trials) for task in tasks]
                results=[]
                for future in futures:
                    results.append(future.result())
                    (directory/'report.json').write_text(json.dumps({'sequences':results,'benchmark_score':None},indent=2));record['artifacts']['report']='report.json';save(directory,record)
            (directory/'stdout.txt').write_text(assessment(results));record['artifacts']['stdout']='stdout.txt'
            record.update(status='completed_unscored',exit_code=0)
        except BaseException:
            record.update(status='error',exit_code=1)
            (directory/'stderr.txt').write_text(traceback.format_exc());record['artifacts']['stderr']='stderr.txt'
            raise
        finally:
            record['finished_at']=now();save(directory,record)


if __name__=='__main__':main()
