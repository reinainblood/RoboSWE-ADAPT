"""User-authorized Qwen VL trials; credentials remain outside the code sandbox."""
import argparse
import base64
import json
import mimetypes
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from pairing.run_toaster_learning import FreshEpisode, instructions, write_human_outputs
from pairing.run_hand_microwave import trial_slot
from pairing.simulation_wait import WAIT_SCHEMA, WaitingAttemptRunner
from roboswe.campaign import drive_task, publish_video
from roboswe.execution.sandbox import CodeSandbox
from roboswe.jobs import create, now, save
from roboswe.library.attempts import Journal
from roboswe.library.serial import serial_spec

BASE_URL = 'https://example.invalid/legacy-provider/v1'
MODEL = 'Qwen/Qwen3.8-Max-VL-Thinking'
BASELINE = '20261005T052831Z-1a99ce077a'


def credential():
    import os
    value = os.environ.get("ROBOSWE_LEGACY_API_KEY")
    if not value:
        raise RuntimeError("Set ROBOSWE_LEGACY_API_KEY in the trusted coordinator environment")
    return value


class ModelProtocolError(ValueError):
    """A returned final message is not an executable structured decision."""


def parse_decision(message, schema):
    answer = message.get('content')
    if not isinstance(answer, str):
        raise ModelProtocolError('Qwen returned no structured final content')
    try:
        decision = json.loads(answer)
    except json.JSONDecodeError:
        raise ModelProtocolError('Qwen final content is not valid JSON') from None
    if not isinstance(decision, dict) or set(decision) != {'operation', 'path', 'content', 'message'}:
        raise ModelProtocolError('Qwen returned an invalid decision object')
    if not all(isinstance(v, str) for v in decision.values()):
        raise ModelProtocolError('Decision values must be strings')
    if decision['operation'] not in schema['properties']['operation']['enum']:
        raise ModelProtocolError('Unknown model operation')
    return decision


class QwenModel:
    def __init__(self, prompt, directory):
        self.key = credential()
        self.directory = Path(directory)
        self.messages = [{'role': 'system', 'content': prompt}]
        self.calls = 0
        self.metadata = {
            'model': MODEL, 'base_url': BASE_URL, 'environment': 'FDR',
            'reasoning_effort': 'high', 'context_window_advertised': 1_000_000,
            'max_completion_tokens': 32768, 'context_policy': 'Full messages and provider reasoning retained within each task; no trimming or compaction',
            'explanations': 'Provider-returned reasoning_content and structured public message recorded separately; not a claim to expose inaccessible internal computation',
            'credential_source': 'Existing macOS Keychain Modal FDR resolver; host memory only',
        }

    def ask(self, text, *, images=(), schema=WAIT_SCHEMA, timeout=180, as_tool_output=False):
        content = [{'type': 'text', 'text': text}]
        for path in images:
            path = Path(path)
            mime = mimetypes.guess_type(path.name)[0] or 'image/png'
            content.append({'type': 'image_url', 'image_url': {
                'url': f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode()}})
        pending = {'role': 'user', 'content': content}
        payload = {
            'model': MODEL, 'messages': self.messages + [pending], 'store': False,
            'reasoning_effort': 'high', 'max_tokens': 32768,
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'robot_decision', 'strict': True, 'schema': schema}},
        }
        request = urllib.request.Request(BASE_URL + '/chat/completions',
            data=json.dumps(payload).encode(), headers={
                'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=max(.1, timeout)) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'Qwen endpoint HTTP {exc.code}; request credentials suppressed') from None
        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(exc, TimeoutError) or isinstance(getattr(exc, 'reason', None), TimeoutError):
                raise TimeoutError('Qwen request reached the remaining task deadline') from None
            raise RuntimeError('Qwen transport failed') from None
        self.calls += 1
        observation = json.loads(text).get('observation', {})
        step = observation.get('control_step', 0)
        receipt = {'call': self.calls, 'control_step': step, 'video_seconds': step / 20,
            'elapsed_seconds': time.monotonic() - started, 'response': result}
        with (self.directory / 'provider-responses.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt) + '\n')
        message = result['choices'][0]['message']
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        with (self.directory / 'stdout.txt').open('a') as stream:
            stream.write(f'\nVideo {step / 20:.2f}s · call {self.calls}\nPROVIDER-RETURNED REASONING\n{reasoning or "No reasoning text returned"}\n')
            stream.write('STRUCTURED RESPONSE / STATED RATIONALE\n' + str(message.get('content')) + '\n')
        history = {'role': 'assistant', 'content': message.get('content') if isinstance(message.get('content'),str) else ''}
        if message.get('reasoning_content'):
            history['reasoning_content'] = message['reasoning_content']
        self.messages.extend([pending, history])
        with (self.directory / 'model-usage.jsonl').open('a') as stream:
            stream.write(json.dumps({'call': self.calls, 'usage': result.get('usage'),
                'reasoning_characters': len(reasoning), 'finish_reason': result['choices'][0].get('finish_reason')}) + '\n')
        # Preserve rejected responses and their usage for a bounded coordinator
        # retry. Only validated FINAL content can become an operation.
        return parse_decision(message, schema)


def toast_worker(pipe, config, spec, directory, number, previous):
    import pairing.run_gr1_trial as gr1
    from pairing.gr1_compat import install_stock_gr1_placement
    directory = Path(directory)
    gr1.ROOT = directory
    install_stock_gr1_placement()
    session = runtime = evidence = None
    try:
        session = gr1.GR1Session(config)
        session.recording_root = directory / 'recordings'
        session.start_recording({'recording_id': 'attempt', 'width': 640, 'height': 480})
        evidence = (directory / 'actions.jsonl').open('x')
        runtime = gr1.RetrievalRuntime(session, spec, timeout=spec['task_timeout_seconds'], max_steps=6000, evidence=evidence)
        pipe.send({'ok': runtime.metadata(), 'fingerprint': session.initial})
        while True:
            request = pipe.recv()
            op = request['operation']
            try:
                if op == 'start_task': result = runtime.start_task(0)
                elif op == 'observe': result = runtime.observe()
                elif op == 'bridge': result = runtime.bridge(request['message'])
                elif op == 'wait': result = runtime.wait(request['steps'])
                elif op == 'finish_task': result = runtime.finish_task(code_errors=request['code_errors'])
                elif op == 'digest': result = runtime.action_digest.hexdigest()
                elif op == 'close':
                    session.stop_recording({'recording_id': 'attempt'})
                    (directory / 'final-snapshot.zip').write_bytes(session.snapshot())
                    pipe.send({'ok': {'recording_dir': str(session.recording.directory)}})
                    break
                else: raise ValueError('Unknown coordinator operation')
                pipe.send({'ok': result})
            except Exception as exc:
                pipe.send({'error': traceback.format_exc(), 'error_type': type(exc).__name__})
    except BaseException:
        try: pipe.send({'error': traceback.format_exc()})
        except (BrokenPipeError, EOFError): pass
    finally:
        if runtime: runtime.diagnostic.close()
        if evidence: evidence.close()
        if session:
            if session.recording and session.recording.active:
                session.stop_recording({'recording_id': 'attempt'})
            session.close()
        pipe.close()


def run_attempt(directory, record, image, task):
    from pairing.overnight_suite import suite_worker, config_for
    config = config_for(task, 'GR1' if task == 'GetToastedBread' else 'PandaOmron')
    prompt = instructions(task, timeout=1800, trials=1)
    if task == 'GetToastedBread':
        from pairing.run_gr1_trial import GOAL
        prompt = prompt.replace('Success is checked by the official task predicate.', 'Success is checked by the supplied retrieval-only diagnostic predicate.')
        prompt += '\n' + GOAL + '\nUse the supplied two-hand GR1 native absolute pose interface. The wait operation holds current hand poses and finger targets. This is not a full RoboCasa task score.\n'
    spec = serial_spec(task, timeout=1800, max_steps=6000)
    spec['learning_trials'] = 1
    plan = {'config': config, 'instructions': prompt, 'model': MODEL, 'reasoning_effort': 'high',
        'active_work_seconds': 1800, 'max_control_steps': 6000, 'code_image': image,
        'fresh_context_and_library': True, 'astra_reference_job': BASELINE if task == 'CloseMicrowave' else '20261005T041113Z-0332917bd7',
        'comparison_limit': 'Descriptive model trial. Historical Astra ICL and reset/runtime changes prevent a matched causal comparison.'}
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2))
    record['artifacts'].update(plan='plan.json', stdout='stdout.txt')
    (directory / 'stdout.txt').write_text('Qwen provider-returned reasoning and separately stated rationale. Intent is not verified behavior. Video timestamps use simulation time.\n')
    runtime = journal = None
    try:
        with CodeSandbox(image, lifetime_seconds=3200) as sandbox:
            model = QwenModel(prompt, directory)
            (directory / 'model.json').write_text(json.dumps(model.metadata, indent=2))
            runtime = FreshEpisode(config, spec, directory, 1, [], worker=toast_worker if task == 'GetToastedBread' else suite_worker)
            (directory / 'initial.json').write_text(json.dumps(runtime.initial, indent=2))
            journal = Journal(directory / 'submissions.jsonl')
            runner = WaitingAttemptRunner(runtime, sandbox, journal)
            record.update(status='running', execution_started_at=now())
            save(directory, record)
            with (directory / 'model-transcript.jsonl').open('x') as transcript:
                report, _ = drive_task(runner, model, 0,
                    {'max_control_steps': 6000, 'max_attempt_steps': 50, 'attempt_timeout_seconds': 60},
                    directory, transcript, time.monotonic() + 2700)
            report['library'] = sandbox.archive(directory / 'library.zip')
            report['model_calls'] = model.calls
            (directory / 'report.json').write_text(json.dumps(report, indent=2))
            # The standard human output includes execution stdout. Retain the
            # requested provider reasoning when adding those action summaries.
            reasoning_text = (directory / 'stdout.txt').read_text()
            write_human_outputs(directory)
            action_text = (directory / 'stdout.txt').read_text()
            (directory / 'stdout.txt').write_text(reasoning_text + '\nACTION RATIONALES AND EXECUTION OUTPUT\n' + action_text)
            record['artifacts'].update(report='report.json', library='library.zip')
            record['status'] = ('diagnostic_pass' if report['observed_success'] else 'diagnostic_incomplete') if task == 'GetToastedBread' else ('observed_pass' if report['observed_success'] else 'observed_fail')
    except BaseException:
        record.update(status='infrastructure_error', exit_code=1)
        (directory / 'stderr.txt').write_text(traceback.format_exc())
        record['artifacts']['stderr'] = 'stderr.txt'
    finally:
        if journal: journal.close()
        try:
            if runtime:
                receipt = runtime.close()
                publish_video(Path(receipt['recording_dir']), directory, record, phase='original')
        except Exception:
            record['artifact_errors'].append({'stage': 'finalize', 'error': traceback.format_exc()})
        record['finished_at'] = now()
        save(directory, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-image', required=True)
    args = parser.parse_args()
    parent, campaign = create('jobs', 'Quinny · Qwen VL high · microwave and GR1 toast', 'qwen-two-task-trial')
    campaign.update(status='queued', attempt_jobs=[], benchmark_score=None)
    attempts = []
    for task in ('CloseMicrowave', 'GetToastedBread'):
        directory, record = create('jobs', f'Quinny high · {task}' + (' · GR1 retrieval only' if task == 'GetToastedBread' else ' · PandaOmron'), 'qwen-model-task')
        record.update(status='queued', task=task, campaign_id=campaign['id'], benchmark_score=None,
            verifier_status='diagnostic_only' if task == 'GetToastedBread' else 'observed_only')
        (directory / 'source.py').write_bytes(Path(__file__).read_bytes())
        record['artifacts']['source'] = 'source.py'
        save(directory, record)
        attempts.append((directory, record, task))
        campaign['attempt_jobs'].append(record['id'])
    save(parent, campaign)
    Path('artifacts/qwen-trials-job.json').write_text(json.dumps({'id': campaign['id'], 'attempt_jobs': campaign['attempt_jobs']}, indent=2))
    print(json.dumps({'campaign': campaign['id'], 'attempt_jobs': campaign['attempt_jobs']}), flush=True)
    try:
        with trial_slot({'jobs_dir': 'jobs', 'execution': {'memory_mb': 6144, 'cpu': 4, 'min_free_disk_gb': 10}}, True):
            campaign['status'] = 'running'
            save(parent, campaign)
            for directory, record, task in attempts:
                run_attempt(directory, record, args.code_image, task)
                print(json.dumps({'job': record['id'], 'status': record['status']}), flush=True)
        campaign['status'] = 'completed_unscored' if all(r['status'] != 'infrastructure_error' for _, r, _ in attempts) else 'completed_with_errors'
    except BaseException:
        campaign['status'] = 'infrastructure_error'
        raise
    finally:
        campaign['finished_at'] = now()
        save(parent, campaign)


if __name__ == '__main__':
    main()
