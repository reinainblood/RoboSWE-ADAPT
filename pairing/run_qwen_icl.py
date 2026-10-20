"""Three 30-minute Qwen attempts per task, retaining context and workspace."""
import argparse
import gzip
import hashlib
import json
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from pairing.overnight_suite import config_for, suite_worker
from pairing.run_hand_microwave import trial_slot
from pairing.run_qwen_trials import BASE_URL, MODEL, QwenModel, toast_worker
from pairing.run_toaster_learning import FreshEpisode, instructions, write_human_outputs
from pairing.simulation_wait import WaitingAttemptRunner
from roboswe.campaign import drive_task, publish_video
from roboswe.execution.sandbox import CodeSandbox
from roboswe.jobs import create, now, save
from roboswe.library.attempts import Journal
from roboswe.library.serial import serial_spec

COMPACT_AT = 800_000
TRIALS = 3
SECONDS = 1800


def save_context(path, messages):
    raw = json.dumps(messages, ensure_ascii=False).encode()
    with gzip.open(path, 'wb') as stream:
        stream.write(raw)
    return {'file': Path(path).name, 'sha256': hashlib.sha256(raw).hexdigest(), 'messages': len(messages)}


class LearningQwen(QwenModel):
    def __init__(self, prompt, directory):
        super().__init__(prompt, directory)
        self.context_tokens = 0
        self.compactions = 0
        self.metadata.update(compact_at=COMPACT_AT, context_policy='Retain full conversation across three attempts; model-generated compaction near 800000 tokens. Preserve archived pre-compaction history and continue from the same summary afterward.')

    def compact(self, timeout):
        number = self.compactions + 1
        checkpoint = save_context(self.directory / f'context-before-compaction-{number}.json.gz', self.messages)
        prompt = (
            'Summarize your current robot-task working context for your own continuation. '
            'Preserve the task, robot and controller conventions, file names and useful library APIs, '
            'tested observations, execution errors, unsuccessful assumptions, current physical state, '
            'remaining plan, and attempt/reset history. Clearly distinguish measured evidence from '
            'hypotheses. Do not claim code ran when it was only written. Do not solve a new step. '
            'The code library and NOTES.md persist separately. Return JSON with one string field summary.'
        )
        payload = {'model': MODEL, 'messages': self.messages + [{'role':'user','content':prompt}],
            'reasoning_effort':'high', 'max_tokens':16384, 'store':False,
            'response_format':{'type':'json_schema','json_schema':{'name':'context_checkpoint','strict':True,
                'schema':{'type':'object','properties':{'summary':{'type':'string'}},'required':['summary'],'additionalProperties':False}}}}
        request = urllib.request.Request(BASE_URL+'/chat/completions', data=json.dumps(payload).encode(),
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=max(.1,timeout)) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'Context compaction HTTP {exc.code}; credentials suppressed') from None
        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(exc,TimeoutError) or isinstance(getattr(exc,'reason',None),TimeoutError):
                raise TimeoutError('Context compaction reached task deadline') from None
            raise RuntimeError('Context compaction transport failed') from None
        (self.directory/f'compaction-{number}.json').write_text(json.dumps({'checkpoint':checkpoint,'response':result},indent=2))
        summary = json.loads(result['choices'][0]['message']['content'])['summary']
        if not isinstance(summary,str) or not summary.strip():
            raise ValueError('Invalid context compaction; original context preserved')
        self.messages = [self.messages[0], {'role':'user','content':
            'Continue the same task conversation using your saved working-context summary below. '
            'The simulator and library have not changed during compaction. Any later reset notice supersedes physical-state descriptions.\n'+summary}]
        self.compactions = number
        self.context_tokens = len(summary)  # Conservative until the next provider receipt.
        with (self.directory/'model-usage.jsonl').open('a') as stream:
            stream.write(json.dumps({'event':'compaction','number':number,'usage':result.get('usage')})+'\n')
        with (self.directory/'stdout.txt').open('a') as stream:
            stream.write(f'\nCONTEXT COMPACTION {number}\n{summary}\n')

    def ask(self, text, **kwargs):
        started = time.monotonic()
        timeout = kwargs.get('timeout',180)
        if self.context_tokens >= COMPACT_AT:
            self.compact(timeout)
        kwargs['timeout'] = timeout - (time.monotonic()-started)
        if kwargs['timeout'] <= 0:
            raise TimeoutError('No active-work time remains after compaction')
        decision = super().ask(text, **kwargs)
        last = json.loads((self.directory/'model-usage.jsonl').read_text().splitlines()[-1])
        usage = last.get('usage') or {}
        self.context_tokens = usage.get('prompt_tokens',0) + usage.get('completion_tokens',0)
        return decision


class LearningEpisode(FreshEpisode):
    def __init__(self, *args, number, previous, **kwargs):
        self.number = number
        self.previous = previous
        super().__init__(*args, number, previous, **kwargs)

    def start_task(self, index):
        result = super().start_task(index)
        if self.number == 1:
            notice = 'Attempt 1 of 3: fresh conversation and empty library. No earlier model code was supplied.'
        else:
            reason = self.previous[-1]['termination_reason']
            ending = 'reached its 30-minute active-work deadline' if reason in ('task_deadline','model_deadline') else 'ended: '+reason
            notice = f'Attempt {self.number-1} {ending}. The complete simulator scene, appliance, objects, robot body and hands have RESET to the same starting state. Your conversation, including any compaction summary, code library and NOTES.md have been maintained. The last physical pose has NOT persisted.'
        result.update(trial_number=self.number,total_trials=TRIALS,previous_attempts=self.previous,
            reset_notice=notice+' You now have a fresh 1800-second active-work allowance; bounded passive waiting is accounted separately.')
        return result


def trial_prompt(task):
    prompt = instructions(task,timeout=SECONDS,trials=TRIALS)
    if task == 'GetToastedBread':
        from pairing.run_gr1_trial import GOAL
        prompt = prompt.replace('Success is checked by the official task predicate.','Success is checked by the supplied retrieval-only diagnostic predicate.')
        prompt += '\n'+GOAL+'\nUse the supplied two-hand GR1 native absolute pose interface. Passive wait holds current hand poses and finger targets. This is not a full RoboCasa task score.'
    return prompt


def run_series(parent, campaign, task, attempts, image):
    config = config_for(task, 'GR1' if task == 'GetToastedBread' else 'PandaOmron')
    prompt = trial_prompt(task)
    prior = []
    fingerprint = None
    with CodeSandbox(image,lifetime_seconds=9000) as sandbox:
        model = LearningQwen(prompt,attempts[0][0])
        for number,(directory,record) in enumerate(attempts,1):
            runtime = journal = None
            model.directory = directory
            (directory/'stdout.txt').write_text('Provider-returned reasoning, stated rationale, and actual execution output. Simulation timestamps identify the physical state for each decision.\n')
            (directory/'model.json').write_text(json.dumps(model.metadata,indent=2))
            (directory/'plan.json').write_text(json.dumps({'config':config,'instructions':prompt,'trials':TRIALS,
                'attempt_number':number,'active_work_seconds':SECONDS,'max_control_steps':6000,
                'max_steps_per_execution':50,'execution_timeout_seconds':60,'code_image':image,
                'feedback_protocol':'explicit-operation-and-depth-contract-v1','model':model.metadata},indent=2))
            record['artifacts'].update(plan='plan.json',stdout='stdout.txt')
            continuity = {'workspace_keeper':sandbox.keeper,'retained_context':number>1,'retained_library':number>1,
                'compactions_before':model.compactions,'context':save_context(directory/'context-start.json.gz',model.messages),
                'library':sandbox.archive(directory/'library-start.zip')}
            (directory/'continuity.json').write_text(json.dumps(continuity,indent=2))
            calls_before = model.calls
            try:
                spec = serial_spec(task,timeout=SECONDS,max_steps=6000)
                spec['learning_trials'] = TRIALS
                runtime = LearningEpisode(config,spec,directory,number=number,previous=prior,
                    worker=toast_worker if task=='GetToastedBread' else suite_worker)
                actual = runtime.initial['fingerprint']
                if fingerprint is None: fingerprint = actual
                if actual != fingerprint: raise RuntimeError('Reset fingerprint changed; refusing to mix initial states within ICL sequence')
                continuity['initial_fingerprint'] = actual
                (directory/'continuity.json').write_text(json.dumps(continuity,indent=2))
                journal = Journal(directory/'submissions.jsonl')
                record.update(status='running',execution_started_at=now())
                save(directory,record)
                with (directory/'model-transcript.jsonl').open('x') as transcript:
                    report,_ = drive_task(WaitingAttemptRunner(runtime,sandbox,journal),model,0,
                        {'max_control_steps':6000,'max_attempt_steps':50,'attempt_timeout_seconds':60},
                        directory,transcript,time.monotonic()+2700)
                report.update(attempt_number=number,retained_context=number>1,retained_library=number>1,
                    model_calls=model.calls-calls_before,compactions_total=model.compactions,
                    library=sandbox.archive(directory/'library.zip'))
                prior.append({k:report[k] for k in ('attempt_number','termination_reason','observed_success','control_steps')})
                (directory/'report.json').write_text(json.dumps(report,indent=2))
                with ZipFile(directory/'library.zip') as archive:
                    if 'NOTES.md' in archive.namelist():
                        (directory/'NOTES.md').write_bytes(archive.read('NOTES.md'))
                        record['artifacts']['notes'] = 'NOTES.md'
                reasoning = (directory/'stdout.txt').read_text()
                write_human_outputs(directory)
                outputs = (directory/'stdout.txt').read_text()
                (directory/'stdout.txt').write_text(reasoning+'\nACTION RATIONALES AND EXECUTION OUTPUT\n'+outputs)
                record['artifacts'].update(report='report.json',library='library.zip')
                if task=='GetToastedBread': record['status']='diagnostic_pass' if report['observed_success'] else 'diagnostic_incomplete'
                else: record['status']='observed_pass' if report['observed_success'] else 'observed_fail'
            except BaseException:
                record.update(status='infrastructure_error',exit_code=1)
                (directory/'stderr.txt').write_text(traceback.format_exc())
                record['artifacts']['stderr']='stderr.txt'
                raise
            finally:
                save_context(directory/'context-end.json.gz',model.messages)
                try:
                    if journal: journal.close()
                    if runtime:
                        receipt=runtime.close()
                        publish_video(Path(receipt['recording_dir']),directory,record,phase='original')
                finally:
                    record['finished_at']=now()
                    save(directory,record)
            print(json.dumps({'task':task,'attempt':number,'status':record['status'],'job':record['id']}),flush=True)
            # Reset notice in the next prompt carries prior outcome; the very
            # same model messages, compacted context and workspace remain alive.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-image',required=True)
    args=parser.parse_args()
    parent,campaign=create('jobs','Quinny high · 3×30m ICL each · microwave + GR1 toast','qwen-icl-campaign')
    campaign.update(status='queued',attempt_jobs=[],benchmark_score=None)
    tasks={}
    with ZipFile(parent/'coordinator-source.zip','w') as archive:
        for name in ('pairing/run_qwen_icl.py','pairing/run_qwen_trials.py','pairing/run_gr1_trial.py','pairing/run_toaster_learning.py','pairing/overnight_suite.py','pairing/simulation_wait.py','roboswe/campaign.py','roboswe/library/attempts.py','roboswe/library/runtime.py','roboswe/library/serial.py'):
            archive.write(name,name)
    campaign['artifacts']['source_archive']='coordinator-source.zip'
    for task in ('CloseMicrowave','GetToastedBread'):
        tasks[task]=[]
        for number in range(1,TRIALS+1):
            directory,record=create('jobs',f'Quinny high · {task} · ICL {number}/3 · 30 min','qwen-learning-attempt')
            record.update(status='queued',task=task,attempt_number=number,campaign_id=campaign['id'],benchmark_score=None,
                verifier_status='diagnostic_only' if task=='GetToastedBread' else 'observed_only')
            save(directory,record)
            tasks[task].append((directory,record))
            campaign['attempt_jobs'].append(record['id'])
    save(parent,campaign)
    Path('artifacts/qwen-icl-job.json').write_text(json.dumps({'id':campaign['id'],'attempt_jobs':campaign['attempt_jobs']},indent=2))
    print(json.dumps({'campaign':campaign['id'],'attempt_jobs':campaign['attempt_jobs']}),flush=True)
    failures=[]
    try:
        with trial_slot({'jobs_dir':'jobs','execution':{'memory_mb':6144,'cpu':4,'min_free_disk_gb':10}},True):
            campaign['status']='running';save(parent,campaign)
            for task,attempts in tasks.items():
                try: run_series(parent,campaign,task,attempts,args.code_image)
                except Exception:  # noqa: BLE001 -- Preserve the failed series; run the independently authorized other task.
                    failures.append(task)
                    for directory,record in attempts:
                        if record['status']=='queued':
                            record.update(status='blocked_by_series_error',finished_at=now());save(directory,record)
        campaign['status']='completed_with_errors' if failures else 'completed_unscored'
    except BaseException:
        campaign['status']='infrastructure_error'
        raise
    finally:
        campaign.update(finished_at=now(),failed_series=failures)
        save(parent,campaign)


if __name__=='__main__': main()
