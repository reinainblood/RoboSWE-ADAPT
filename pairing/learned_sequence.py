"""Opt-in experiment: execute authored script sequences with bounded feedback.

Not enabled in the ongoing cohort. A matched experiment must expose this same
operation from attempt one in both retained and fresh-context arms. The harness
supplies no task code, script order, motion targets, or successful prefix.
"""
import copy
import json
import time
from pairing.cohort50 import EfficientRunner, SCHEMA
from pairing.run_toaster_learning import LearningModel
from roboswe.execution.sandbox import validate_path

SEQUENCE_SCHEMA=copy.deepcopy(SCHEMA)
SEQUENCE_SCHEMA['properties']['operation']['enum'].append('execute_sequence')
MAX_SCRIPTS=12
MAX_SECONDS=600
INSTRUCTIONS='''
Optional operation execute_sequence runs an ordered list of YOUR existing .py
scripts. Set path to an empty string and content to a JSON array of 1–12 relative
.py paths. Each script retains its usual 50-control-step execution limit and
per-script wall limit. The complete sequence is additionally limited to 600
seconds and the remaining episode budget. Every operation is journaled. The
sequence stops after a script error, official sustained success, or exhausted
budget. No physics or model budget is granted by using a sequence.
Use this for your already tested code when its observed preconditions match.
Put sensor checks and exception-based stopping conditions in your own scripts;
there is no model inspection between scripts. Use individual execution for new
probes or when you need full stdout. Sequence feedback includes a bounded stdout
tail per script; full outputs remain in the run evidence. A reset still resets
all physics. No task-specific sequence is provided by the harness.
'''


class SequenceRunner(EfficientRunner):
    def start_task(self,index):
        result=super().start_task(index)
        result['sequence_limits']={'max_scripts':MAX_SCRIPTS,'wall_seconds':MAX_SECONDS,'max_control_steps':600,'scope':'Additional batch caps; original per-script and episode caps still apply.'}
        return result

    def apply(self,decision,*,timeout=60,max_control_steps=50,**kwargs):
        if decision.get('operation')!='execute_sequence':
            return super().apply(decision,timeout=timeout,max_control_steps=max_control_steps,**kwargs)
        if decision.get('path')!='':raise ValueError('Sequence path must be empty')
        try:paths=json.loads(decision.get('content',''))
        except (ValueError,TypeError):raise ValueError('Sequence content must be a JSON array of Python paths') from None
        if not isinstance(paths,list) or not 1<=len(paths)<=MAX_SCRIPTS:raise ValueError('Sequence requires 1–12 scripts')
        for path in paths:validate_path(path,operation='execute')
        if max_control_steps is None or not 1<=max_control_steps<=50:raise ValueError('Sequence requires the original bounded per-script control limit')
        started=time.monotonic();receipts=[];stop='completed';steps=0
        for path in paths:
            episode_remaining=self.runtime.timeout-(time.monotonic()-self.runtime.started)
            batch_remaining=MAX_SECONDS-(time.monotonic()-started)
            if episode_remaining<=0:stop='task_deadline';break
            if batch_remaining<=0:stop='sequence_wall_limit';break
            result=super().apply({'operation':'execute','path':path,'content':'','message':decision.get('message','')},timeout=min(timeout,episode_remaining,batch_remaining),max_control_steps=min(max_control_steps,600-steps),**kwargs)
            # AttemptRunner.apply returns the execution receipt directly; only
            # the journal wraps it under a result key. result['result'] is often None.
            detail=result;accepted=detail.get('accepted_control_steps',0);steps+=accepted
            output=detail.get('output','')
            receipts.append({'path':path,'returncode':detail.get('returncode'),'error':detail.get('error'),'accepted_control_steps':accepted,'output_tail':output[-2000:],'output_tail_truncated':len(output)>2000 or bool(detail.get('output_truncated'))})
            if detail.get('returncode') or detail.get('error') or result.get('protocol_error'):
                stop='execution_error';break
            observation=self.runtime.observe()
            if observation.get('task_success'):stop='sustained_success';break
            if observation.get('simulator_phase')!='active':raise RuntimeError('Simulator faulted during sequence')
            if steps>=600:stop='sequence_control_limit';break
        return {'operation':'execute_sequence','scripts':receipts,'accepted_control_steps':steps,'stop_reason':stop,'seconds':time.monotonic()-started,'notice':'Only submitted scripts executed. Full per-script receipts are preserved in the journal; no intermediate model decisions occurred.'}


class SequenceAstra(LearningModel):
    """Explicit experimental client; existing Astra clients remain unchanged."""
    def ask(self,text,**kwargs):
        kwargs['schema']=SEQUENCE_SCHEMA
        return super().ask(text,**kwargs)
