import json,time
from types import SimpleNamespace
import pytest
from pairing.learned_sequence import SequenceRunner,EfficientRunner


def runner(monkeypatch,results,success_at=None):
    calls=[];r=object.__new__(SequenceRunner)
    r.runtime=SimpleNamespace(timeout=1800,started=time.monotonic(),observe=lambda:{'simulator_phase':'active','task_success':len(calls)==success_at})
    def apply(self,decision,**kw):calls.append((decision,kw));return results[len(calls)-1]
    monkeypatch.setattr(EfficientRunner,'apply',apply)
    return r,calls


def decision(paths):return {'operation':'execute_sequence','path':'','content':json.dumps(paths),'message':'Execute my tested scripts'}


def test_sequence_keeps_per_script_limits_and_stops_at_success(monkeypatch):
    result={'returncode':0,'error':None,'result':None,'accepted_control_steps':50,'output':'ok'}
    r,calls=runner(monkeypatch,[result]*3,success_at=2)
    receipt=r.apply(decision(['approach.py','grasp.py','move.py']))
    assert len(calls)==2 and receipt['stop_reason']=='sustained_success'
    assert receipt['accepted_control_steps']==100
    assert all(c[1]['max_control_steps']==50 and c[1]['timeout']<=60 for c in calls)


def test_sequence_validates_all_paths_before_any_execution(monkeypatch):
    r,calls=runner(monkeypatch,[])
    with pytest.raises(ValueError):r.apply(decision(['good.py','../bad.py']))
    assert not calls


def test_sequence_stops_on_error_and_clamps_remaining_episode(monkeypatch):
    r,calls=runner(monkeypatch,[{'returncode':1,'error':None,'result':None,'accepted_control_steps':3,'output':'x'*3000}])
    r.runtime.started=time.monotonic()-1795
    result=r.apply(decision(['guarded.py','later.py']))
    assert len(calls)==1 and 0<calls[0][1]['timeout']<=5
    assert result['stop_reason']=='execution_error'
    assert len(result['scripts'][0]['output_tail'])==2000
    assert result['scripts'][0]['output_tail_truncated']


def test_real_attempt_runner_receipt_and_journal_contract(tmp_path):
    import hashlib
    from roboswe.library.attempts import Journal,read_journal
    class Runtime:
        timeout=60
        started=time.monotonic()
        steps=0
        action_digest=hashlib.sha256()
        def metadata(self):return {}
        def start_task(self,index):return {'observation':self.observe()}
        def observe(self):return {'simulator_phase':'active','control_step':self.steps,'task_success':self.steps>=4}
        def bridge(self,message):
            self.steps+=1;self.action_digest.update(b'step');return self.observe()
    class Sandbox:
        image='contract-test'
        def request(self,operation,path,*,content,bridge,timeout):
            for _ in range(2):bridge({'operation':'step','action':[0]})
            return {'returncode':0,'error':None,'result':None,'output':'two ticks','output_truncated':False}
    journal=Journal(tmp_path/'submissions.jsonl');r=SequenceRunner(Runtime(),Sandbox(),journal);r.start_task(0)
    result=r.apply(decision(['a.py','b.py','never.py']));journal.close()
    assert result['stop_reason']=='sustained_success' and result['accepted_control_steps']==4
    assert r.execution_attempts==2
    entries=read_journal(tmp_path/'submissions.jsonl')
    assert sum(e['kind']=='operation_result' for e in entries)==2
