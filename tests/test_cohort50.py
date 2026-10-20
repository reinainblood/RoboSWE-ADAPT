import json
from types import SimpleNamespace

import pytest

from pairing import cohort50 as c


def test_frozen_cohort_balanced_unique_and_required(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    result=c.freeze()
    rows=result['tasks']
    assert len(rows)==len({r['task'] for r in rows})==50
    assert rows[1]['task']=='BlendIngredients'
    assert {r['config']['robot'] for r in rows}=={'PandaOmron','GR1'}
    assert len({r['config']['env_kwargs']['layout_ids'] for r in rows})==5
    assert sum(r['difficulty_prior']=='hard' for r in rows)==15
    assert c.freeze()==result
    assert 'Human coaching from Kirsten' not in c.prompt_for('ToastBagel')


def test_combined_operation_journals_both_and_preserves_execution_cap(monkeypatch):
    calls=[]
    def apply(self,d,**kw):
        calls.append((d,kw));return {'returncode':0,'error':None}
    monkeypatch.setattr(c.WaitingAttemptRunner,'apply',apply)
    runner=object.__new__(c.EfficientRunner)
    runner.apply({'operation':'write_execute','path':'probe.py','content':'print(1)','message':'probe'},timeout=60,max_control_steps=50)
    assert [x[0]['operation'] for x in calls]==['write','execute']
    assert calls[1][0]['content']==''
    assert calls[1][1]['max_control_steps']==50
    assert 0<calls[1][1]['timeout']<=60


def test_failed_save_never_executes(monkeypatch):
    calls=[]
    def apply(self,d,**kw):
        calls.append(d);return {'returncode':1,'error':'write failed'}
    monkeypatch.setattr(c.WaitingAttemptRunner,'apply',apply)
    runner=object.__new__(c.EfficientRunner)
    result=runner.apply({'operation':'write_execute','path':'probe.py','content':'bad','message':''})
    assert len(calls)==1 and result['error']=='write failed'
    with pytest.raises(ValueError):runner.apply({'operation':'write_execute','path':'NOTES.md'})


def test_qwen_byte_compaction_reserves_remaining_time(monkeypatch,tmp_path):
    model=object.__new__(c.EfficientQwen)
    model.messages=[{'role':'user','content':'x'*8_000_001}]
    calls=[]
    monkeypatch.setattr(model,'compact',lambda timeout:calls.append(timeout))
    def ask(self,text,**kw):
        assert kw['schema']==c.SCHEMA
        assert 0<kw['timeout']<=30
        return {'ok':True}
    monkeypatch.setattr(c.LearningQwen,'ask',ask)
    assert model.ask('{}',timeout=30)=={'ok':True}
    assert len(calls)==1


def test_drive_omits_duplicate_images_but_explicit_observe_refreshes(tmp_path,monkeypatch):
    import time
    from PIL import Image
    monkeypatch.setattr(c,'require_disk_headroom',lambda *a:None)
    runtime=SimpleNamespace(timeout=30,started=time.monotonic(),step=0)
    observed=[]
    def observation():
        observed.append(runtime.step)
        return {'control_step':runtime.step,'simulator_phase':'active','task_success':False}
    runtime.observe=observation
    class Runner:
        task_active=True
        def start_task(self,index):
            return {'precondition_valid':True,'observation':observation()}
        def apply(self,decision,**kwargs):
            if decision['operation']=='write_execute':runtime.step+=2
            if decision['operation']=='finish':self.task_active=False
            return {'operation':decision['operation'],'accepted_control_steps':2 if decision['operation']=='write_execute' else 0}
    runner=Runner();runner.runtime=runtime
    def model_input(feedback,obs,directory,index):
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'camera-{index}.png';Image.new('RGB',(8,8)).save(path)
        return json.dumps({'feedback':feedback,'observation':obs}),[path]
    monkeypatch.setattr(c,'model_input',model_input)
    class Model:
        calls=[]
        def ask(self,text,**kwargs):
            self.calls.append((json.loads(text),len(kwargs['images']),kwargs['as_tool_output']))
            self.last_provider_thinking_trace={
                'schema_version':1,
                'call':len(self.calls),
                'blocks':[{'content_block_index':0,'block':{'type':'thinking','thinking':f'trace-{len(self.calls)}','signature':'signed'}}],
            }
            operation=['write','observe','write_execute','finish'][len(self.calls)-1]
            return {'operation':operation,'path':'x.py' if operation in ('write','write_execute') else '','content':'','message':''}
    model=Model();c.drive(runner,model,tmp_path)
    assert [x[1] for x in model.calls]==[1,0,1,1]
    assert [x[2] for x in model.calls]==[False,True,True,True]
    assert model.calls[1][0]['observation']['unchanged_since_previous_observation']
    assert model.calls[2][0]['observation']['control_step']==0
    assert 'unchanged_since_previous_observation' not in model.calls[2][0]['observation']
    assert model.calls[3][0]['observation']['control_step']==2
    assert observed==[0,0,2]
    assert [call[0]['episode_budget']['control_steps_remaining'] for call in model.calls]==[6000,6000,6000,5998]
    remaining=[call[0]['episode_budget']['active_work_seconds_remaining'] for call in model.calls]
    assert all(0<value<=30 for value in remaining)
    assert remaining==sorted(remaining,reverse=True)
    transcript=[json.loads(line) for line in (tmp_path/'model-transcript.jsonl').read_text().splitlines()]
    traces=[row for row in transcript if row['event']=='provider_thinking']
    assert [row['turn'] for row in traces]==[0,1,2,3]
    assert traces[2]['trace']['blocks'][0]['block']['thinking']=='trace-3'


def test_empty_final_is_retried_without_executing_reasoning(tmp_path,monkeypatch):
    from pairing.run_qwen_trials import ModelProtocolError
    model=object.__new__(c.EfficientQwen);model.messages=[];model.directory=tmp_path;model.calls=1
    requests=[]
    def ask(self,text,**kwargs):
        requests.append((text,kwargs))
        if len(requests)==1:raise ModelProtocolError('empty final')
        return {'operation':'observe','path':'','content':'','message':'Inspect'}
    monkeypatch.setattr(c.LearningQwen,'ask',ask)
    result=model.ask('{"observation":{"control_step":42}}',timeout=30,images=['original.jpg'])
    assert result['operation']=='observe'
    assert len(requests)==2
    retry=json.loads(requests[1][0]);assert retry['observation']['control_step']==42
    assert requests[1][1]['images']==()
    assert 0<requests[1][1]['timeout']<=30
    assert json.loads((tmp_path/'protocol-recovery.jsonl').read_text())['executed'] is False


def test_protocol_retry_is_bounded(tmp_path,monkeypatch):
    from pairing.run_qwen_trials import ModelProtocolError
    model=object.__new__(c.EfficientQwen);model.messages=[];model.directory=tmp_path;model.calls=2
    count=[]
    def ask(self,*args,**kwargs):
        count.append(1);raise ModelProtocolError('invalid')
    monkeypatch.setattr(c.LearningQwen,'ask',ask)
    with pytest.raises(ModelProtocolError):model.ask('{}',timeout=30)
    assert len(count)==2


def test_model_initialization_failure_finalizes_series_record(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    def fail(*args,**kwargs):raise RuntimeError('Existing Modal FDR Keychain credential unavailable; output suppressed')
    monkeypatch.setattr(c,'EfficientQwen',fail)
    monkeypatch.setattr(c,'CodeSandbox',lambda *a,**k:pytest.fail('Sandbox must not start after credential failure'))
    row={'id':'t01','task':'CloseMicrowave','config':{'robot':'PandaOmron'}}
    with pytest.raises(RuntimeError,match='credential unavailable'):c.series(row,'qwen','parent','test-image')
    records=list((tmp_path/'jobs').glob('*/job.json'));assert len(records)==1
    record=json.loads(records[0].read_text())
    assert record['status']=='infrastructure_error'
    assert record['finished_at']
    assert record['artifacts']['stderr']=='stderr.txt'
    assert 'report' not in record['artifacts']


def test_case_revision_is_explicit_and_cannot_change_robot(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    row={'id':'t22','task':'OpenDrawer','config':{'task':'robocasa/OpenDrawer','robot':'GR1','seed':0,'env_kwargs':{'layout_ids':3,'style_ids':1}}}
    (tmp_path/'case-revisions').mkdir()
    revised=json.loads(json.dumps(row['config']));revised['env_kwargs']['layout_ids']=2
    change={'original_config':row['config'],'config':revised,'revision_id':'layout-2-v1','reason':'Original scene failed initialization before any model run'}
    c.emit(tmp_path/'case-revisions/t22.json',change)
    actual=c.effective_row(row)
    assert row['config']['env_kwargs']['layout_ids']==3
    assert actual['config']['env_kwargs']['layout_ids']==2
    assert c.preflight_directory(actual)==tmp_path/'preflight/t22/layout-2-v1'
    change['config']['robot']='PandaOmron';c.emit(tmp_path/'case-revisions/t22.json',change)
    with pytest.raises(ValueError):c.effective_row(row)
def test_observe_feedback_does_not_embed_sensor_payloads():
    from pairing.cohort50 import compact_observation_feedback
    observation={'control_step':685,'sensors':{'joint':1},'camera_calibration':{'camera':1}}
    feedback={**observation,'images_png_base64':{'camera':'x'*2_000_000},
        'depth_npz_base64':'y'*5_000_000,'depth_arrays':[[1,2]],
        'controller_receipt':{'operation':'observe'}}
    assert compact_observation_feedback(feedback,observation)=={'controller_receipt':{'operation':'observe'}}
    assert 'depth_npz_base64' in feedback
    error={'error':'script failed','output':'measurement=0.12'}
    assert compact_observation_feedback(error,observation)==error
