import json
from pathlib import Path
import pytest
from pairing import adaptive_run as a


def plan(tmp_path,monkeypatch):
    root=tmp_path/'cohort';root.mkdir()
    monkeypatch.setattr(a.c,'ROOT',root)
    row={'id':'t01','task':'CloseMicrowave','config':{'robot':'PandaOmron','task':'robocasa/CloseMicrowave','seed':0,'env_kwargs':{}}}
    (root/'cohort.json').write_text(json.dumps({'tasks':[row]}))
    pre=root/'preflight'/'t01';pre.mkdir(parents=True)
    (pre/'receipt.json').write_text(json.dumps({'config':row['config'],'fingerprint':'fixed'}))
    config=dict(provider='openai',model='pinned',context_window=1_000_000,max_output_tokens=131_000)
    return a.build_plan('t01',config,code_image='code-image')


def test_default_five_trial_plan_has_explicit_protocol_not_upstream_defaults(tmp_path,monkeypatch):
    p=plan(tmp_path,monkeypatch)
    assert p['trials']==5 and p['limits']==a.LIMITS
    assert '5-attempt' in p['prompt']
    assert 'any compaction summary' not in p['prompt']
    assert p['context_policy']=='sqlite-sliding-v1'
    assert p['legacy_resume'] is False
    assert 'NOT claimed upstream' in p['budget_source']


def test_openrouter_run_refuses_before_job_or_inference(tmp_path,monkeypatch):
    p=plan(tmp_path,monkeypatch);p['participant']['provider']='openrouter'
    monkeypatch.setattr(a,'create',lambda *args:pytest.fail('must fail before creating job'))
    with pytest.raises(ValueError,match='counter'):a.run(p)


@pytest.mark.parametrize('archive_fails', [False, True])
def test_failed_public_attempt_preserves_library_and_archives_without_masking_error(tmp_path, monkeypatch, archive_fails):
    p=plan(tmp_path,monkeypatch)
    p.update(protocol='adaptive-public-sqlite-v1',trials=1,
             execution={'backend':'modal','simulator_image':'im-test'})
    jobs=tmp_path/'jobs';jobs.mkdir()
    archived=[]
    class Refusal(RuntimeError):
        status='model_refusal'
    class Model:
        thread_id='history';metadata={}
        def __init__(self):self.history_store=self
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def export(self,directory):(directory/'history.sqlite').write_bytes(b'test')
        def save_context(self,path):Path(path).write_text('test');self.export(Path(path).parent)
    class Sandbox:
        keeper='keeper'
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def archive(self,path):Path(path).write_bytes(b'partial-code');return str(path)
        checkpoint=archive
    class Episode:
        initial={'fingerprint':'fixed'}
        def __init__(self,*args,**kwargs):pass
        def close(self):return {'recording_dir':str(tmp_path)}
    def fail(*args,**kwargs):raise Refusal('provider declined')
    def archive(directory,execution):
        job=json.loads((directory/'job.json').read_text())
        assert job['finished_at'] and job['status']=='model_refusal'
        assert (directory/'history.sqlite').exists()
        assert (directory/'library-interrupted.zip').read_bytes()==b'partial-code'
        archived.append(directory)
        if archive_fails:raise RuntimeError('storage unavailable')
    monkeypatch.setattr(a,'create_participant',lambda *args,**kwargs:Model())
    monkeypatch.setattr(a.c,'CodeSandbox',Sandbox)
    monkeypatch.setattr(a.c,'FreshEpisode',Episode)
    monkeypatch.setattr(a.c,'EfficientRunner',lambda *args:object())
    monkeypatch.setattr(a.c,'drive',fail)
    monkeypatch.setattr(a.c,'publish_video',lambda *args,**kwargs:None)
    monkeypatch.setattr(a.c,'require_disk_headroom',lambda *args:None)
    monkeypatch.setattr(a.c,'serial_spec',lambda *args,**kwargs:{})
    monkeypatch.setattr('roboswe.adaptive_transport.archive_finalized_attempt',archive)
    with pytest.raises(Refusal,match='provider declined'):a.run(p,jobs_dir=jobs)
    assert len(archived)==1
    job=json.loads((archived[0]/'job.json').read_text())
    assert bool(job['artifact_errors']) is archive_fails
    assert job['artifacts']['interrupted_library']=='library-interrupted.zip'


def test_offline_trial_loop_retains_client_and_sandbox_and_exports(tmp_path,monkeypatch):
    p=plan(tmp_path,monkeypatch);p['trials']=2
    jobs=tmp_path/'jobs';jobs.mkdir()
    counters={'models':0,'workers':0,'drives':0,'closed':0}
    class Model:
        thread_id='same-local-history';metadata={'context_policy':'sqlite-sliding-v1'}
        def __init__(self):self.history_store=self
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def export(self,directory):(directory/'history.sqlite').write_bytes(b'offline-test-export')
        def save_context(self,path):Path(path).write_text('saved');self.export(Path(path).parent)
    def factory(*args,**kwargs):counters['models']+=1;return Model()
    class Sandbox:
        keeper='same-sandbox'
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def archive(self,path):Path(path).write_bytes(b'archive');return str(path)
    class Episode:
        def __init__(self,*args,**kwargs):counters['workers']+=1;self.initial={'fingerprint':'fixed'}
        def close(self):counters['closed']+=1;return {'recording_dir':str(tmp_path)}
    def drive(*args,**kwargs):
        counters['drives']+=1
        return dict(observed_success=True,termination_reason='sustained_success',control_steps=50,elapsed_seconds=3)
    monkeypatch.setattr(a,'create_participant',factory)
    monkeypatch.setattr(a.c,'CodeSandbox',Sandbox)
    monkeypatch.setattr(a.c,'FreshEpisode',Episode)
    monkeypatch.setattr(a.c,'EfficientRunner',lambda *args:object())
    monkeypatch.setattr(a.c,'drive',drive)
    monkeypatch.setattr(a.c,'publish_video',lambda *args,**kwargs:None)
    monkeypatch.setattr(a.c,'require_disk_headroom',lambda *args:None)
    monkeypatch.setattr(a.c,'serial_spec',lambda *args,**kwargs:{})
    directory=a.run(p,jobs_dir=jobs)
    assert counters=={'models':1,'workers':2,'drives':2,'closed':2}
    report=json.loads((directory/'report.json').read_text())
    assert len(report['trials'])==2
    assert json.loads((directory/'job.json').read_text())['status']=='completed_unscored'
    children=[x for x in jobs.iterdir() if x!=directory]
    assert all((x/'history.sqlite').exists() for x in children)
    assert all('history.sqlite' in json.loads((x/'job.json').read_text())['artifacts'] for x in children)


@pytest.mark.parametrize('completed_count',[1,2,4])
def test_certified_resume_restores_full_workspace_and_reuses_original_trial_two(tmp_path,monkeypatch,completed_count):
    p=plan(tmp_path,monkeypatch)
    p.update(protocol='adaptive-public-sqlite-v1',execution={'backend':'docker','simulator_image':'image'})
    jobs=tmp_path/'jobs';parent=jobs/'original-parent';prefix=jobs/'original-second'
    parent.mkdir(parents=True);prefix.mkdir()
    (prefix/'original-evidence.txt').write_text('preserve')
    (parent/'source.zip').write_bytes(b'original-source')
    record={'id':parent.name,'artifacts':{},'artifact_errors':[]}
    prefix_record={'id':prefix.name,'artifacts':{},'artifact_errors':[],'attempt_number':completed_count+1}
    put=lambda path,value:path.write_text(json.dumps(value))
    put(parent/'job.json',record);put(prefix/'job.json',prefix_record)
    class Stop(RuntimeError):pass
    restored=[];started=[]
    class Model:
        thread_id='original-thread';metadata={}
        def __init__(self):self.history_store=self
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def export(self,directory):(Path(directory)/'history.sqlite').write_bytes(b'current-history')
        def save_context(self,path):Path(path).write_text('context')
    class Sandbox:
        keeper='new-keeper'
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def restore_checkpoint(self,path):restored.append(path)
        def checkpoint(self,path):Path(path).write_bytes(b'full');return {'schema_version':'workspace-checkpoint-v2'}
    class Episode:
        initial={'fingerprint':'fixed'}
        def __init__(self,config,spec,directory,number,previous,**kwargs):
            assert restored==['full-checkpoint.zip']
            started.append((directory.name,number,len(previous)))
        def close(self):return {'recording_dir':str(tmp_path)}
    monkeypatch.setattr(a,'create_participant',lambda *a,**k:pytest.fail('Fresh model would lose retained context'))
    monkeypatch.setattr(a,'create',lambda *a,**k:pytest.fail('Do not allocate duplicate child or parent'))
    monkeypatch.setattr(a.c,'CodeSandbox',Sandbox)
    monkeypatch.setattr(a.c,'EfficientRunner',lambda *a:object())
    monkeypatch.setattr(a.c,'publish_video',lambda *a,**k:None)
    monkeypatch.setattr(a.c,'drive',lambda *a,**k:(_ for _ in ()).throw(Stop('bounded mock stop')))
    previous=[{'attempt_number':n,'observed_success':False} for n in range(1,completed_count+1)]
    state={'parent':str(parent),'record':record,'model':Model(),'reports':previous,'next_trial_number':completed_count+1,
           'receipt':'maintenance.json','prefix':str(prefix),'preserved_prefix':str(parent/'preserved-prefix'),
           'prefix_record':prefix_record,'workspace':'full-checkpoint.zip'}
    with pytest.raises(Stop):a.run(p,jobs_dir=jobs,episode_factory=Episode,resume_state=state)
    assert started==[('original-second',completed_count+1,completed_count)]
    assert (parent/'preserved-prefix/original-evidence.txt').read_text()=='preserve'
    assert json.loads((prefix/'job.json').read_text())['id']=='original-second'
    assert json.loads((prefix/'job.json').read_text())['campaign_id']=='original-parent'
    assert (parent/'source.zip').read_bytes()==b'original-source'
    assert (parent/'history.sqlite').read_bytes()==b'current-history'
