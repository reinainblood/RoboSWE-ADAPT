import hashlib
import json
from pathlib import Path

import pytest
from roboswe.admission import prepare, run
from roboswe.cli import main


def plan(tmp_path, backend='docker'):
    dataset=tmp_path/'dataset.json'
    dataset.write_text(json.dumps({'tasks':[{'id':'one','task':'BlendIngredients','config':{'task':'robocasa/BlendIngredients','robot':'PandaOmron','seed':0}}]}))
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'participant':{'provider':'openai','model':'unused','context_window':100000,'max_output_tokens':10000},'execution':{'backend':backend}}))
    images=tmp_path/'images.json'
    receipt={'backend':backend,'code_image':'sha256:'+'a'*64,'simulator_image':'sha256:'+'b'*64 if backend=='docker' else 'im-ABC123'}
    if backend=='modal':receipt.update(modal_profile='explicit',modal_environment='FDR')
    images.write_text(json.dumps(receipt))
    return prepare(dataset,config,images=images,output=tmp_path/'admitted.json')


def gateway(*, fingerprints=('same','same'), initial_valid=True, fail_step=False, fail_close=False, readback=True):
    state={'allocated':0,'closed':0,'active':0,'steps':0}
    class Runtime:
        def __init__(self,config,spec,directory,number,previous):
            assert state['active']==0
            self.initial={'fingerprint':fingerprints[state['allocated']]}
            state['allocated']+=1;state['active']+=1
            self.steps=0
            assert spec['simulator_lifetime_seconds']==1020
            assert spec['max_control_steps_per_task']==5
        def start_task(self,index):return {'precondition_valid':initial_valid,'observation':{'action_schema':{'shape':[12]}}}
        def bridge(self,message):
            if fail_step:raise RuntimeError('transport failure')
            assert message=={'operation':'step','action':[0.0]*12}
            self.steps+=1;state['steps']+=1
            return {'control_step':self.steps}
        def finish_task(self,**kwargs):return {'observed_success':False}
        def close(self):
            state['closed']+=1;state['active']-=1
            if fail_close:raise RuntimeError('cleanup unconfirmed')
            return {'remote_artifacts':{'persisted_readback_verified':readback}}
    return Runtime,state


def test_sequential_admission_preserves_source_and_provenance(tmp_path):
    p=plan(tmp_path);before=Path(p['dataset']).read_bytes()
    factory,state=gateway()
    result=run(p,factory=factory)
    assert result['admitted']==1 and state=={'allocated':2,'closed':2,'active':0,'steps':10}
    assert Path(p['dataset']).read_bytes()==before
    output=json.loads(Path(p['output']).read_text())
    assert output['tasks'][0]['initial_fingerprint']=='same'
    assert output['source_task_count']==output['admitted_task_count']==1
    assert output['source_planned_attempts']==output['admitted_planned_attempts']==5
    assert output['image_build_receipt']['sha256']==hashlib.sha256((tmp_path/'images.json').read_bytes()).hexdigest()
    assert len(list(Path(p['evidence']).rglob('admission-receipt.json')))==2
    with pytest.raises(ValueError,match='overwrite'):run(p,factory=factory)
    assert state['allocated']==2


@pytest.mark.parametrize('options,allocated',[
    ({'fingerprints':('one','two')},2),
    ({'initial_valid':False},1),
    ({'fail_step':True},1),
    ({'fail_close':True},1),
])
def test_failures_never_admit_or_retry_and_close_owned_runtime(tmp_path,options,allocated):
    p=plan(tmp_path);factory,state=gateway(**options)
    result=run(p,factory=factory)
    assert result['admitted']==0
    assert state['allocated']==state['closed']==allocated
    assert json.loads(Path(p['output']).read_text())['tasks']==[]
    output=json.loads(Path(p['output']).read_text())
    assert output['source_task_count']==1 and output['admitted_task_count']==0
    assert output['source_planned_attempts']==5 and output['admitted_planned_attempts']==0
    assert json.loads(Path(p['rejected_output']).read_text())['tasks'][0]['status']=='environment_admission_rejected_not_model_outcome'


def test_modal_requires_readback(tmp_path):
    p=plan(tmp_path,'modal');factory,state=gateway(readback=False)
    assert run(p,factory=factory)['stopped_on_infrastructure_failure']
    assert state['allocated']==1


def test_constructor_failure_stops_batch(tmp_path):
    p=plan(tmp_path)
    p['tasks'].append(p['tasks'][0])
    called=[]
    def bad(*args):
        called.append(True)
        raise TimeoutError('startup unknown')
    result=run(p,factory=bad)
    assert len(called)==1 and result['rejected_or_unattempted']==2


def test_admission_plan_does_not_import_actor_adapters(tmp_path,monkeypatch):
    import builtins
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith(('pairing.cohort50','pairing.direct_models','pairing.anthropic_model','pairing.openrouter_model')):
            pytest.fail('Actor module imported by admission planning: '+name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    p=plan(tmp_path)
    assert p['tasks'][0]['prompt'] is None
    assert not Path(p['evidence']).exists()


def test_cli_dry_plan_never_allocates(tmp_path,monkeypatch,capsys):
    p=plan(tmp_path)
    import roboswe.admission
    monkeypatch.setattr(roboswe.admission,'run',lambda _:pytest.fail('allocated'))
    assert main(['admit',p['dataset'],'--config',str(tmp_path/'config.json'),'--images',str(tmp_path/'images.json'),'-e','docker','--out',p['output'],'--dry-run'])==0
    assert json.loads(capsys.readouterr().out)['model_calls']==0
    assert not Path(p['output']).exists()


def test_pinned_fingerprint_is_not_silently_replaced(tmp_path):
    p=plan(tmp_path)
    p['tasks'][0]['cohort']['initial_fingerprint']='pinned'
    factory,state=gateway()
    result=run(p,factory=factory)
    assert result['admitted']==0 and state['allocated']==1 and state['steps']==0


@pytest.mark.parametrize('reserved',['output','evidence','rejected_output'])
def test_existing_output_or_receipts_block_allocation(tmp_path,reserved):
    p=plan(tmp_path)
    target=Path(p[reserved])
    if reserved=='evidence':target.mkdir()
    else:target.write_text('preserve')
    factory,state=gateway()
    with pytest.raises(ValueError,match='overwrite'):run(p,factory=factory)
    assert state['allocated']==0
    if reserved!='evidence':assert target.read_text()=='preserve'


def test_successful_modal_admission_requires_both_readbacks(tmp_path):
    p=plan(tmp_path,'modal');factory,state=gateway()
    assert run(p,factory=factory)['admitted']==1
    receipts=[json.loads(f.read_text()) for f in Path(p['evidence']).rglob('admission-receipt.json')]
    assert len(receipts)==2 and all(r['close']['remote_artifacts']['persisted_readback_verified'] for r in receipts)
