import json
from pathlib import Path

import pytest
from pairing import adapt_next_pairs as q


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def series(tmp_path, name='series'):
    jobs = tmp_path / 'jobs'
    parent = jobs / name
    put(parent / 'job.json', {'id':name,'status':'completed_unscored','finished_at':'now'})
    put(parent / 'report.json', {'trials':[{}]*5})
    for n in range(1,6):
        d=jobs/f'{name}-{n}'
        put(d/'job.json', {'id':d.name,'campaign_id':name,'kind':'adaptive-direct-attempt','attempt_number':n,'status':'verified_fail','finished_at':'now'})
        put(d/'verification.json', {'kind':'independent_code_replay','reproduced':True,'mismatches':[],'verified_success':False,'trials':[{'reproduced':True,'infrastructure_valid':True,'precondition_valid':True,'observed_success':False,'verified_success':False,'replay':{'precondition_valid':True,'observed_success':False}}]})
    return parent,jobs


def plan(tmp_path):
    root=tmp_path/'campaign';root.mkdir()
    dataset=put(tmp_path/'dataset.json',{'tasks':[{'task':'BeverageSorting','initial_fingerprint':'abc','config':{'task':'robocasa/BeverageSorting'}}]})
    configs={}
    for provider in q.PROVIDERS:
        configs[provider]=str(put(tmp_path/f'{provider}.json',{'participant':{**q.PINS[provider],'ledger_path':str(root/(provider+'-spend.json'))},'trials':5,'limits':{'active_seconds':1800},'execution':{'backend':'modal','modal_environment':'FDR','modal_profile':'snorkelai','artifact_transfer':'remote-video'},'video_judge':None}))
    path=put(tmp_path/'plan.json',{'pairs':[{'id':'beverage-sorting','dataset':str(dataset),'configs':configs}]})
    return path,root,configs


def test_failed_model_trials_are_valid_complete_series(tmp_path):
    parent,jobs=series(tmp_path)
    assert q.validate_series(parent,jobs)=='series'


@pytest.mark.parametrize('damage',['missing_child','replay_mismatch','contradiction','artifact_error','unfinished','bad_precondition'])
def test_parent_completion_does_not_hide_invalid_children(tmp_path,damage):
    parent,jobs=series(tmp_path);child=jobs/'series-3'
    if damage=='missing_child': (child/'job.json').unlink()
    elif damage in {'artifact_error','unfinished'}:
        x=q.read(child/'job.json');x.update({'artifact_errors':['bad']} if damage=='artifact_error' else {'finished_at':None});put(child/'job.json',x)
    else:
        x=q.read(child/'verification.json')
        if damage=='replay_mismatch':x['mismatches']=['drift']
        if damage=='contradiction':x['verified_success']=True
        if damage=='bad_precondition':x['trials'][0]['replay']['precondition_valid']=False
        put(child/'verification.json',x)
    with pytest.raises(ValueError):q.validate_series(parent,jobs)


def test_lane_verified_fail_is_completion_not_infra(tmp_path):
    parent,jobs=series(tmp_path);lane=tmp_path/'lane'
    put(lane/'active.json',{'status':'completed','finished_epoch':1,'completed':[str(parent)]})
    assert q.lane_result('openrouter',lane,0,jobs)['classification']=='completed'


def test_lane_budget_block_classified_without_retry(tmp_path):
    lane=tmp_path/'lane';put(lane/'active.json',{'status':'stopped_error','error':'campaign spend guard reached','finished_epoch':1})
    assert q.lane_result('openrouter',lane,1,tmp_path/'jobs')['classification']=='provider_blocked'


def test_plan_rejects_new_budget_or_model_or_unmatched_limits(tmp_path):
    path,root,configs=plan(tmp_path);pairs,hashes=q.validate_plan(path,root);assert len(pairs)==1 and len(hashes)==4
    original=q.read(configs['openrouter'])
    for change in ['ledger','model','limits']:
        x=json.loads(json.dumps(original))
        if change=='ledger':x['participant']['ledger_path']=str(tmp_path/'fresh.json')
        if change=='model':x['participant']['model']='substitute'
        if change=='limits':x['limits']['active_seconds']=900
        put(Path(configs['openrouter']),x)
        with pytest.raises(ValueError):q.validate_plan(path,root)


def test_predecessor_requires_owned_terminal_receipt(tmp_path,monkeypatch):
    parent,jobs=series(tmp_path);path=tmp_path/'previous.json'
    value={'pid':99,'finished_epoch':1,'status':'partial_stopped','results':[{'provider':'anthropic','classification':'provider_blocked'},{'provider':'openrouter','classification':'completed','completed':[str(parent)]}]}
    put(path,value);monkeypatch.setattr(q,'pid_live',lambda pid:False)
    assert q.predecessor_blocked(path,99,jobs)=={'anthropic'}
    with pytest.raises(ValueError):q.predecessor_blocked(path,98,jobs)
    value['finished_epoch']=None;put(path,value)
    with pytest.raises(ValueError):q.predecessor_blocked(path,99,jobs)


def test_stop_prevents_any_launch_while_waiting(tmp_path,monkeypatch):
    path,root,_=plan(tmp_path);folder=tmp_path/'state';folder.mkdir();(folder/'STOP').touch()
    monkeypatch.setattr(q,'pid_live',lambda pid:True)
    monkeypatch.setattr(q.subprocess,'Popen',lambda *a,**kw:pytest.fail('STOP must prevent paid launch'))
    out=q.main(['--plan',str(path),'--state-dir',str(folder),'--campaign-root',str(root),'--predecessor-state',str(tmp_path/'absent.json'),'--predecessor-pid','99'])
    assert out['status']=='stopped' and out['children']==[]
    with pytest.raises(RuntimeError,match='Existing queue receipt'):
        q.main(['--plan',str(path),'--state-dir',str(folder),'--campaign-root',str(root),'--predecessor-state',str(tmp_path/'absent.json'),'--predecessor-pid','99'])


def test_cleanup_signals_only_owned_live_processes(monkeypatch):
    class Process:
        def __init__(self,alive):self.alive=alive;self.pid=123;self.signals=[]
        def poll(self):return None if self.alive else 0
        def send_signal(self,s):self.signals.append(s)
        def wait(self,timeout):self.alive=False
    live,done=Process(True),Process(False)
    assert q.cleanup([('openrouter',live),('anthropic',done)])==[]
    assert len(live.signals)==1 and done.signals==[]


@pytest.mark.parametrize('failure,expected_launches',[(None,4),('provider_blocked',3),('blocked_infrastructure',2)])
def test_pairs_are_bounded_and_failures_stop_correct_scope(tmp_path,monkeypatch,failure,expected_launches):
    path,root,configs=plan(tmp_path)
    data=q.read(path)
    second=put(tmp_path/'second.json',{'tasks':[{'task':'DateNight','initial_fingerprint':'def','config':{'task':'robocasa/DateNight'}}]})
    data['pairs'].append({'id':'date-night','dataset':str(second),'configs':configs});put(path,data)
    monkeypatch.setattr(q,'pid_live',lambda pid:False)
    monkeypatch.setattr(q,'predecessor_blocked',lambda *args:set())
    monkeypatch.setattr(q,'live_simulators',lambda:False)
    monkeypatch.setenv('GEMINI_API_KEY','test-count-key')
    launched=[];active=[];events=[]
    class Process:
        def __init__(self,command,**kwargs):
            # The next task cannot launch until both preceding processes terminated.
            if len(launched)==2:assert all(p.done for p in active)
            self.pid=100+len(launched);self.done=False;self.signals=[]
            self.provider='openrouter' if 'openrouter' in command[command.index('--config')+1] else 'anthropic'
            launched.append(self);active.append(self)
        def poll(self):self.done=True;return 0
        def send_signal(self,s):self.signals.append(s);self.done=True
        def wait(self,timeout):self.done=True
    monkeypatch.setattr(q.subprocess,'Popen',Process)
    def result(provider,lane,code,jobs):
        category=failure if failure and provider=='openrouter' and lane.parent.name=='beverage-sorting' else 'completed'
        events.append((lane.parent.name,provider,category))
        return {'provider':provider,'classification':category,'status':'test'}
    monkeypatch.setattr(q,'lane_result',result)
    out=q.main(['--plan',str(path),'--state-dir',str(tmp_path/'state'),'--campaign-root',str(root),'--predecessor-state',str(tmp_path/'previous.json'),'--predecessor-pid','99'])
    assert len(launched)==expected_launches
    assert out['status']==('completed' if failure is None else 'stopped' if failure=='blocked_infrastructure' else 'partial_provider_blocked')
