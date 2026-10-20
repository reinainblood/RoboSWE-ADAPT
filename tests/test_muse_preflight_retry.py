import json
import pytest
from pairing import muse_preflight_retry as m

@pytest.mark.parametrize('actual,passed',[('original',True),('different',False)])
def test_exact_original_target_only(tmp_path,monkeypatch,actual,passed):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    folder=tmp_path/'preflight'/'t14';folder.mkdir(parents=True)
    original={'same_reset':False,'fingerprints':['original','rejected'],'50_steps_seconds':[10,12],'config':{},'passes_execution_margin':True}
    (folder/'receipt.json').write_text(json.dumps(original))
    monkeypatch.setattr(m,'task_row',lambda task:{'config':{'robot':'PandaOmron'}})
    monkeypatch.setattr(m,'spec_for',lambda row:{})
    calls=[]
    class Remote:
        def __init__(self,*a,**k):self.initial={'fingerprint':actual}
        def start_task(self,n):return {'precondition_valid':True,'observation':{'action_schema':{'shape':[2]}}}
        def bridge(self,msg):calls.append(msg)
        def close(self):pass
    monkeypatch.setattr(m,'RemoteEpisode',Remote)
    result=m.retry('t14');assert result['passed']==passed
    saved=json.loads((folder/'receipt.json').read_text())
    assert json.loads((folder/'receipt-before-exact-retry.json').read_text())==original
    assert len(calls)==(50 if passed else 0)
    if passed:assert saved['fingerprints']==['original','original'] and saved['admitted_reset_indices']==[1,3]
    else:assert saved==original
    with pytest.raises(ValueError):m.retry('t14')


@pytest.mark.parametrize('times,index', [([73.8,76.3],1),([76.0,67.0],2)])
def test_throughput_retry_keeps_original_target_and_margin(tmp_path,monkeypatch,times,index):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    folder=tmp_path/'preflight'/'t22';folder.mkdir(parents=True)
    original={'same_reset':True,'fingerprints':['original','original'],'50_steps_seconds':times,'passes_execution_margin':False}
    (folder/'receipt.json').write_text(json.dumps(original))
    monkeypatch.setattr(m,'task_row',lambda task:{'config':{'robot':'GR1'}})
    monkeypatch.setattr(m,'spec_for',lambda row:{})
    class Remote:
        def __init__(self,*a,**k):self.initial={'fingerprint':'original'}
        def start_task(self,n):return {'precondition_valid':True,'observation':{'whole_body_ik':{'current_pose_hold_action':[0]}}}
        def bridge(self,msg):pass
        def close(self):pass
    monkeypatch.setattr(m,'RemoteEpisode',Remote)
    assert m.retry('t22')['passed']
    saved=json.loads((folder/'receipt.json').read_text())
    assert saved['50_steps_seconds'][0]==times[index-1]
    assert saved['admitted_reset_indices']==[index,3]
    assert saved['fingerprints']==['original','original']
    assert json.loads((folder/'receipt-before-exact-retry.json').read_text())==original
