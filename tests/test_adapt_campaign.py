import json
import pytest
from pairing.adapt_campaign import main


def test_supervisor_is_bounded_and_refuses_relaunch(tmp_path,monkeypatch):
    task={'cohort':{'id':'task'},'participant':{'provider':'openrouter','model':'pinned'},'context_policy':'policy'}
    monkeypatch.setattr('roboswe.adaptive.build_plan',lambda *a,**kw:{'tasks':[task]})
    finished=tmp_path/'finished';finished.mkdir();(finished/'job.json').write_text(json.dumps({'id':'done','status':'completed_unscored'}))
    monkeypatch.setattr('roboswe.adaptive.run',lambda plan:[str(finished)])
    monkeypatch.setattr('pairing.muse_model.credential',lambda:'test-key')
    argv=['--dataset','dataset','--config','config','--state-dir',str(tmp_path)]
    main(argv)
    state=json.loads((tmp_path/'active.json').read_text())
    assert state['status']=='completed' and state['completed']==[str(finished)]
    assert 'test-key' not in (tmp_path/'active.json').read_text()
    with pytest.raises(RuntimeError,match='implicit restart'):main(argv)


def test_failure_stops_without_retry(tmp_path,monkeypatch):
    task={'cohort':{'id':'task'},'participant':{'provider':'openrouter','model':'pinned'},'context_policy':'policy'}
    calls=[]
    monkeypatch.setattr('roboswe.adaptive.build_plan',lambda *a,**kw:{'tasks':[task,task]})
    def fail(plan):calls.append(1);raise RuntimeError('failed')
    monkeypatch.setattr('roboswe.adaptive.run',fail)
    monkeypatch.setattr('pairing.muse_model.credential',lambda:'test-key')
    with pytest.raises(RuntimeError,match='failed'):
        main(['--dataset','d','--config','c','--state-dir',str(tmp_path),'--max-series','2'])
    assert calls==[1]
    assert json.loads((tmp_path/'active.json').read_text())['status']=='stopped_error'


@pytest.mark.parametrize('status',['infrastructure_error','model_refusal','verification_invalid'])
def test_returned_failed_job_stops_lane_not_completed(tmp_path,monkeypatch,status):
    task={'cohort':{'id':'task'},'participant':{'provider':'openrouter','model':'pinned'},'context_policy':'policy'}
    failed=tmp_path/'failed';failed.mkdir();(failed/'job.json').write_text(json.dumps({'id':'failed','status':status}))
    calls=[]
    monkeypatch.setattr('roboswe.adaptive.build_plan',lambda *a,**kw:{'tasks':[task,task]})
    def result(plan):calls.append(1);return [str(failed)]
    monkeypatch.setattr('roboswe.adaptive.run',result)
    monkeypatch.setattr('pairing.muse_model.credential',lambda:'test-key')
    main(['--dataset','d','--config','c','--state-dir',str(tmp_path),'--max-series','2'])
    state=json.loads((tmp_path/'active.json').read_text())
    assert calls==[1] and state['completed']==[] and state['completed_series_count']==0
    assert state['status']=='stopped_child_result' and state['child_status']==status
