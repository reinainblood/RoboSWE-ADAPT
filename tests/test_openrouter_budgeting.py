import json
import pytest
from pairing.openrouter_budgeting import ConservativeCounter,POLICY
from pairing.openrouter_model import OpenRouterLearningModel


def body():
    return {'messages':[{'role':'user','content':[{'type':'text','text':'hello'},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,AAAA'}}]}], 'response_format':{}}


def test_image_reserve_and_upward_only_calibration(tmp_path):
    c=ConservativeCounter(tmp_path)
    count=c.count_body(body());assert count>=16384+8192
    c.observe(count*2,estimated=count,call=1)
    assert c.multiplier==2.5
    c.observe(1,estimated=count,call=2)
    assert c.multiplier==2.5
    assert '"exact_native_count": false' in (tmp_path/'context-estimates.jsonl').read_text()


def model(tmp_path,monkeypatch):
    monkeypatch.setattr('pairing.openrouter_model.credential',lambda:'private-test')
    return OpenRouterLearningModel('policy',tmp_path,model='google/gemini-4-argon',context_window=1048576,
        max_output_tokens=32768,context_policy=POLICY,ledger_path=tmp_path/'ledger.json',spend_limit_usd=150)


def payload():
    decision={'operation':'observe','path':'','content':'','message':'inspect'}
    return {'model':'google/gemini-4-argon','choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(decision)}}],
        'usage':{'prompt_tokens':100,'completion_tokens':20,'cost':.0004}}


def test_receipted_request_settles_cap_and_archives_approximate_policy(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_post',lambda *a,**k:payload())
    assert m.ask('hello')['operation']=='observe'
    assert m.ledger.snapshot()['spent_usd']==.0004
    assert m.ledger.snapshot()['reserved_usd']==0
    assert m.history_store.metadata['method']==POLICY
    assert (tmp_path/'context-estimates.jsonl').exists()


def test_unknown_receipt_retains_reservation_no_action(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch);p=payload();p['usage'].pop('cost')
    monkeypatch.setattr(m,'_post',lambda *a,**k:p)
    with pytest.raises(RuntimeError,match='receipt missing'):m.ask('hello')
    assert m.ledger.snapshot()['reserved_usd']>0


def test_second_body_count_time_is_subtracted_before_inference(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch);clock=[0.0];timeouts=[]
    monkeypatch.setattr('pairing.openrouter_model.time.monotonic',lambda:clock[0])
    def assembled(body):clock[0]+=3;return 100
    monkeypatch.setattr(m.approximate_counter,'count_body',assembled)
    def post(*a,timeout):timeouts.append(timeout);return payload()
    monkeypatch.setattr(m,'_post',post)
    assert m.ask('hello',timeout=10)['operation']=='observe'
    assert timeouts==[7]


def test_exhausted_body_count_never_reserves_or_generates(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch);clock=[0.0]
    monkeypatch.setattr('pairing.openrouter_model.time.monotonic',lambda:clock[0])
    def assembled(body):clock[0]+=11;return 100
    monkeypatch.setattr(m.approximate_counter,'count_body',assembled)
    monkeypatch.setattr(m,'_post',lambda *a,**kw:pytest.fail('Inference must not be sent'))
    with pytest.raises(TimeoutError,match='original request deadline'):m.ask('hello',timeout=10)
    assert m.ledger.snapshot()['reserved_usd']==0
