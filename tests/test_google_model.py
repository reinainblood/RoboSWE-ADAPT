import io
import json
import pytest
from pairing.google_model import GoogleLearningModel


def model(tmp_path,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','private-test-key')
    return GoogleLearningModel('policy',tmp_path,model='gemini-4-argon',context_window=1048576,
        max_output_tokens=32768,ledger_path=tmp_path/'spend.json')


def response():
    decision={'operation':'observe','path':'','content':'','message':'inspect'}
    return {'candidates':[{'finishReason':'STOP','content':{'role':'model','parts':[
        {'thought':True,'text':'returned thought summary','thoughtSignature':'opaque'},
        {'text':json.dumps(decision),'thoughtSignature':'output-signature'}]}}],
        'usageMetadata':{'promptTokenCount':100,'candidatesTokenCount':20,'thoughtsTokenCount':30,'totalTokenCount':150},
        'modelVersion':'gemini-4-argon'}


def test_native_count_and_generation_same_request_thinking_archive_budget(tmp_path,monkeypatch):
    calls=[]
    def post(req,timeout):
        body=json.loads(req.data);calls.append((req.full_url,body))
        return io.BytesIO(json.dumps({'totalTokens':100} if req.full_url.endswith('countTokens') else response()).encode())
    monkeypatch.setattr('urllib.request.urlopen',post)
    m=model(tmp_path,monkeypatch)
    assert m.ask('hello')['operation']=='observe'
    counted=calls[-2][1]['generateContentRequest']
    generated=calls[-1][1]
    assert {k:v for k,v in counted.items() if k!='model'}==generated
    assert m.messages[-1]['parts'][0]['thoughtSignature']=='opaque'
    assert 'returned thought summary' in (tmp_path/'provider-thinking.jsonl').read_text()
    ledger=m.ledger.snapshot()
    assert ledger['spent_usd']==pytest.approx(.0007)
    assert ledger['reserved_usd']==0
    assert 'private-test-key' not in (tmp_path/'provider-responses.jsonl').read_text()
    assert (tmp_path/'history.sqlite').exists()


def test_uncertain_generation_keeps_reservation(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch)
    monkeypatch.setattr(m.counter,'count_native_request',lambda *a,**k:100)
    def broken(*a):raise RuntimeError('unknown transport')
    monkeypatch.setattr(m,'_generate',broken)
    with pytest.raises(RuntimeError,match='unknown transport'):m.ask('hello')
    assert m.ledger.snapshot()['reserved_usd']>0
    assert m.calls==0


def test_missing_usage_never_executes_and_keeps_full_receipt(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch)
    monkeypatch.setattr(m.counter,'count_native_request',lambda *a,**k:100)
    payload=response();payload.pop('usageMetadata')
    monkeypatch.setattr(m,'_generate',lambda *a:payload)
    with pytest.raises(RuntimeError,match='usage missing'):m.ask('hello')
    assert m.ledger.snapshot()['reserved_usd']>0
    assert (tmp_path/'provider-responses.jsonl').exists()


def test_public_google_plan_accepts_native_provider(tmp_path,monkeypatch):
    from roboswe.adaptive import build_plan
    dataset=tmp_path/'dataset.json';config=tmp_path/'config.json'
    dataset.write_text(json.dumps({'tasks':[{'id':'x','task':'BlendIngredients','config':{'task':'robocasa/BlendIngredients','robot':'PandaOmron','seed':0}}]}))
    config.write_text(json.dumps({'participant':{'provider':'google','model':'gemini-4-argon','context_window':1048576,'max_output_tokens':32768,'ledger_path':str(tmp_path/'spend.json'),'spend_limit_usd':125},
        'execution':{'simulator_image':'sha256:'+'a'*64,'code_image':'sha256:'+'b'*64}}))
    plan=build_plan(dataset,config)
    assert plan['tasks'][0]['participant']['api_key_env']=='GEMINI_API_KEY'
    assert plan['tasks'][0]['context_policy']=='sqlite-sliding-v1'


def test_blocked_prompt_preserves_input_usage_and_reservation(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch)
    monkeypatch.setattr(m.counter,'count_native_request',lambda *a,**k:100)
    payload={'promptFeedback':{'blockReason':'OTHER'},
             'usageMetadata':{'promptTokenCount':100,'totalTokenCount':100}}
    calls=[]
    def blocked(*args):
        calls.append(1)
        return payload
    monkeypatch.setattr(m,'_generate',blocked)
    with pytest.raises(RuntimeError,match='provider blocked response'):m.ask('hello')
    assert calls==[1]
    assert m.ledger.snapshot()['reserved_usd']>0
    assert m.ledger.snapshot()['spent_usd']==0
    assert json.loads((tmp_path/'provider-responses.jsonl').read_text())['response']==payload
    assert not m.messages


def test_quota_diagnostic_redacts_key_and_preserves_reason():
    from pairing.google_model import sanitize_google_error
    key='private-secret'
    raw={'error':{'code':429,'status':'RESOURCE_EXHAUSTED','message':'quota '+key,
        'details':[{'@type':'type.googleapis.com/google.rpc.QuotaFailure','violations':[{'quotaMetric':'free_tier_requests','quotaValue':'0','arbitrary':'should not persist'}]}]}}
    result=sanitize_google_error(io.BytesIO(json.dumps(raw).encode()),key)
    assert result['status']=='RESOURCE_EXHAUSTED'
    assert result['details'][0]['violations'][0]['quotaValue']=='0'
    assert key not in json.dumps(result)
    assert 'arbitrary' not in json.dumps(result)
