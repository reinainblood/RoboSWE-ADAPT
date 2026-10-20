import json
from contextlib import contextmanager
import pytest
from pairing import muse_model as m


def test_exact_model_and_bounded_protocol_recovery(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'credential',lambda:'test-key')
    requests=[]
    @contextmanager
    def open_request(request,timeout):
        from io import StringIO
        payload=json.loads(request.data);requests.append(payload)
        content='Preface '+json.dumps({'operation':'finish','path':'','content':'','message':'done'}) if len(requests)==1 else json.dumps({'operation':'finish','path':'','content':'','message':'done'})
        yield StringIO(json.dumps({'choices':[{'message':{'role':'assistant','content':content,'reasoning_details':[{'type':'reasoning.text','text':'test rationale'}]}}]}))
    monkeypatch.setattr(m.urllib.request,'urlopen',open_request)
    model=m.MuseModel('instructions',tmp_path,context_policy='legacy')
    result=model.ask('{}')
    assert result['operation']=='finish'
    assert len(requests)==2
    assert all(r['model']=='meta/muse-spark-1.3-contributor' for r in requests)
    assert requests[1]['messages'][-2]['reasoning_details']
    assert 'No operation executed' in requests[1]['messages'][-1]['content'][0]['text']
    assert json.loads((tmp_path/'protocol-recovery.jsonl').read_text())['executed'] is False


def test_never_extracts_code_from_reasoning(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'credential',lambda:'test-key')
    @contextmanager
    def bad(request,timeout):
        from io import StringIO
        yield StringIO(json.dumps({'choices':[{'message':{'role':'assistant','content':'not JSON','reasoning':'{"operation":"execute"}'}}]}))
    monkeypatch.setattr(m.urllib.request,'urlopen',bad)
    model=m.MuseModel('instructions',tmp_path,context_policy='legacy')
    with pytest.raises(m.ModelProtocolError):model.ask('{}')
    assert model.calls==2


def test_native_function_arguments_are_used_without_rewriting_code(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'credential',lambda:'test-key')
    source='print("quoted")\n# literal backslash: \\n'
    decision={'operation':'write_execute','path':'motion.py','content':source,'message':'bounded probe'}
    @contextmanager
    def response(request,timeout):
        from io import StringIO
        payload=json.loads(request.data)
        assert payload['tool_choice']=='auto'
        assert payload['tools'][0]['function']['parameters']==m.SCHEMA
        yield StringIO(json.dumps({'choices':[{'message':{'role':'assistant','content':'Optional provider preface','tool_calls':[{'id':'call_test','type':'function','function':{'name':'robot_decision','arguments':json.dumps(decision)}}]}}]}))
    monkeypatch.setattr(m.urllib.request,'urlopen',response)
    model=m.MuseModel('instructions',tmp_path,context_policy='legacy')
    assert model.ask('{}')==decision
    assert json.loads(model.messages[-1]['tool_calls'][0]['function']['arguments'])['content']==source
    assert model.pending_call=='call_test'


def test_compaction_preserves_archived_history_and_uses_saved_summary(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'credential',lambda:'test-key')
    model=m.MuseModel('instructions',tmp_path,context_policy='legacy')
    model.messages.append({'role':'user','content':'x'*8_000_001})
    calls=[]
    def ask(text,**kwargs):
        calls.append(text)
        return {'operation':'finish','path':'','content':'','message':'Verified script order: approach.py then grasp.py. Scene has reset.'}
    monkeypatch.setattr(model,'_ask',ask)
    model.ask('fresh feedback',timeout=60)
    assert len(calls)==2 and model.compactions==1
    assert (tmp_path/'context-before-compaction-1.json.gz').exists()
    assert 'approach.py then grasp.py' in model.messages[1]['content']


def test_transient_model_not_found_retries_same_request(tmp_path,monkeypatch):
    from io import BytesIO,StringIO
    monkeypatch.setattr(m,'credential',lambda:'test-key')
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    requests=[]
    @contextmanager
    def endpoint(request,timeout):
        requests.append(request.data)
        if len(requests)==1:raise m.urllib.error.HTTPError(request.full_url,404,'provider',{},BytesIO(b'{"error":"model_not_found"}'))
        yield StringIO(json.dumps({'choices':[{'message':{'role':'assistant','content':json.dumps({'operation':'finish','path':'','content':'','message':'done'})}}]}))
    monkeypatch.setattr(m.urllib.request,'urlopen',endpoint)
    model=m.MuseModel('instructions',tmp_path,context_policy='legacy')
    assert model.ask('{}')['operation']=='finish'
    assert len(requests)==2 and requests[0]==requests[1]
    assert model.calls==1


def test_legacy_resume_keeps_original_history_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'credential',lambda:'test-key')
    decision={'operation':'observe','path':'','content':'','message':'inspect'}
    @contextmanager
    def response(request,timeout):
        from io import StringIO
        yield StringIO(json.dumps({'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':'legacy_call','type':'function','function':{'name':'robot_decision','arguments':json.dumps(decision)}}]}}]}))
    monkeypatch.setattr(m.urllib.request,'urlopen',response)
    model=m.MuseModel('instructions',tmp_path,context_policy='legacy',decision_transport='single-function-auto-validated-v1')
    assert model.ask('{}')==decision
    assert model.pending_call is None
    assert 'tool_calls' not in model.messages[-1]
    assert json.loads(model.messages[-1]['content'])==decision


def test_provider_retry_after_is_respected_within_budget(tmp_path,monkeypatch):
    from io import BytesIO,StringIO
    monkeypatch.setattr(m,'credential',lambda:'test-key')
    sleeps=[];monkeypatch.setattr(m.time,'sleep',sleeps.append)
    calls=[]
    @contextmanager
    def endpoint(request,timeout):
        calls.append(request.data)
        if len(calls)==1:raise m.urllib.error.HTTPError(request.full_url,503,'overloaded',{'Retry-After':'60'},BytesIO(b'{"error":"service_overloaded"}'))
        yield StringIO(json.dumps({'choices':[{'message':{'role':'assistant','content':json.dumps({'operation':'finish','path':'','content':'','message':'done'})}}]}))
    monkeypatch.setattr(m.urllib.request,'urlopen',endpoint)
    assert m.MuseModel('instructions',tmp_path,context_policy='legacy').ask('{}',timeout=120)['operation']=='finish'
    assert sleeps==[60] and calls[0]==calls[1]


def test_only_temporary_inflight_402_is_retried(tmp_path,monkeypatch):
    from io import BytesIO,StringIO
    monkeypatch.setattr(m,'credential',lambda:'test-key')
    sleeps=[];monkeypatch.setattr(m.time,'sleep',sleeps.append)
    calls=[]
    @contextmanager
    def endpoint(request,timeout):
        calls.append(request.data)
        if len(calls)==1:
            body=json.dumps({'error':{'metadata':{'reason':'in_flight_budget_exhausted','retry_after_seconds':120}}}).encode()
            raise m.urllib.error.HTTPError(request.full_url,402,'temporary',{},BytesIO(body))
        yield StringIO(json.dumps({'choices':[{'message':{'role':'assistant','content':json.dumps({'operation':'finish','path':'','content':'','message':'done'})}}]}))
    monkeypatch.setattr(m.urllib.request,'urlopen',endpoint)
    assert m.MuseModel('instructions',tmp_path,context_policy='legacy').ask('{}',timeout=180)['operation']=='finish'
    assert sleeps==[60,60] and calls[0]==calls[1]


def test_real_credit_exhaustion_does_not_retry(tmp_path,monkeypatch):
    from io import BytesIO
    monkeypatch.setattr(m,'credential',lambda:'test-key')
    sleeps=[];monkeypatch.setattr(m.time,'sleep',sleeps.append)
    def endpoint(request,timeout):
        raise m.urllib.error.HTTPError(request.full_url,402,'credits',{},BytesIO(b'{"error":{"metadata":{"reason":"insufficient_credits"}}}'))
    monkeypatch.setattr(m.urllib.request,'urlopen',endpoint)
    import pytest
    with pytest.raises(RuntimeError,match='HTTP 402'):m.MuseModel('instructions',tmp_path,context_policy='legacy').ask('{}',timeout=180)
    assert sleeps==[]
