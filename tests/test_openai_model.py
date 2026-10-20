import json
import pytest
from pairing.openai_model import API_URL,OpenAILearningModel


def model(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','test-secret')
    return OpenAILearningModel('fixed',tmp_path,model='pinned-model',context_window=1_000_000,max_output_tokens=131_000)


def response(operation='finish',**extra):
    return {'model':'pinned-model','status':'completed','output':[
        {'type':'reasoning','id':'r1','summary':[{'type':'summary_text','text':'provider summary'}],'encrypted_content':'opaque'},
        {'type':'message','id':'m1','role':'assistant','status':'completed','content':[
            {'type':'output_text','text':json.dumps(dict(operation=operation,path='',content='',message='done')),'annotations':[]}]}],
        'usage':{'input_tokens':50,'output_tokens':10,'total_tokens':60},**extra}


def test_direct_stateless_replay_and_exact_count_payload(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch); requests=[]
    def post(url,body,**kwargs):
        requests.append((url,body))
        if url.endswith('/input_tokens'):return {'input_tokens':50}
        return response()
    monkeypatch.setattr(m,'_post',post)
    assert m.ask('first')['operation']=='finish'
    assert m.ask('second')['operation']=='finish'
    assert requests[0][0]==API_URL+'/input_tokens'
    assert requests[0][1]==requests[1][1]
    assert requests[2][1]==requests[3][1]
    sent=requests[-1][1]
    assert sent['store'] is False and sent['truncation']=='disabled'
    assert 'previous_response_id' not in sent and 'context_management' not in sent
    assert sent['input'][1]['encrypted_content']=='opaque'
    assert sent['max_output_tokens']==131_000
    assert (tmp_path/'history-manifest.json').exists()
    assert 'test-secret' not in (tmp_path/'provider-responses.jsonl').read_text()
    with m.history_store.connect() as db:
        assert db.execute('select count(*) from messages').fetchone()[0]==6


def test_history_query_stays_inside_client(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch); operations=iter(['history_search','finish']); bodies=[]
    def post(url,body,**kwargs):
        if url.endswith('/input_tokens'):return {'input_tokens':50}
        bodies.append(body);return response(next(operations))
    monkeypatch.setattr(m,'_post',post)
    assert m.ask('first')['operation']=='finish'
    assert 'disabled until the first slide' in bodies[1]['input'][-1]['content'][0]['text']


def test_incomplete_response_archived_but_not_executed(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_post',lambda url,body,**kw:{'input_tokens':10} if url.endswith('/input_tokens') else response(status='incomplete'))
    with pytest.raises(RuntimeError,match='incomplete'):m.ask('go')
    with m.history_store.connect() as db:assert db.execute('select count(*) from messages').fetchone()[0]==3


def test_count_failure_never_infers(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch); calls=[]
    def post(url,body,**kw):calls.append(url);return {'wrong':'missing count'}
    monkeypatch.setattr(m,'_post',post)
    with pytest.raises(RuntimeError,match='counter'):m.ask('go')
    assert calls==[API_URL+'/input_tokens']


def test_actual_model_must_match_explicit_pin(tmp_path,monkeypatch):
    m=model(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_post',lambda url,body,**kw:{'input_tokens':10} if url.endswith('/input_tokens') else response(model='different'))
    with pytest.raises(RuntimeError,match='different model'):m.ask('go')


def test_participant_factory_selects_direct_api_without_inference(tmp_path,monkeypatch):
    from pairing.direct_models import create_participant
    monkeypatch.setenv('OPENAI_API_KEY','test-secret')
    result=create_participant('fixed',tmp_path,dict(provider='openai',model='pinned-model',context_window=1_000_000,max_output_tokens=131_000))
    assert isinstance(result,OpenAILearningModel)
    assert result.calls==0
    with pytest.raises(ValueError,match='Missing'):create_participant('fixed',tmp_path,{'provider':'openai'})
