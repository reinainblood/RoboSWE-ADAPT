import copy
import json

import pytest

from pairing.openrouter_google_count import GoogleMappedCounter


class Native:
    def __init__(self):
        self.calls = []

    def count_native_request(self, request, timeout):
        self.calls.append((request, timeout))
        return 100


def body():
    return {'model': 'google/gemini-4-argon', 'max_tokens': 32768, 'reasoning': {'effort': 'high'},
            'response_format': {'type': 'json_schema', 'json_schema': {'schema': {'type': 'object'}}},
            'messages': [{'role': 'system', 'content': 'instruction'},
                         {'role': 'user', 'content': [{'type': 'text', 'text': 'look'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,YWJj', 'detail': 'high'}}]},
                         {'role': 'assistant', 'content': 'answer', 'reasoning': 'thought', 'reasoning_details': [{'type': 'reasoning.text', 'text': 'thought', 'format': 'google-gemini-v1', 'index': 0}, {'type': 'reasoning.encrypted', 'data': 'YWJj', 'format': 'google-gemini-v1', 'index': 1}]}]}


def test_count_preserves_original_and_records_hash_not_content(tmp_path):
    native = Native()
    counter = GoogleMappedCounter(tmp_path, native_counter=native)
    request = body()
    original = copy.deepcopy(request)
    assert counter.count_body(request) == 8322
    assert request == original
    converted = native.calls[0][0]
    assert converted['contents'][1]['parts'][-1]['thoughtSignature'] == 'YWJj'
    counter.observe(130, estimated=8322, call=1)
    rows = [json.loads(x) for x in (tmp_path / 'context-estimates.jsonl').read_text().splitlines()]
    assert len(rows[0]['transformed_request_sha256']) == 64
    assert rows[1]['actual_prompt_tokens'] == 130
    assert counter.multiplier == pytest.approx(1.43)
    counter.count_body(request)
    counter.observe(100, estimated=8335, call=2)
    assert counter.multiplier == pytest.approx(1.43)
    assert 'thoughtSignature' not in (tmp_path / 'context-estimates.jsonl').read_text()


@pytest.mark.parametrize('change', ['format', 'tool', 'duplicate', 'reasoning_only', 'model'])
def test_unknown_metadata_stops_before_count(tmp_path, change):
    native = Native()
    counter = GoogleMappedCounter(tmp_path, native_counter=native)
    request = body()
    if change == 'format':
        request['messages'][2]['reasoning_details'][0]['format'] = 'unknown'
    elif change == 'tool':
        request['messages'][2]['tool_calls'] = []
    elif change == 'duplicate':
        request['messages'][2]['reasoning_details'].append(request['messages'][2]['reasoning_details'][-1])
    elif change == 'reasoning_only':
        request['messages'][2]['reasoning_details'] = []
    else:
        request['model'] = 'other'
    with pytest.raises(ValueError):
        counter.count_body(request)
    assert native.calls == []


def test_expired_deadline_never_counts(tmp_path):
    native = Native()
    counter = GoogleMappedCounter(tmp_path, native_counter=native, deadline=lambda: 0)
    with pytest.raises(TimeoutError):
        counter.count_body(body())
    assert native.calls == []


def test_count_failure_no_fallback(tmp_path):
    class Failed(Native):
        def count_native_request(self, request, timeout):
            raise RuntimeError('count unavailable')
    with pytest.raises(RuntimeError, match='count unavailable'):
        GoogleMappedCounter(tmp_path, native_counter=Failed()).count_body(body())


def test_transient_count_retry_is_clipped_to_original_deadline(tmp_path,monkeypatch):
    from pairing.gemini_token_count import GeminiCountRequestError
    clock=[100.0]
    monkeypatch.setattr('pairing.openrouter_google_count.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('pairing.openrouter_google_count.time.sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    class Transient(Native):
        def count_native_request(self,request,timeout):
            self.calls.append(timeout)
            if len(self.calls)==1:
                clock[0]+=90
                raise GeminiCountRequestError('count transport',diagnostic={'type':'URLError','reason_type':'TimeoutError'},retryable=True)
            return 100
    native=Transient();counter=GoogleMappedCounter(tmp_path,native_counter=native,deadline=lambda:200)
    assert counter.count_body(body())==8322
    assert native.calls==[90,9]
    rows=[json.loads(x) for x in (tmp_path/'context-estimates.jsonl').read_text().splitlines()]
    assert rows[0]['will_retry'] and rows[0]['completion_generated'] is False
    assert rows[-1]['count_attempts']==2


def test_count_retry_limit_and_permanent_failure(tmp_path,monkeypatch):
    from pairing.gemini_token_count import GeminiCountRequestError
    monkeypatch.setattr('pairing.openrouter_google_count.time.sleep',lambda seconds:None)
    class Failed(Native):
        def __init__(self,retryable):super().__init__();self.retryable=retryable
        def count_native_request(self,request,timeout):
            self.calls.append(timeout)
            raise GeminiCountRequestError('count failed',diagnostic={'http_status':503 if self.retryable else 400},retryable=self.retryable)
    for retryable,expected in ((True,2),(False,1)):
        native=Failed(retryable)
        with pytest.raises(GeminiCountRequestError):GoogleMappedCounter(tmp_path,native_counter=native).count_body(body())
        assert len(native.calls)==expected


def test_null_refusal_history_compatibility_does_not_mutate_source(tmp_path):
    native=Native();counter=GoogleMappedCounter(tmp_path,native_counter=native)
    request=body();request['messages'][2]['refusal']=None
    original=copy.deepcopy(request)
    counter.count_body(request)
    assert request==original
    assert 'refusal' not in json.dumps(native.calls[0][0])
    for value in ('blocked', '', False, {}, []):
        request['messages'][2]['refusal']=value
        with pytest.raises(ValueError,match='Unsupported message'):
            counter.count_body(request)
    assert len(native.calls)==1


def test_normal_response_saved_history_then_next_mapped_count(tmp_path,monkeypatch):
    from pairing.openrouter_model import OpenRouterLearningModel
    native=Native()
    monkeypatch.setattr('pairing.openrouter_model.credential',lambda:'test-secret')
    monkeypatch.setattr('pairing.openrouter_google_count.GeminiNativeCounter',lambda *a,**kw:native)
    model=OpenRouterLearningModel('fixed',tmp_path,model='google/gemini-4-argon',
        context_window=1048576,max_output_tokens=32768,context_policy='sqlite-sliding-or-google-count-v2')
    responses=[]
    def post(*args,**kwargs):
        response={'model':model.model,'choices':[{'finish_reason':'stop','message':{'role':'assistant','refusal':None,
          'content':json.dumps({'operation':'observe','path':'','content':'','message':'look'})}}],
          'usage':{'prompt_tokens':100,'completion_tokens':10,'cost':0.0003}}
        responses.append(response)
        return response
    monkeypatch.setattr(model,'_post',post)
    assert model.ask('first')['operation']=='observe'
    assert 'refusal' not in model.messages[-1]
    model.save_context(tmp_path/'saved-context.json.gz')
    # Regression: a second real ask traverses both SQLite prepare counting and
    # assembled-request counting with the first response retained.
    assert model.ask('second')['operation']=='observe'
    assert len(responses)==2 and len(native.calls)>=4
    assert any(part.get('text','').startswith('{"operation"')
               for request,_ in native.calls for content in request['contents']
               if content['role']=='model' for part in content['parts'])
    raw=[json.loads(line) for line in (tmp_path/'provider-responses.jsonl').read_text().splitlines()]
    assert raw[0]['response']['choices'][0]['message']['refusal'] is None
