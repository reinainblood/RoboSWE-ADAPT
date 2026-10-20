import json
from unittest.mock import patch

import pytest

from pairing.run_qwen_trials import QwenModel, parse_decision
from pairing.simulation_wait import WAIT_SCHEMA


def test_invalid_response_never_becomes_an_operation():
    for answer in ('[]', '{"operation":"execute"}', json.dumps({'operation':'host_shell','path':'','content':'','message':''})):
        with pytest.raises(ValueError):
            parse_decision({'content': answer}, WAIT_SCHEMA)


def test_reasoning_is_saved_and_retained_separately_from_rationale(tmp_path):
    decision = {'operation':'observe','path':'','content':'','message':'Inspect the actual hand contact.'}
    result = {'choices':[{'message':{'role':'assistant','content':json.dumps(decision),'reasoning_content':'Provider test reasoning'},'finish_reason':'stop'}], 'usage':{'prompt_tokens':123,'completion_tokens':30}}
    requests = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(result).encode()
    def send(request, **kwargs):
        requests.append(json.loads(request.data))
        return Response()
    with patch('pairing.run_qwen_trials.credential', return_value='secret-test-credential'), patch('urllib.request.urlopen', side_effect=send):
        model = QwenModel('Task instructions', tmp_path)
        for _ in range(2):
            assert model.ask(json.dumps({'observation':{'control_step':40}})) == decision
    assert requests[1]['messages'][2]['reasoning_content'] == 'Provider test reasoning'
    assert requests[0]['reasoning_effort'] == 'high'
    assert 'Provider test reasoning' in (tmp_path/'stdout.txt').read_text()
    assert 'Inspect the actual hand contact.' in (tmp_path/'stdout.txt').read_text()
    assert 'secret-test-credential' not in ''.join(p.read_text() for p in tmp_path.iterdir())
    receipts = [json.loads(line) for line in (tmp_path/'provider-responses.jsonl').read_text().splitlines()]
    assert receipts[0]['video_seconds'] == 2


def test_empty_final_preserves_evidence_but_never_uses_reasoning_as_decision(tmp_path):
    from pairing.run_qwen_trials import ModelProtocolError
    hidden=json.dumps({'operation':'execute','path':'unsafe.py','content':'','message':'not final'})
    result={'choices':[{'message':{'content':'','reasoning_content':hidden},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':100,'completion_tokens':200}}
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(result).encode()
    with patch('pairing.run_qwen_trials.credential',return_value='test-secret'),patch('urllib.request.urlopen',return_value=Response()):
        model=QwenModel('instructions',tmp_path)
        with pytest.raises(ModelProtocolError):model.ask('{"observation":{"control_step":4}}')
    assert model.messages[-1]['content']==''
    assert model.messages[-1]['reasoning_content']==hidden
    assert json.loads((tmp_path/'model-usage.jsonl').read_text())['usage']['completion_tokens']==200
    assert json.loads((tmp_path/'provider-responses.jsonl').read_text())['response']==result
