import io
import json
import pytest
from pairing.gemini_token_count import GeminiNativeCounter


def native():
    return {'model':'models/gemini-4-argon','contents':[{'role':'user','parts':[{'text':'inspect'}, {'inlineData':{'mimeType':'image/jpeg','data':'aGVsbG8='}}]}],
            'systemInstruction':{'parts':[{'text':'robot policy'}]},
            'generationConfig':{'responseMimeType':'application/json','responseJsonSchema':{'type':'object'},'maxOutputTokens':1000}}


def test_complete_native_request_preserved_and_receipt_contains_no_key(tmp_path,monkeypatch):
    monkeypatch.setenv('COUNT_KEY','private-test-key')
    captured=[]
    def post(req,timeout):
        captured.append(json.loads(req.data));return io.BytesIO(b'{"totalTokens":123}')
    monkeypatch.setattr('urllib.request.urlopen',post)
    receipt=tmp_path/'counts.jsonl'
    counter=GeminiNativeCounter('gemini-4-argon',api_key_env='COUNT_KEY',receipt_path=receipt)
    assert counter.count_native_request(native())==123
    assert captured==[{'generateContentRequest':native()}]
    assert 'private-test-key' not in receipt.read_text()
    assert json.loads(receipt.read_text())['openrouter_translation_verified'] is False


def test_missing_google_credential_fails_without_request(monkeypatch):
    monkeypatch.delenv('UNSET_COUNTER_KEY',raising=False)
    with pytest.raises(RuntimeError,match='required'):GeminiNativeCounter('gemini-4-argon',api_key_env='UNSET_COUNTER_KEY')


def test_refuses_openrouter_conversion_and_bad_counts(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','test')
    counter=GeminiNativeCounter('gemini-4-argon')
    request=native();request['messages']=[]
    with pytest.raises(ValueError,match='not verified'):counter.count_native_request(request)
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:io.BytesIO(b'{"totalTokens":true}'))
    with pytest.raises(ValueError,match='totalTokens'):counter.count_native_request(native())
