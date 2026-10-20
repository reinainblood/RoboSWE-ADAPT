import json
import pytest
from pairing.openrouter_model import OpenRouterLearningModel,API_URL


def make(tmp_path,monkeypatch,**kwargs):
    monkeypatch.setattr('pairing.openrouter_model.credential',lambda:'test-secret')
    return OpenRouterLearningModel('fixed',tmp_path,model='google/gemini-4-argon',context_window=1_000_000,max_output_tokens=32768,**kwargs)


def test_missing_gemini_counter_fails_closed(tmp_path,monkeypatch):
    model=make(tmp_path,monkeypatch)
    monkeypatch.setattr(model,'_post',lambda *a,**kw:pytest.fail('must not infer without count'))
    with pytest.raises(RuntimeError,match='token_counter'):model.ask('go')


def test_pinned_gemini_reasoning_and_disabled_compression(tmp_path,monkeypatch):
    model=make(tmp_path,monkeypatch,token_counter=lambda **kw:50,provider='Google')
    bodies=[]
    def post(url,body,**kw):
        assert url==API_URL;bodies.append(body)
        return {'model':'google/gemini-4-argon','choices':[{'finish_reason':'stop','message':{'role':'assistant',
            'reasoning_details':[{'type':'reasoning.encrypted','data':'opaque-signed-content'}],
            'content':json.dumps(dict(operation='finish',path='',content='',message='done'))}}],
            'usage':{'prompt_tokens':50,'completion_tokens':5,'total_tokens':55}}
    monkeypatch.setattr(model,'_post',post)
    assert model.ask('go')['operation']=='finish'
    assert model.ask('again')['operation']=='finish'
    assert bodies[-1]['provider']=={'allow_fallbacks':False,'require_parameters':True,'only':['Google']}
    assert bodies[-1]['plugins']==[{'id':'context-compression','enabled':False}]
    assert bodies[-1]['messages'][2]['reasoning_details'][0]['data']=='opaque-signed-content'
    assert model.usage_summary()['cumulative_token_usage']['prompt_tokens']==100


@pytest.mark.parametrize('finish,native,refusal,content,status',[
    ('content_filter','SAFETY',None,'','model_refusal'),
    ('stop',None,'blocked','','model_refusal'),
    ('stop','SAFETY',None,'','model_refusal'),
    ('length','MAX_TOKENS',None,'{','model_output_limit'),
    ('stop',None,None,'not json','model_error'),
    ('stop',None,None,'{}','model_error'),
])
def test_model_failures_are_typed_preserve_receipts_and_never_retry(tmp_path,monkeypatch,finish,native,refusal,content,status):
    model=make(tmp_path,monkeypatch,token_counter=lambda **kw:50)
    payload={'model':model.model,'choices':[{'finish_reason':finish,'native_finish_reason':native,
        'message':{'role':'assistant','content':content,'refusal':refusal,'reasoning':'provider exposed evidence'}}],
        'usage':{'prompt_tokens':0,'completion_tokens':0,'cost':0}}
    calls=[]
    monkeypatch.setattr(model,'_post',lambda *a,**kw:calls.append(True) or payload)
    with pytest.raises(RuntimeError) as exc:model.ask('go')
    assert exc.value.status==status and len(calls)==1
    assert json.loads((tmp_path/'provider-responses.jsonl').read_text())['response']==payload
    assert (tmp_path/'provider-thinking.jsonl').exists() and (tmp_path/'model-usage.jsonl').exists()
    assert model.messages[-1]['content']==content
    if refusal is not None:
        assert model.messages[-1]['refusal']==refusal
    else:
        assert 'refusal' not in model.messages[-1]


def test_zero_cost_safety_response_settles_and_archives_history(tmp_path,monkeypatch):
    model=make(tmp_path,monkeypatch,token_counter=lambda **kw:50,ledger_path=tmp_path/'spend.json')
    payload={'model':model.model,'choices':[{'finish_reason':'content_filter','native_finish_reason':'SAFETY',
       'message':{'role':'assistant','content':'','refusal':None,'reasoning':'preserved'}}],
       'usage':{'prompt_tokens':0,'completion_tokens':0,'cost':0}}
    monkeypatch.setattr(model,'_post',lambda *a,**kw:payload)
    with pytest.raises(RuntimeError) as exc:model.ask('go')
    assert exc.value.status=='model_refusal'
    assert (tmp_path/'history.sqlite').exists()
    assert 'preserved' in (tmp_path/'provider-thinking.jsonl').read_text()
    ledger=json.loads((tmp_path/'spend.json').read_text())
    assert ledger['requests']==1 and ledger['reserved']=={}
    assert ledger['spent_usd']==0 and ledger['last_cost_usd']==0
