import json
import sqlite3
import pytest
from pairing.context_history import HistoryStore, POLICY


def counter(*, messages, pending, overhead):
    return sum(m.get('tokens',0) for m in messages)+pending.get('tokens',0)+overhead.get('tokens',0)


def store(tmp_path, **kwargs):
    return HistoryStore(tmp_path/'history.sqlite',context_window=1_000_000,output_reserve=131_000,token_counter=counter,**kwargs)


def test_slide_reserve_overhead_chunk_and_durable_reasoning(tmp_path):
    h=store(tmp_path)
    messages=[{'role':'system','content':'fixed','tokens':1000}]
    for i in range(5):
        messages += [{'role':'user','content':str(i),'tokens':100_000},
                     {'role':'assistant','content':str(i),'reasoning':'actual exposed thought','tokens':80_000}]
    kept=h.prepare(messages,trial='trial2',pending={'tokens':10_000},overhead={'tokens':2000})
    assert kept[0]==messages[0]
    assert kept[1:]==messages[5:]  # 360k: complete exchanges, at least requested 200k
    assert h.slid
    assert h.search('actual exposed thought')['results']
    with h.connect() as db:
        assert db.execute('select count(*) from messages').fetchone()[0]==11
        slide=db.execute('select trial,estimated_removed_tokens from slides').fetchone()
        assert slide==('trial2',360_000)
    h.sync(kept+[{'role':'user','content':'next','tokens':10}], 'trial3')
    with h.connect() as db:assert db.execute('select count(*) from messages').fetchone()[0]==12
    dest=tmp_path/'trial3';dest.mkdir();h.export(dest)
    with sqlite3.connect(dest/'history.sqlite') as db:assert db.execute('pragma integrity_check').fetchone()[0]=='ok'


def test_disabled_before_slide_and_literal_bounded_query(tmp_path):
    h=store(tmp_path)
    assert h.search('anything')['enabled'] is False
    with h.connect() as db:db.execute("insert into slides(trial,epoch) values('1',0)")
    h.sync([{'role':'user','content':'x'*20_000}], '1')
    assert h.search("' OR 1=1 --")['results']==[]
    result=h.search('x')
    assert len(result['results'][0]['excerpt'])==2400
    assert result['results'][0]['truncated']


def test_cannot_split_native_tool_exchange(tmp_path):
    h=store(tmp_path)
    messages=[{'role':'user','tokens':300_000},
              {'role':'assistant','tool_calls':[{'id':'x'}],'tokens':300_000},
              {'role':'user','content':'image during outstanding tool','tokens':300_000}]
    with pytest.raises(ValueError,match='complete exchange'):h.prepare(messages,trial='1',pending={},overhead={})


def test_missing_counter_fails_before_inference(tmp_path):
    h=HistoryStore(tmp_path/'history.sqlite',context_window=1_000_000,output_reserve=131_000)
    with pytest.raises(RuntimeError,match='token_counter'):h.prepare([],trial='1',pending={},overhead={})


def test_oversized_current_input_fails_without_deleting_history(tmp_path):
    h=store(tmp_path)
    with pytest.raises(ValueError,match='fixed overhead'):h.prepare([],trial='1',pending={'tokens':900_000},overhead={})
    assert not h.slid


def test_muse_query_intercept_and_reasoning_archive(tmp_path,monkeypatch):
    from pairing import muse_model as m
    monkeypatch.setattr(m,'credential',lambda:'not-a-real-secret')
    model=m.MuseModel('fixed',tmp_path,context_policy=POLICY,token_counter=lambda **kw:100)
    replies=[{'operation':'history_search','path':'','content':'old','message':''},
             {'operation':'finish','path':'','content':'','message':'done'}]
    received=[]
    def fake(text,**kwargs):
        received.append(text)
        decision=replies.pop(0)
        model.messages.extend([{'role':'user','content':text},{'role':'assistant','content':json.dumps(decision),'reasoning':'real provider field'}])
        return decision
    monkeypatch.setattr(model,'_ask',fake)
    assert model.ask('hello')['operation']=='finish'
    assert 'disabled until the first slide' in received[1]
    assert model.compactions==0
    with model.history_store.connect() as db:
        assert 'real provider field' in db.execute("select body from messages where role='assistant' limit 1").fetchone()[0]


def test_codex_explicit_sqlite_policy_is_fail_closed():
    from roboswe.codex_login import CodexLogin
    with pytest.raises(ValueError,match='not supported'):CodexLogin(context_policy=POLICY)


def test_anthropic_sqlite_retains_signed_thinking_without_rollover(tmp_path,monkeypatch):
    from pairing.anthropic_model import AnthropicLearningModel, MODEL
    monkeypatch.setattr('pairing.anthropic_model.credential', lambda:'test-only')
    model=AnthropicLearningModel('fixed',tmp_path,workspace_id='wrkspc_test',token_counter=lambda **kw:100)
    model.latest_context_tokens=999_999
    def reply(body,**kwargs):
        assert 'history_search' in body['output_config']['format']['schema']['properties']['operation']['enum']
        return {'model':MODEL,'content':[{'type':'thinking','thinking':'actual','signature':'signature'},
                  {'type':'text','text':json.dumps({'operation':'finish','path':'','content':'','message':'done'})}]}
    monkeypatch.setattr(model,'_request',reply)
    assert model.ask('go')['operation']=='finish'
    assert model.rollovers==0
    with model.history_store.connect() as db:
        assert 'signature' in db.execute("select body from messages where role='assistant'").fetchone()[0]


def test_sparsie_sqlite_intercepts_history_search(tmp_path,monkeypatch):
    from pairing.sparsie_model import SparsieLearningModel,MODEL,PROVIDER,TOOL_NAME,LANE
    model=SparsieLearningModel(instructions='fixed',directory=tmp_path,context_policy=POLICY,token_counter=lambda **kw:100)
    operations=iter(['history_search','finish'])
    def reply(body,**kwargs):
        assert 'history_search' in body['tools'][0]['function']['parameters']['properties']['operation']['enum']
        op=next(operations)
        return {'model':MODEL,'sparsie':{'lane':LANE,'provider':PROVIDER,'provider_only':True},
            'choices':[{'message':{'role':'assistant','reasoning':'available','tool_calls':[{'id':op,'type':'function',
              'function':{'name':TOOL_NAME,'arguments':json.dumps(dict(operation=op,path='',content='',message=''))}}]}}],
            'usage':{'prompt_tokens':100,'completion_tokens':10}}
    monkeypatch.setattr(model,'_api',reply)
    assert model.ask('go')['operation']=='finish'
    assert model.compactions==0
    with model.history_store.connect() as db:
        assert db.execute('select count(*) from messages').fetchone()[0]==5


def test_anthropic_default_count_endpoint_includes_schema_and_images(tmp_path,monkeypatch):
    from pairing import anthropic_model as m
    from contextlib import contextmanager
    from io import StringIO
    monkeypatch.setattr(m,'credential',lambda:'test-secret')
    model=m.AnthropicLearningModel('instructions',tmp_path,workspace_id='wrkspc_test')
    image=tmp_path/'camera.png';image.write_bytes(b'fake-image')
    requests=[]
    @contextmanager
    def count(request,timeout):
        assert request.full_url==m.API_URL+'/count_tokens'
        requests.append(json.loads(request.data))
        yield StringIO('{"input_tokens": 2345}')
    monkeypatch.setattr(m.urllib.request,'urlopen',count)
    def reply(body,**kwargs):
        assert requests[-1]['messages']==body['messages']
        assert requests[-1]['system']==body['system']
        assert requests[-1]['output_config']==body['output_config']
        return {'model':m.MODEL,'content':[{'type':'text','text':json.dumps(dict(operation='finish',path='',content='',message='done'))}]}
    monkeypatch.setattr(model,'_request',reply)
    assert model.ask('observe',images=[image])['operation']=='finish'
    assert len(requests)==1
    assert requests[0]['messages'][-1]['content'][1]['source']['data']
    assert 'test-secret' not in (tmp_path/'token-counts.jsonl').read_text()
