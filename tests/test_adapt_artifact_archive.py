import json,sqlite3
from pathlib import Path
import pytest
from pairing.adapt_artifact_archive import archive_job,snapshot,restore_payload

class Volume:
    def __init__(self):self.files={}
    def read_file(self,path):
        if path not in self.files:raise FileNotFoundError(path)
        yield self.files[path]
    def batch_upload(self,force=False):return self
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def put_file(self,file,path):self.files[path]=file.read()

def job(tmp_path):
    p=tmp_path/'job';p.mkdir()
    (p/'job.json').write_text(json.dumps({'id':'job','kind':'adaptive-direct-attempt','status':'running'}))
    (p/'.env').write_text('secret excluded')
    with sqlite3.connect(p/'history.sqlite') as db:db.execute('create table messages(value text)');db.execute("insert into messages values ('hello')")
    return p

def test_checkpoint_remote_readback_restore_no_eviction(tmp_path):
    p=job(tmp_path);v=Volume();before=(p/'history.sqlite').read_bytes()
    r=archive_job(p,v,checkpoint=True,profile='test-profile')
    assert r['modal_profile']=='test-profile' and 'MODAL_PROFILE=test-profile' in r['restore_command']
    assert r['environment']=='FDR' and r['volume']=='roboswe-job-artifacts-v1'
    assert r['scope']=='checkpoint' and r['remote_readback_verified'] and r['restore_test_passed']
    assert (p/'history.sqlite').read_bytes()==before and (p/'.env').exists()
    restore_payload(v.files[r['remote_path']],r,tmp_path/'restore')
    assert not (tmp_path/'restore'/'.env').exists()
    with sqlite3.connect(tmp_path/'restore'/'history.sqlite') as db:assert db.execute('select value from messages').fetchone()[0]=='hello'

def test_active_final_and_credential_rejected(tmp_path):
    p=job(tmp_path)
    with pytest.raises(ValueError,match='checkpoint'):snapshot(p)
    (p/'report.json').write_text(json.dumps({'leak':'sk-or-v1-'+'a'*40}))
    with pytest.raises(ValueError,match='credential'):snapshot(p,checkpoint=True)

def test_readback_corruption_refuses_receipt(tmp_path):
    p=job(tmp_path)
    class Broken(Volume):
        def put_file(self,file,path):self.files[path]=b'broken'
    with pytest.raises(ValueError):archive_job(p,Broken(),checkpoint=True,profile='test-profile')
    assert not list(p.glob('adapt-archive-*.json'))

@pytest.mark.parametrize('key',['sk-ant-'+('x'*40),'sk-'+('x'*40),'AIza'+('x'*35)])
def test_more_provider_credentials_rejected(tmp_path,key):
    p=job(tmp_path);(p/'model.json').write_text(json.dumps({'accident':key}))
    with pytest.raises(ValueError,match='credential'):snapshot(p,checkpoint=True)


def test_custom_environment_volume_receipt_and_restore_flags(tmp_path):
    import shlex
    p=job(tmp_path);v=Volume()
    receipt=archive_job(p,v,checkpoint=True,profile='community',environment='evaluation',volume_name='community-evidence')
    assert receipt['environment']=='evaluation' and receipt['volume']=='community-evidence'
    args=shlex.split(receipt['restore_command'])
    assert args[:2]==['MODAL_ENVIRONMENT=evaluation','MODAL_PROFILE=community']
    assert args[args.index('--environment')+1]=='evaluation'
    assert args[args.index('--volume')+1]=='community-evidence'
    assert args[args.index('--profile')+1]=='community'
    assert receipt['restore_test_passed'] and receipt['remote_readback_verified']


def test_cli_custom_restore_and_mismatch_before_file_read(tmp_path,monkeypatch,capsys):
    import sys
    from types import SimpleNamespace
    from pairing.adapt_artifact_archive import main
    p=job(tmp_path);v=Volume();receipt=archive_job(p,v,checkpoint=True,profile='community',environment='evaluation',volume_name='community-evidence')
    selected=[]
    def volume(name,environment_name):selected.append((name,environment_name));return v
    monkeypatch.setitem(sys.modules,'modal',SimpleNamespace(Volume=SimpleNamespace(from_name=volume)))
    monkeypatch.setenv('MODAL_PROFILE','community');monkeypatch.setenv('MODAL_ENVIRONMENT','evaluation')
    path=next(p.glob('adapt-archive-*.json'))
    main(['--restore',str(path),'--destination',str(tmp_path/'restored'),'--environment','evaluation','--volume','community-evidence','--profile','community'])
    assert selected==[('community-evidence','evaluation')]
    assert json.loads(capsys.readouterr().out)['restored_files']>0
    with pytest.raises(ValueError,match='mismatch'):
        main(['--restore',str(path),'--destination',str(tmp_path/'wrong'),'--environment','FDR','--volume','community-evidence','--profile','community'])
    assert not (tmp_path/'wrong').exists()
