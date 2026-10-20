import hashlib
import threading
import time
import pytest
from pairing.modal_episode import verify_volume_files


def test_all_persisted_bytes_checked_in_parallel():
    barrier=threading.Barrier(4)
    seen=[]
    class Volume:
        def read_file(self,path):
            seen.append(path);barrier.wait(timeout=2)
            yield path[-1:].encode()
    files=[{'path':str(n),'bytes':1,'sha256':hashlib.sha256(str(n).encode()).hexdigest()} for n in range(4)]
    result=verify_volume_files(Volume(),'/owned',files)
    assert len(result)==4 and set(seen)=={'/owned/'+str(n) for n in range(4)}
    assert [r['path'] for r in result]==['0','1','2','3']


def test_corrupt_persisted_file_fails_even_if_other_files_pass():
    class Volume:
        def read_file(self,path):yield b'wrong'
    with pytest.raises(RuntimeError,match='readback mismatch: video.mp4'):
        verify_volume_files(Volume(),'/owned',[{'path':'video.mp4','bytes':5,'sha256':hashlib.sha256(b'right').hexdigest()}])


def test_startup_cleanup_targets_recorded_sandbox_and_fdr(tmp_path):
    import json
    from types import SimpleNamespace
    from pairing.modal_episode import cleanup_startup_sandbox
    (tmp_path/'modal-session.json').write_text(json.dumps({'sandbox_id':'sb-owned','environment':'FDR'}))
    seen=[]
    def run(args,**kwargs):
        seen.append((args,kwargs))
        return SimpleNamespace(returncode=0,stdout='{"exit_before":null,"exit_after":137}')
    cleanup_startup_sandbox(tmp_path,run=run)
    assert seen[0][0][-1]=='sb-owned'
    assert seen[0][1]['env']['MODAL_ENVIRONMENT']=='FDR'
    assert json.loads((tmp_path/'startup-cleanup.json').read_text())['exit_after']==137


def test_startup_cleanup_rejects_wrong_environment(tmp_path):
    import json
    from pairing.modal_episode import cleanup_startup_sandbox
    (tmp_path/'modal-session.json').write_text(json.dumps({'sandbox_id':'sb-owned','environment':'main'}))
    with pytest.raises(ValueError):cleanup_startup_sandbox(tmp_path,run=lambda *a,**k:pytest.fail('must not call Modal'))


def test_delayed_visibility_must_match_original_manifest(monkeypatch):
    monkeypatch.setattr('pairing.modal_episode.time.sleep',lambda _:None)
    class Volume:
        def __init__(self):self.calls=0
        def read_file(self,path):
            self.calls+=1
            yield b'stale' if self.calls==1 else b'right'
    v=Volume();expected={'path':'video.mp4','bytes':5,'sha256':hashlib.sha256(b'right').hexdigest()}
    result=verify_volume_files(v,'/owned',[expected])
    assert v.calls==2 and result[0]['readback_attempts']==2
    assert expected['sha256']==hashlib.sha256(b'right').hexdigest()


def test_close_reaps_proxy_even_when_stdin_is_broken():
    import io
    from types import SimpleNamespace
    from pairing.modal_episode import RemoteEpisode
    waited=[]
    def broken():raise BrokenPipeError('closed pipe')
    def failed_call(operation):raise RuntimeError('original transport failure')
    runtime=object.__new__(RemoteEpisode)
    runtime.proc=SimpleNamespace(stdin=SimpleNamespace(close=broken),wait=lambda **kw:waited.append(kw))
    runtime.log=io.StringIO();runtime.call=failed_call
    with pytest.raises(RuntimeError,match='original transport failure'):runtime.close()
    assert waited==[{'timeout':30}] and runtime.log.closed
