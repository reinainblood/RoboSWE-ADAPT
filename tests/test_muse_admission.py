import json
import pytest
from pairing import muse_cohort as c


def test_reset_retries_preserve_evidence_and_require_exact_state(tmp_path,monkeypatch):
    made=[]
    class Episode:
        def __init__(self,config,spec,directory,number,reports,worker):
            self.initial={'fingerprint':'wrong' if not made else 'exact'}
            self.closed=False;made.append(self)
            (directory/'initial-snapshot.zip').write_bytes(self.initial['fingerprint'].encode())
        def close(self):self.closed=True
    monkeypatch.setattr(c,'RemoteEpisode',Episode)
    (tmp_path/'job.json').write_text('{}')
    result=c.admit_episode({'config':{}},{},tmp_path,2,[],'exact')
    assert result is made[1] and made[0].closed and not made[1].closed
    assert (tmp_path/'rejected-resets/1/initial-snapshot.zip').read_bytes()==b'wrong'
    assert (tmp_path/'initial-snapshot.zip').read_bytes()==b'exact'
    assert (tmp_path/'job.json').exists()
    assert len(json.loads((tmp_path/'reset-admission.json').read_text())['resets'])==2


def test_reset_mismatch_never_bypasses_gate(tmp_path,monkeypatch):
    closed=[]
    class Episode:
        initial={'fingerprint':'wrong'}
        def __init__(self,*a,**kw):pass
        def close(self):closed.append(True)
    monkeypatch.setattr(c,'RemoteEpisode',Episode)
    with pytest.raises(RuntimeError,match='no model episode started'):
        c.admit_episode({'config':{}},{},tmp_path,2,[],'exact')
    assert len(closed)==3
