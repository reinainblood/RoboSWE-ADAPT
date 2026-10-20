import json
from contextlib import contextmanager
import pytest
from pairing.modal_image_archive import archive_job, restore

class Volume:
    def __init__(self):self.files={};self.corrupt=False
    def read_file(self,path):
        if path not in self.files:raise FileNotFoundError(path)
        yield self.files[path]+(b'corrupt' if self.corrupt else b'')
    @contextmanager
    def batch_upload(self,force=False):yield self
    def put_file(self,source,path):self.files[path]=source.read()

def job(tmp_path):
    p=tmp_path/'job';p.mkdir();(p/'job.json').write_text(json.dumps({'id':'job','finished_at':'done','status':'observed_pass'}));(p/'model-images').mkdir();(p/'model-images'/'one.jpg').write_bytes(b'image-one');(p/'model-images'/'two.png').write_bytes(b'image-two');return p

def test_verified_archive_and_restore(tmp_path):
    p=job(tmp_path);v=Volume();assert archive_job(p,v)['files']==2
    assert not list((p/'model-images').iterdir())
    assert restore(p,v)['restored']==2
    assert (p/'model-images'/'one.jpg').read_bytes()==b'image-one'
    assert archive_job(p,v)['files']==2

def test_corrupt_remote_never_evicts(tmp_path):
    p=job(tmp_path);v=Volume();v.corrupt=True
    with pytest.raises(ValueError,match='verification'):archive_job(p,v)
    assert len(list((p/'model-images').iterdir()))==2

def test_changed_local_never_evicted(tmp_path):
    p=job(tmp_path);v=Volume();archive_job(p,v);restore(p,v)
    (p/'model-images'/'one.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError,match='differs'):archive_job(p,v)
    assert len(list((p/'model-images').iterdir()))==2

def test_active_job_refused(tmp_path):
    p=job(tmp_path);(p/'job.json').write_text(json.dumps({'status':'running'}))
    with pytest.raises(ValueError,match='finalized'):archive_job(p,Volume())
