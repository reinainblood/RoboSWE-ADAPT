import gzip
import json
from zipfile import ZipFile
from pairing.muse_checkpoint import checkpoint


def test_resumed_series_can_recover_its_original_completed_attempt(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    old=tmp_path/'jobs'/'original-attempt';new=tmp_path/'jobs'/'resumed-series'
    old.mkdir(parents=True);new.mkdir()
    summary={'attempt_number':1,'observed_success':False,'termination_reason':'model_deadline','control_steps':10,'elapsed_seconds':1800}
    (old/'job.json').write_text(json.dumps({'campaign_id':'original-series','cohort_id':'t05','provider':'muse','attempt_number':1,'status':'observed_fail'}))
    (old/'report.json').write_text(json.dumps(summary))
    (old/'remote-artifacts.json').write_text('{"persisted_readback_verified":true}')
    with gzip.open(old/'context-end.json.gz','wb') as f:f.write(json.dumps([{'role':'system','content':'same prompt'}],ensure_ascii=False).encode())
    with ZipFile(old/'library.zip','w') as z:z.writestr('library-manifest.json','{"files":[]}')
    (new/'job.json').write_text('{"status":"infrastructure_error","cohort_id":"t05"}')
    (new/'plan.json').write_text(json.dumps({'prompt':'same prompt','spec':{},'modal_image':{'image':'im-pinned'},'original_attempt_jobs':[str(old)]}))
    (new/'report.json').write_text(json.dumps({'trials':[summary]}))
    (new/'model.json').write_text('{"model":"meta/muse-spark-1.3-contributor"}')
    result=checkpoint('resumed-series')
    assert result['source_attempt']=='jobs/original-attempt'
    assert result['prior']==[summary]
    assert result['spec']['modal_image_id']=='im-pinned'
