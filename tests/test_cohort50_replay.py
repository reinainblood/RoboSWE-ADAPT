import hashlib
import json
from zipfile import ZipFile

import pytest

from pairing.cohort50_replay import starting_sources


def archive(tmp_path, name='helper.py', contents=b'print(1)', checksum=None):
    manifest={'files':[{'path':name,'bytes':len(contents),'sha256':checksum or hashlib.sha256(contents).hexdigest()}],'total_bytes':len(contents)}
    (tmp_path/'continuity.json').write_text(json.dumps({'library_start':manifest}))
    with ZipFile(tmp_path/'library-start.zip','w') as z:
        z.writestr(name,contents)
        z.writestr('library-manifest.json',json.dumps(manifest))


def test_starting_sources_preserves_bytes(tmp_path):
    archive(tmp_path,contents=b'print("hello")\n')
    assert starting_sources(tmp_path)==[('helper.py','print("hello")\n')]


def test_starting_sources_rejects_tampering(tmp_path):
    archive(tmp_path,checksum='0'*64)
    with pytest.raises(ValueError,match='checksum'):
        starting_sources(tmp_path)


@pytest.mark.parametrize('name',['../outside.py','/absolute.py','state.npy','nested/../helper.py'])
def test_starting_sources_rejects_unsafe_or_unrecorded_file_types(tmp_path,name):
    archive(tmp_path,name=name)
    with pytest.raises(ValueError,match='path'):
        starting_sources(tmp_path)
def test_explicit_finish_receipt_is_a_complete_journal(tmp_path):
    from pairing.cohort50_replay import validate_source
    from roboswe.library.attempts import Journal
    (tmp_path/'report.json').write_text(json.dumps({'attempt_number':1,'cumulative_action_sha256':'digest'}))
    (tmp_path/'continuity.json').write_text(json.dumps({'retained':False,'library_start':{'files':[]}}))
    journal=Journal(tmp_path/'submissions.jsonl')
    journal.append('initial',{'provenance':'test'})
    journal.append('operation_request',{'decision':{'operation':'finish','path':'','content':'','message':''}})
    journal.append('finish_task',{'cumulative_action_sha256':'digest'})
    journal.append('operation_result',{'operation':'finish','cumulative_action_sha256':'digest'})
    journal.close()
    assert validate_source(tmp_path)=={'provenance':'test'}
