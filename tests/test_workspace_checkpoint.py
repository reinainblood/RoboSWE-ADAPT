import json
import os
from pathlib import Path
from zipfile import ZipFile
import pytest
from roboswe.execution.workspace_checkpoint import checkpoint,restore,inspect_archive


def test_binary_workspace_roundtrip(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'empty').mkdir()
    for name,data in {'yaw_rot.npy':b'\x93NUMPY\0\xff','data.npz':b'PK\0\x80','state.json':b'{"x":1}','plain.txt':b'hello','motion.py':b'print(1)'}.items():
        p=source/name;p.write_bytes(data);p.chmod(0o640);os.utime(p,ns=(1234567890123,1234567890123))
    archive=tmp_path/'checkpoint.zip';manifest=checkpoint(source,archive)
    target=tmp_path/'target';target.mkdir();restore(archive,target)
    assert (target/'empty').is_dir()
    for row in manifest['files']:
        a,b=source/row['path'],target/row['path']
        assert a.read_bytes()==b.read_bytes()
        assert b.stat().st_mtime_ns==row['mtime_ns']
        assert b.stat().st_mode & 0o777 == row['mode']
    with pytest.raises(ValueError,match='empty'):restore(archive,target)


@pytest.mark.parametrize('kind',['symlink','hardlink','fifo'])
def test_unsupported_entries_fail_closed(tmp_path,kind):
    root=tmp_path/'source';root.mkdir();(root/'file').write_text('x')
    if kind=='symlink':(root/'other').symlink_to('file')
    elif kind=='hardlink':os.link(root/'file',root/'other')
    else:os.mkfifo(root/'other')
    with pytest.raises(ValueError,match='unsupported'):checkpoint(root,tmp_path/'a.zip')


def test_tampered_hash_and_unsafe_members_rejected(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'file').write_text('x')
    original=tmp_path/'a.zip';checkpoint(root,original)
    with ZipFile(original) as z:contents={n:z.read(n) for n in z.namelist()}
    for name,changes in [('hash',{'payload/file':b'y'}),('path',{'../evil':b'x'})]:
        path=tmp_path/(name+'.zip')
        with ZipFile(path,'w') as z:
            for key,value in {**contents,**changes}.items():z.writestr(key,value)
        with pytest.raises(ValueError):inspect_archive(path)


def test_legacy_source_archive_restore(tmp_path):
    import hashlib
    archive=tmp_path/'legacy.zip'
    with ZipFile(archive,'w') as z:
        z.writestr('motion.py',b'pass')
        z.writestr('library-manifest.json',json.dumps({'files':[{'path':'motion.py','bytes':4,'sha256':hashlib.sha256(b'pass').hexdigest()}],'total_bytes':4}))
    target=tmp_path/'target';target.mkdir();restore(archive,target)
    assert (target/'motion.py').read_bytes()==b'pass'


@pytest.mark.parametrize('mutation',['traversal','symlink','duplicate','quota','mode','file_parent'])
def test_malicious_checkpoint_rejected_before_any_restore(tmp_path,mutation):
    import hashlib,stat
    manifest={'schema_version':'workspace-checkpoint-v2','files':[{'path':'file','bytes':1,'sha256':hashlib.sha256(b'x').hexdigest(),'mode':0o600,'mtime_ns':1}], 'directories':[], 'total_bytes':1}
    name='payload/file'
    if mutation=='traversal':manifest['files'][0]['path']='../escape';name='payload/../escape'
    elif mutation=='quota':manifest['files'][0]['bytes']=257*1024*1024;manifest['total_bytes']=257*1024*1024
    elif mutation=='mode':manifest['files'][0]['mode']=0o4600
    elif mutation=='file_parent':manifest['directories']=[{'path':'file/child','mode':0o700,'mtime_ns':1}]
    archive=tmp_path/'bad.zip'
    from zipfile import ZipInfo
    with ZipFile(archive,'w') as z:
        info=ZipInfo(name)
        if mutation=='symlink':info.external_attr=(stat.S_IFLNK|0o777)<<16
        z.writestr(info,b'x')
        z.writestr('checkpoint-manifest.json',json.dumps(manifest))
        if mutation=='duplicate':
            with pytest.warns(UserWarning):z.writestr(name,b'x')
    target=tmp_path/'target';target.mkdir()
    with pytest.raises(ValueError):restore(archive,target)
    assert list(target.iterdir())==[]


def test_real_docker_numpy_checkpoint_roundtrip(tmp_path):
    if os.environ.get('ROBOSWE_TEST_DOCKER')!='1':pytest.skip('Opt-in bounded Docker checkpoint roundtrip')
    import subprocess
    from roboswe.execution.sandbox import CodeSandbox,docker_path,checkpoint_volume
    image=subprocess.check_output([docker_path(),'image','inspect','roboswe-code:local','--format','{{.Id}}'],text=True).strip()
    archive=tmp_path/'full.zip'
    with CodeSandbox(image) as original:
        original.request('write','seed.py',content="import numpy as np\nfrom pathlib import Path\nnp.save('yaw_rot.npy',np.array([1.25,-2.5]))\nnp.savez('cache.npz',x=np.arange(4))\nPath('state.json').write_text('{\\\"stage\\\":2}')\nPath('data.txt').write_text('retained')\n")
        result=original.request('execute','seed.py');assert result['returncode']==0,result
        first=checkpoint_volume(image,original.volume,archive)
    with CodeSandbox(image) as target:
        restored=target.restore_checkpoint(archive)
        assert restored['files']==first['files']
        target.request('write','verify.py',content="import numpy as np\nfrom pathlib import Path\nassert np.load('yaw_rot.npy').tolist()==[1.25,-2.5]\nassert np.load('cache.npz')['x'].tolist()==[0,1,2,3]\nassert Path('data.txt').read_text()=='retained'\n")
        result=target.request('execute','verify.py');assert result['returncode']==0,result
