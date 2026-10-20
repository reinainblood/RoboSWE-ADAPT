"""Move finalized model-input images to immutable, readback-verified Modal archives.

Playback videos, reports, code and context remain local. Restore with --restore JOB.
"""
import argparse
import io
import json
import shutil
import tarfile
import time
import fcntl
from pathlib import Path
from datetime import datetime, timezone

from pairing.modal_backup import atomic_json, digest_chunks, remote_digest, ensure_file

VOLUME = 'roboswe-job-artifacts-v1'
ENVIRONMENT = 'FDR'
MANIFEST = 'model-images-remote.json'


def archive_job(directory, volume):
    directory = Path(directory)
    job = json.loads((directory/'job.json').read_text())
    if not job.get('finished_at') or job['status'] in ('running','starting','queued'):
        raise ValueError('Only finalized jobs may be archived')
    parent=directory.parent/str(job.get('campaign_id',''))/'job.json'
    if job.get('campaign_id') and parent.exists() and not json.loads(parent.read_text()).get('finished_at'):
        raise ValueError('Parent learning series is still active')
    manifest_path = directory/MANIFEST
    images = directory/'model-images'
    if manifest_path.exists():
        receipt = json.loads(manifest_path.read_text())
        if remote_digest(volume, receipt['remote_path']) != receipt['archive']:
            raise ValueError('Existing remote archive failed readback')
    else:
        paths = sorted(images.iterdir()) if images.exists() else []
        if not paths:
            return None
        if any(p.is_symlink() or not p.is_file() for p in paths):
            raise ValueError('Image directory must contain only regular files')
        files = []
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
            for p in paths:
                data = p.read_bytes()
                files.append({'path':p.name, **digest_chunks([data])})
                info = tarfile.TarInfo(p.name)
                info.size = len(data)
                info.mtime = int(p.stat().st_mtime)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
        payload = buffer.getvalue()
        digest = digest_chunks([payload])
        remote_path = f'/model-image-archives/{directory.name}/{digest["sha256"]}.tar.gz'
        ensure_file(volume, io.BytesIO(payload), remote_path, digest)
        receipt = {'schema_version':1,'job_id':job['id'],'environment':ENVIRONMENT,
                   'volume':VOLUME,'remote_path':remote_path,'archive':digest,'files':files,
                   'verified_at':datetime.now(timezone.utc).isoformat(),
                   'verification':'Full remote archive readback matched byte count and SHA-256',
                   'restore_command':f'MODAL_ENVIRONMENT=FDR MODAL_PROFILE=snorkelai ~/.local/share/uv/tools/modal/bin/python -m pairing.modal_image_archive --restore {directory}'}
        # Publish both manifests before removing any local image bytes.
        body = json.dumps(receipt,sort_keys=True,indent=2).encode()
        ensure_file(volume,io.BytesIO(body),remote_path+'.json',digest_chunks([body]))
        atomic_json(manifest_path,receipt)
    # Validate every surviving local file before any eviction, including crash resume.
    expected = {item['path']:item for item in receipt['files']}
    surviving = list(images.iterdir()) if images.exists() else []
    for p in surviving:
        item = expected.get(p.name)
        if p.is_symlink() or not p.is_file() or item is None or digest_chunks([p.read_bytes()]) != {k:item[k] for k in ('bytes','sha256')}:
            raise ValueError(f'Local image differs from verified archive: {p.name}')
    for p in surviving:
        p.unlink()
    receipt['local_images_evicted'] = True
    atomic_json(manifest_path,receipt)
    return {'job':directory.name,'files':len(surviving),'bytes_evicted':sum(expected[p.name]['bytes'] for p in surviving)}


def restore(directory,volume):
    directory=Path(directory);receipt=json.loads((directory/MANIFEST).read_text())
    payload=b''.join(volume.read_file(receipt['remote_path']))
    if digest_chunks([payload])!=receipt['archive']:raise ValueError('Archive checksum mismatch')
    expected={x['path']:x for x in receipt['files']};decoded={}
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile() or Path(member.name).name!=member.name or member.name not in expected or member.name in decoded:
                raise ValueError('Invalid archive member')
            data=tar.extractfile(member).read();item=expected[member.name]
            if digest_chunks([data])!={k:item[k] for k in ('bytes','sha256')}:raise ValueError('Image checksum mismatch')
            decoded[member.name]=data
    if set(decoded)!=set(expected):raise ValueError('Incomplete archive')
    images=directory/'model-images';images.mkdir(exist_ok=True)
    for name,data in decoded.items():
        path=images/name
        if path.exists() and (path.is_symlink() or path.read_bytes()!=data):raise ValueError('Refusing to replace changed image')
        tmp=images/(name+'.restore-tmp');tmp.write_bytes(data);tmp.replace(path)
    receipt['local_images_evicted']=False;atomic_json(directory/MANIFEST,receipt)
    return {'restored':len(decoded),'job':directory.name}


def archive_available(volume,args):
    candidates=[]
    for jp in Path('jobs').glob('*/job.json'):
        j=json.loads(jp.read_text());images=jp.parent/'model-images'
        parent=jp.parent.parent/str(j.get('campaign_id',''))/'job.json'
        if j.get('campaign_id') and parent.exists() and not json.loads(parent.read_text()).get('finished_at'):continue
        if j.get('finished_at') and j['status'] not in ('running','starting','queued') and images.exists():
            size=sum(p.stat().st_size for p in images.iterdir() if p.is_file())
            if size:candidates.append((size,jp.parent))
    for _,directory in sorted(candidates,reverse=True)[:args.limit]:
        if shutil.disk_usage('.').free/2**30>=args.target_free_gib:break
        print(json.dumps(archive_job(directory,volume)),flush=True)
    print(json.dumps({'free_gib':shutil.disk_usage('.').free/2**30}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--restore',type=Path);parser.add_argument('--target-free-gib',type=float,default=10);parser.add_argument('--limit',type=int,default=10000);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    import modal
    volume=modal.Volume.from_name(VOLUME,environment_name=ENVIRONMENT)
    if args.restore:print(json.dumps(restore(args.restore,volume)),flush=True);return
    lock=Path('artifacts/muse-learning-20261006/image-archive.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        archive_available(volume,args)
        if not args.watch:break
        progress=json.loads(Path('artifacts/muse-learning-20261006/progress.json').read_text())
        if not progress.get('pending') and not progress.get('active'):break
        time.sleep(60)


if __name__=='__main__':main()
