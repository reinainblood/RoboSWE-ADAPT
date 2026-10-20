"""Immutable small ADAPT evidence archives in Modal FDR; never evict local data.

Live --checkpoint snapshots use SQLite backup and explicitly are not final or
crash-resumable certificates. Existing image archive watcher is independent.
"""
import argparse
import gzip
import io
import json
import os
import re
import shlex
import sqlite3
import tarfile
import tempfile
from datetime import datetime,timezone
from pathlib import Path
from zipfile import ZipFile
from pairing.modal_backup import digest_chunks,ensure_file,atomic_json

ENVIRONMENT='FDR'
VOLUME='roboswe-job-artifacts-v1'
ALLOW={'job.json','plan.json','model.json','report.json','verification.json','continuity.json',
       'visual-audit.json','protocol-review.json','failure-review.json','stderr.txt','source.zip','library-start.zip','library.zip','library-interrupted.zip','context-start.json.gz','context-end.json.gz',
       'context-policy.json','model-transcript.jsonl','provider-thinking.jsonl','provider-responses.jsonl',
       'model-usage.jsonl','token-counts.jsonl','provider-request-events.jsonl','provider-unsettled-responses.jsonl','context-estimates.jsonl','submissions.jsonl','timing.jsonl',
       'actions.jsonl','docker-session.json','modal-session.json','remote-artifacts.json'}
LIMIT=256*1024*1024
SECRET=re.compile(rb'(?:sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{30,})')


def check_secret(data,name):
    if SECRET.search(data):raise ValueError('Possible credential pattern; archive refused: '+name)
    if name.endswith('.gz'):check_secret(gzip.decompress(data),name[:-3])
    if name.endswith('.zip'):
        with ZipFile(io.BytesIO(data)) as archive:
            if sum(i.file_size for i in archive.infolist())>LIMIT:raise ValueError('Source zip exceeds bound')
            for item in archive.infolist():
                if not item.is_dir():check_secret(archive.read(item),item.filename)


def snapshot(directory,*,checkpoint=False):
    directory=Path(directory).resolve()
    job=json.loads((directory/'job.json').read_text())
    if job.get('id')!=directory.name or not str(job.get('kind','')).startswith('adaptive-direct-'):
        raise ValueError('Only identified new ADAPT jobs are supported')
    if not checkpoint and (not job.get('finished_at') or job.get('status') in ('running','starting','queued')):
        raise ValueError('Active job requires explicit --checkpoint')
    files={}
    for name in sorted(ALLOW):
        path=directory/name
        if not path.exists():continue
        if path.is_symlink() or not path.is_file():raise ValueError('Nonregular artifact: '+name)
        before=path.stat();data=path.read_bytes();after=path.stat()
        if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
            raise ValueError('Artifact changed while snapshotting; no upload: '+name)
        if len(data)>LIMIT:raise ValueError('Artifact exceeds bound')
        check_secret(data,name);files[name]=data
    dbpath=directory/'history.sqlite'
    if not dbpath.is_file() or dbpath.is_symlink():raise ValueError('Regular history.sqlite required')
    with tempfile.TemporaryDirectory(prefix='roboswe-sqlite-backup-') as temp:
        backup=Path(temp)/'history.sqlite'
        source=sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True,timeout=5)
        target=sqlite3.connect(backup)
        try:
            source.backup(target,pages=256,sleep=.01)
            if target.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity check failed')
        finally:target.close();source.close()
        files['history.sqlite']=backup.read_bytes()
    check_secret(files['history.sqlite'],'history.sqlite')
    if sum(map(len,files.values()))>LIMIT:raise ValueError('Archive input exceeds256MiB bound')
    manifest={'schema_version':1,'job_id':job['id'],'scope':'checkpoint' if checkpoint else 'finalized-job',
        'created_at':datetime.now(timezone.utc).isoformat(),'sqlite_consistency':'SQLite backup API + integrity_check',
        'cross_file_atomic':False,'crash_resume_certified':False,
        'files':[{'path':n,**digest_chunks([data])} for n,data in sorted(files.items())],
        'exclusions':'strict named evidence allowlist; no env/credentials, images or videos; local files retained'}
    files['archive-manifest.json']=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for name,data in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600
            tar.addfile(info,io.BytesIO(data))
    return buffer.getvalue(),manifest


def restore_payload(payload,receipt,destination):
    if digest_chunks([payload])!=receipt['archive']:raise ValueError('Archive hash/size mismatch')
    destination=Path(destination)
    if destination.exists():raise ValueError('Restore destination must not exist')
    expected={r['path']:r for r in receipt['files']};decoded={}
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile() or Path(member.name).name!=member.name or member.name in decoded:
                raise ValueError('Unsafe/duplicate archive member')
            if member.name not in expected and member.name!='archive-manifest.json':raise ValueError('Unexpected member')
            if member.size>LIMIT:raise ValueError('Oversized member')
            data=tar.extractfile(member).read()
            if member.name in expected and digest_chunks([data])!={k:expected[member.name][k] for k in ('bytes','sha256')}:
                raise ValueError('Member checksum mismatch')
            decoded[member.name]=data
    if set(decoded)!=(set(expected)|{'archive-manifest.json'}):raise ValueError('Incomplete archive')
    destination.mkdir(parents=True)
    for name,data in decoded.items():(destination/name).write_bytes(data)
    with sqlite3.connect(destination/'history.sqlite') as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Restored SQLite invalid')
    return {'restored_files':len(decoded),'destination':str(destination)}


def archive_job(directory,volume,*,checkpoint=False,profile=None,environment=ENVIRONMENT,volume_name=VOLUME):
    profile=profile or os.environ.get('MODAL_PROFILE')
    if not profile:raise ValueError('Explicit Modal profile required for durable restore provenance')
    if not isinstance(environment,str) or not environment.strip() or not isinstance(volume_name,str) or not volume_name.strip():
        raise ValueError('Explicit nonempty environment and volume required')
    directory=Path(directory).resolve();payload,manifest=snapshot(directory,checkpoint=checkpoint)
    digest=digest_chunks([payload]);remote=f'/adapt-evidence/{directory.name}/{manifest["scope"]}/{digest["sha256"]}.tar.gz'
    ensure_file(volume,io.BytesIO(payload),remote,digest)
    # Test restore of bytes actually read back, not merely the local source.
    readback=b''.join(volume.read_file(remote))
    receipt={'schema_version':1,'job_id':directory.name,'scope':manifest['scope'],'environment':environment,
        'volume':volume_name,'modal_profile':profile,'remote_path':remote,'archive':digest,'files':manifest['files'],
        'remote_readback_verified':True,'local_files_evicted':False,'crash_resume_certified':False,
        'verified_at':datetime.now(timezone.utc).isoformat()}
    with tempfile.TemporaryDirectory(prefix='roboswe-restore-test-') as temp:
        restore_payload(readback,receipt,Path(temp)/'restored')
    receipt['restore_test_passed']=True
    path=directory/f'adapt-archive-{digest["sha256"]}.json'
    receipt['restore_command']=f'MODAL_ENVIRONMENT={shlex.quote(environment)} MODAL_PROFILE={shlex.quote(profile)} python -m pairing.adapt_artifact_archive --environment {shlex.quote(environment)} --volume {shlex.quote(volume_name)} --profile {shlex.quote(profile)} --restore {shlex.quote(str(path))} --destination /path/to/new-restored-directory'
    body=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode()
    ensure_file(volume,io.BytesIO(body),remote+'.json',digest_chunks([body]))
    atomic_json(path,receipt)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',nargs='?');p.add_argument('--checkpoint',action='store_true')
    p.add_argument('--restore');p.add_argument('--destination');p.add_argument('--profile',default=os.environ.get('MODAL_PROFILE'));p.add_argument('--environment',default=ENVIRONMENT);p.add_argument('--volume',default=VOLUME);args=p.parse_args(argv)
    if not args.profile:p.error('Explicit --profile or MODAL_PROFILE required')
    if not args.environment.strip() or not args.volume.strip():p.error('Nonempty --environment and --volume required')
    receipt=None
    if args.restore:
        if not args.destination:p.error('--restore requires --destination')
        receipt=json.loads(Path(args.restore).read_text())
        if receipt.get('modal_profile')!=args.profile or receipt['environment']!=args.environment or receipt['volume']!=args.volume:raise ValueError('Receipt environment/volume mismatch')
    os.environ['MODAL_PROFILE']=args.profile
    os.environ['MODAL_ENVIRONMENT']=args.environment
    import modal
    volume=modal.Volume.from_name(args.volume,environment_name=args.environment)
    if receipt is not None:
        print(json.dumps(restore_payload(b''.join(volume.read_file(receipt['remote_path'])),receipt,args.destination)));return
    if not args.job:p.error('job required')
    print(json.dumps(archive_job(args.job,volume,checkpoint=args.checkpoint,profile=args.profile,environment=args.environment,volume_name=args.volume)))


if __name__=='__main__':main()
