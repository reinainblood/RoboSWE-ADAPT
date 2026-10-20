"""Trusted regular-file checkpoints. Never import or execute workspace contents."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

MAX_BYTES = 256 * 1024 * 1024
MAX_FILES = 16384
MANIFEST = 'checkpoint-manifest.json'


def safe_path(name):
    if not isinstance(name,str) or not name or '\\' in name or '\0' in name:
        raise ValueError('Invalid checkpoint path')
    path=PurePosixPath(name)
    if path.is_absolute() or any(part in ('..','.') for part in name.split('/')) or str(path)!=name:
        raise ValueError('Unsafe checkpoint path')
    return name


def inspect_archive(path):
    with ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or len(names)>MAX_FILES+1:
            raise ValueError('Duplicate or excessive checkpoint members')
        modern=MANIFEST in names
        manifest_name=MANIFEST if modern else 'library-manifest.json'
        if manifest_name not in names or z.getinfo(manifest_name).file_size>8*1024*1024:
            raise ValueError('Missing or oversized checkpoint manifest')
        manifest=json.loads(z.read(manifest_name))
        if modern and manifest.get('schema_version')!='workspace-checkpoint-v2':
            raise ValueError('Unsupported checkpoint version')
        files=manifest.get('files');directories=manifest.get('directories',[]) if modern else []
        if not isinstance(files,list) or not isinstance(directories,list) or len(files)+len(directories)>MAX_FILES:
            raise ValueError('Invalid checkpoint inventory')
        expected={manifest_name}; paths=set(); total=0
        for row in directories+files:
            name=safe_path(row['path'])
            if name in paths:raise ValueError('Duplicate workspace path')
            paths.add(name)
            if modern and (type(row.get('mode')) is not int or not 0<=row['mode']<=0o777 or type(row.get('mtime_ns')) is not int):
                raise ValueError('Invalid checkpoint metadata')
        file_paths={r['path'] for r in files}
        if any(str(p) in file_paths for name in paths for p in PurePosixPath(name).parents if str(p)!='.'):
            raise ValueError('File used as directory')
        for row in files:
            name=row['path'];member='payload/'+name if modern else name
            if not modern and not (name.endswith('.py') or name=='NOTES.md'):
                raise ValueError('Unsupported legacy source member')
            expected.add(member)
            size=row['bytes']
            if type(size) is not int or size<0:raise ValueError('Invalid file size')
            total+=size
            if total>MAX_BYTES:raise ValueError('Checkpoint exceeds workspace quota')
            info=z.getinfo(member)
            mode=info.external_attr>>16
            if info.file_size!=size or (stat.S_IFMT(mode) not in (0,stat.S_IFREG)):
                raise ValueError('Nonregular or mismatched checkpoint member')
            digest=hashlib.sha256()
            with z.open(member) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
            if digest.hexdigest()!=row['sha256']:raise ValueError('Checkpoint hash mismatch')
        if set(names)!=expected or manifest.get('total_bytes')!=total:
            raise ValueError('Unrecorded checkpoint member or total mismatch')
        return manifest


def checkpoint(root, destination):
    root=Path(root); files=[];directories=[];total=0
    with ZipFile(destination,'x',compression=ZIP_DEFLATED) as z:
        for parent,dirs,names in os.walk(root,followlinks=False):
            dirs.sort();names.sort()
            for name in dirs+names:
                path=Path(parent)/name;before=path.lstat();relative=path.relative_to(root).as_posix()
                safe_path(relative)
                mode=stat.S_IMODE(before.st_mode)
                if mode>0o777:raise ValueError('Special permission bits unsupported')
                row={'path':relative,'mode':mode,'mtime_ns':before.st_mtime_ns}
                if stat.S_ISDIR(before.st_mode):directories.append(row)
                elif stat.S_ISREG(before.st_mode) and before.st_nlink==1:
                    total+=before.st_size
                    if total>MAX_BYTES:raise ValueError('Checkpoint exceeds workspace quota')
                    digest=hashlib.sha256()
                    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
                    with os.fdopen(fd,'rb') as source,z.open('payload/'+relative,'w') as target:
                        opened=os.fstat(source.fileno())
                        if (opened.st_ino,opened.st_dev)!=(before.st_ino,before.st_dev):raise ValueError('Workspace changed during checkpoint')
                        for chunk in iter(lambda:source.read(1024*1024),b''):
                            digest.update(chunk);target.write(chunk)
                        after=os.fstat(source.fileno())
                    if (after.st_size,after.st_mtime_ns,after.st_ctime_ns)!=(before.st_size,before.st_mtime_ns,before.st_ctime_ns):
                        raise ValueError('Workspace changed during checkpoint')
                    files.append({**row,'bytes':before.st_size,'sha256':digest.hexdigest()})
                else:raise ValueError('Symlink, special file or hardlink unsupported')
                if len(files)+len(directories)>MAX_FILES:raise ValueError('Too many workspace entries')
        manifest={'schema_version':'workspace-checkpoint-v2','files':files,'directories':directories,'total_bytes':total,
                  'scope':'regular files and directories; no cross-file atomicity during concurrent mutation'}
        z.writestr(MANIFEST,json.dumps(manifest))
    return manifest


def restore(archive, root):
    manifest=inspect_archive(archive);root=Path(root)
    if any(root.iterdir()):raise ValueError('Restore requires empty workspace')
    modern=manifest.get('schema_version')=='workspace-checkpoint-v2'
    with ZipFile(archive) as z:
        for row in manifest.get('directories',[]):(root/row['path']).mkdir(parents=True,exist_ok=True)
        for row in manifest['files']:
            target=root/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(('payload/' if modern else '')+row['path']) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
            os.chmod(target,row.get('mode',0o600))
            if modern:os.utime(target,ns=(row['mtime_ns'],row['mtime_ns']))
    for row in sorted(manifest.get('directories',[]),key=lambda r:len(PurePosixPath(r['path']).parts),reverse=True):
        target=root/row['path'];os.chmod(target,row['mode']);os.utime(target,ns=(row['mtime_ns'],row['mtime_ns']))
    return manifest


def main():
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'workspace.zip'
        if sys.argv[1]=='export':
            checkpoint('/workspace',archive)
            with archive.open('rb') as source:shutil.copyfileobj(source,sys.stdout.buffer)
        elif sys.argv[1]=='restore':
            size=0
            with archive.open('xb') as target:
                for chunk in iter(lambda:sys.stdin.buffer.read(1024*1024),b''):
                    size+=len(chunk)
                    if size>MAX_BYTES+16*1024*1024:raise ValueError('Oversized checkpoint archive')
                    target.write(chunk)
            print(json.dumps(restore(archive,'/workspace')))
        else:raise ValueError('Unknown maintenance operation')


if __name__=='__main__':main()
