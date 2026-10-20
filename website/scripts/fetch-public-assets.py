"""Inspect optional public assets; download only with explicit --download."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from urllib.parse import urlsplit
import urllib.request

ORIGIN='https://roboswe.com'
MAX_FILE=128*1024*1024
MAX_TOTAL=2*1024*1024*1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ValueError('Asset redirects are refused; refresh the reviewed manifest')


def entries(manifest):
    if manifest.get('schema_version')!=1 or manifest.get('origin')!=ORIGIN:
        raise ValueError('Unsupported public asset manifest')
    rows=manifest.get('assets',[])
    if not isinstance(rows,list) or len(rows)>512:raise ValueError('Invalid asset count')
    seen=set();total=0
    for row in rows:
        name=row.get('path','');p=PurePosixPath(name)
        if not isinstance(name,str) or not name or '\\' in name or '%' in name or p.is_absolute() or str(p)!=name or any(part in {'.','..'} or part.startswith('.') for part in p.parts) or p.parts[0] not in {'assets','episodes'}:
            raise ValueError('Unsafe asset destination')
        if name in seen:raise ValueError('Duplicate asset destination')
        seen.add(name)
        route=name[:-10] if name.endswith('index.html') else name
        url=urlsplit(row.get('url',''))
        if row.get('url')!=ORIGIN+'/'+route or url.scheme!='https' or url.netloc!='roboswe.com' or url.query or url.fragment or url.username:
            raise ValueError('Asset URL must match its reviewed HTTPS origin and path')
        size=row.get('bytes');digest=row.get('sha256')
        if type(size) is not int or not 0<=size<=MAX_FILE or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid asset size/hash')
        total+=size
    if total>MAX_TOTAL:raise ValueError('Manifest exceeds bounded download size')
    return rows


def target_path(root,name):
    root=Path(root).resolve();target=root/name
    for p in [target,*target.parents]:
        if p==root:break
        if p.is_symlink():raise ValueError('Symlink asset destination refused')
    if root not in target.resolve().parents:raise ValueError('Asset escapes website root')
    return target


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def fetch(manifest,root,*,download=False,opener=None):
    rows=entries(manifest);missing=[];verified=0
    for row in rows:
        path=target_path(root,row['path'])
        if path.exists():
            if not path.is_file() or path.stat().st_size!=row['bytes'] or file_hash(path)!=row['sha256']:
                raise ValueError('Existing asset differs; move it aside explicitly: '+row['path'])
            verified+=1;continue
        missing.append(row)
    if not download:return {'mode':'preview_only','verified':verified,'missing':len(missing),'download_bytes':sum(r['bytes'] for r in missing)}
    opener=opener or urllib.request.build_opener(NoRedirect()).open
    for row in missing:
        path=target_path(root,row['path']);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=None
        try:
            with opener(urllib.request.Request(row['url'],headers={'User-Agent':'RoboSWE-public-asset-fetch/1'}),timeout=60) as response:
                if response.status!=200 or response.geturl()!=row['url']:raise ValueError('Unexpected asset response or redirect')
                with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.asset-download-',delete=False) as stream:
                    temporary=Path(stream.name);count=0;h=hashlib.sha256()
                    while True:
                        block=response.read(min(1024*1024,row['bytes']-count+1))
                        if not block:break
                        count+=len(block)
                        if count>row['bytes']:raise ValueError('Asset exceeds declared size')
                        h.update(block);stream.write(block)
                    stream.flush();os.fsync(stream.fileno())
                if count!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Asset hash/size mismatch')
                target_path(root,row['path'])
                if path.exists():raise ValueError('Asset appeared during fetch; no overwrite')
                os.link(temporary,path);temporary.unlink();temporary=None
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
    return {'mode':'downloaded_and_verified','downloaded':len(missing),'previously_verified':verified,'bytes':sum(r['bytes'] for r in missing)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--download',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    print(json.dumps(fetch(json.loads((root/'public-assets.json').read_text()),root,download=args.download),indent=2))


if __name__=='__main__':main()
