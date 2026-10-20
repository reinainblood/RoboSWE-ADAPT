"""Independent code replay of completed cohort attempts.

Uses the strict journal replayer. Retained Python sources and notes can be
restored explicitly; passive waits use the original bounded wait policy.
"""
import argparse
import hashlib
import json
import traceback
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from pairing.overnight_suite import suite_worker
from pairing.run_toaster_learning import FreshEpisode
from pairing.simulation_wait import WaitingAttemptRunner
from roboswe.campaign import publish_video
from roboswe.execution.sandbox import CodeSandbox
from roboswe.jobs import create, save, now
from roboswe.library.attempts import read_journal, replay_journal


def starting_sources(directory):
    """Validate archived source bytes before sending them into a fresh sandbox."""
    continuity=json.loads((directory/'continuity.json').read_text())
    with ZipFile(directory/'library-start.zip') as archive:
        manifest=json.loads(archive.read('library-manifest.json'))
        if manifest!=continuity['library_start']:
            raise ValueError('Starting library manifest differs from continuity record')
        files=manifest['files']; names=[entry['path'] for entry in files]
        if len(set(names))!=len(names) or sorted(archive.namelist())!=sorted(names+['library-manifest.json']):
            raise ValueError('Unexpected or duplicate starting library entries')
        sources=[]
        for entry in files:
            name=entry['path'];path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or str(path)!=name or '\\' in name or not (path.suffix=='.py' or name=='NOTES.md'):
                raise ValueError('Invalid starting source path')
            data=archive.read(name)
            if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
                raise ValueError('Starting source checksum mismatch')
            sources.append((name,data.decode('utf-8')))
        return sources


def validate_source(directory, *, restore_library=False):
    report=json.loads((directory/'report.json').read_text())
    continuity=json.loads((directory/'continuity.json').read_text())
    entries=read_journal(directory/'submissions.jsonl')
    if not restore_library and (report.get('attempt_number')!=1 or continuity.get('retained') or continuity['library_start']['files']):
        raise ValueError('This replay path requires an empty initial code workspace')
    if restore_library:starting_sources(directory)
    if any(e['kind']=='operation_request' and e['data']['decision']['operation']=='wait' and e['data'].get('policy')!='passive_wait_v1' for e in entries):
        raise ValueError('Unsupported recorded passive-wait policy')
    finishes=[e for e in entries if e['kind']=='finish_task']
    terminal_finish_receipt=bool(len(entries)>=3 and entries[-1]['kind']=='operation_result'
        and entries[-1]['data'].get('operation')=='finish' and entries[-2]['kind']=='finish_task'
        and entries[-3]['kind']=='operation_request'
        and entries[-3]['data']['decision']['operation']=='finish')
    if not entries or entries[0]['kind']!='initial' or len(finishes)!=1 or not (entries[-1]['kind']=='finish_task' or terminal_finish_receipt):
        raise ValueError('Incomplete original journal')
    if finishes[0]['data']['cumulative_action_sha256']!=report['cumulative_action_sha256'] or entries[-1]['data']['cumulative_action_sha256']!=report['cumulative_action_sha256']:
        raise ValueError('Report and final journal action digests differ')
    return entries[0]['data']


def run(source, *, restore_library=False):
    source=source.resolve();initial=validate_source(source,restore_library=restore_library)
    directory,record=create('jobs',f'Independent code replay · {source.name}', 'cohort50-code-replay')
    record.update(source_job=source.name,status='starting',benchmark_score=None)
    (directory/'source.json').write_text(json.dumps({'source_job':source.name,'source_path':str(source),'initial':initial},indent=2))
    record['artifacts']['source']='source.json';save(directory,record)
    runtime=None
    try:
        runtime=FreshEpisode(initial['config'],initial['spec'],directory,1,[],worker=suite_worker)
        with CodeSandbox(initial['code_image'],lifetime_seconds=2400) as sandbox:
            if restore_library:
                for path,content in starting_sources(source):
                    receipt=sandbox.request('write',path,content=content)
                    if receipt.get('returncode') or receipt.get('error'):
                        raise RuntimeError(f'Starting source restore failed: {path}')
                restored=sandbox.archive(directory/'restored-library.zip')
                expected=json.loads((source/'continuity.json').read_text())['library_start']
                if restored!=expected:raise ValueError('Restored source library differs from recorded starting library')
                record['artifacts']['restored_library']='restored-library.zip'
            record['status']='running';save(directory,record)
            result=replay_journal(source/'submissions.jsonl',runtime,sandbox,directory/'replay-submissions.jsonl',runner_factory=WaitingAttemptRunner)
            # Extra state check beyond the existing action+predicate comparison.
            original=json.loads((source/'report.json').read_text())
            actual=result['trials'][-1]['replay']
            result['final_fingerprint_matches']=actual['end_fingerprint']==original['end_fingerprint']
            result['source_job']=source.name
            result['scope']='Fresh simulator and Docker workspace; '+('restores checksummed starting Python sources and NOTES.md, then ' if restore_library else '')+'executes original saved code operations. No model inference; replay duration is not a learning score. Other workspace files are not restored; any resulting divergence prevents verification.'
            (directory/'report.json').write_text(json.dumps(result,indent=2));record['artifacts']['report']='report.json'
            record['status']='reproduced_success' if result['trials'][-1]['verified_success'] and result['final_fingerprint_matches'] else 'replay_diverged' if not result['reproduced'] or not result['final_fingerprint_matches'] else 'reproduced_failure'
    except BaseException:
        record['status']='infrastructure_error';(directory/'stderr.txt').write_text(traceback.format_exc());record['artifacts']['stderr']='stderr.txt';raise
    finally:
        try:
            if runtime:
                receipt=runtime.close();publish_video(Path(receipt['recording_dir']),directory,record,phase='replay')
        finally:record['finished_at']=now();save(directory,record)
    print(json.dumps({'job':record['id'],'status':record['status']}))


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--restore-library',action='store_true');a=p.parse_args();run(a.source,restore_library=a.restore_library)


if __name__=='__main__':main()
