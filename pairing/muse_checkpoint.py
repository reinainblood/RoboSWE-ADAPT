"""Validate and restore a completed Muse learning boundary, never partial work."""
import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from pairing.continue_gr1_icl import restore_library


def checkpoint(source_series):
    source=Path('jobs')/source_series
    job=json.loads((source/'job.json').read_text())
    if job['status']!='infrastructure_error':raise ValueError('Resume requires a terminal infrastructure interruption')
    plan=json.loads((source/'plan.json').read_text())
    model=json.loads((source/'model.json').read_text())
    if model['model']!='meta/muse-spark-1.3-contributor':raise ValueError('Wrong source model')
    prior=json.loads((source/'report.json').read_text())['trials']
    if not prior or [r['attempt_number'] for r in prior]!=list(range(1,len(prior)+1)) or len(prior)>=3:raise ValueError('Need a contiguous, incomplete learning sequence')
    completed={}
    original_paths={Path(path).resolve() for path in plan.get('original_attempt_jobs',[])}
    if any(path.parent!=Path('jobs').resolve() for path in original_paths):raise ValueError('Original attempts must belong to this jobs directory')
    for path in Path('jobs').glob('*/job.json'):
        record=json.loads(path.read_text())
        original=path.parent.resolve() in original_paths
        if original and (record.get('cohort_id')!=job.get('cohort_id') or record.get('provider')!='muse'):raise ValueError('Original attempt belongs to a different task/provider')
        if (record.get('campaign_id')==source_series or original) and record.get('attempt_number') in range(1,len(prior)+1) and record['status'] in ('observed_pass','observed_fail'):
            n=record['attempt_number']
            if n in completed:raise ValueError('Ambiguous completed source attempt')
            completed[n]=path.parent
    if set(completed)!=set(range(1,len(prior)+1)):raise ValueError('Missing completed source attempts')
    for summary in prior:
        directory=completed[summary['attempt_number']]
        report=json.loads((directory/'report.json').read_text())
        if any(report.get(k)!=v for k,v in summary.items()):raise ValueError('Source report differs from series summary')
        if not json.loads((directory/'remote-artifacts.json').read_text())['persisted_readback_verified']:raise ValueError('Source artifacts not verified')
    end=completed[len(prior)]
    context=gzip.decompress((end/'context-end.json.gz').read_bytes())
    messages=json.loads(context)
    if messages[0]['content']!=plan['prompt']:raise ValueError('Saved model prompt changed')
    with ZipFile(end/'library.zip') as archive:
        manifest=json.loads(archive.read('library-manifest.json'))
        for item in manifest['files']:
            if hashlib.sha256(archive.read(item['path'])).hexdigest()!=item['sha256']:raise ValueError('Saved library checksum mismatch')
    spec=dict(plan['spec']);spec['modal_image_id']=plan['modal_image']['image']
    spec['sha256']=hashlib.sha256(json.dumps({k:v for k,v in spec.items() if k!='sha256'},sort_keys=True).encode()).hexdigest()
    return {'source_series':source_series,'source_attempt':str(end),'plan':plan,'model':model,'prior':prior,'messages':messages,'context_sha256':hashlib.sha256(context).hexdigest(),'library_sha256':hashlib.sha256((end/'library.zip').read_bytes()).hexdigest(),'spec':spec,'completed_directories':[str(completed[n]) for n in sorted(completed)]}


def restore(state,model,sandbox,directory):
    end=Path(state['source_attempt'])
    library=restore_library(sandbox,end/'library.zip',directory/'restored-library.zip')
    model.messages=state['messages']
    calls=model.messages[-1].get('tool_calls') or []
    model.pending_call=calls[0]['id'] if calls and not model.legacy_history else None
    compactions=[];call_numbers=[]
    for folder in state['completed_directories']:
        folder=Path(folder)
        compactions.extend(int(p.stem.rsplit('-',1)[1]) for p in folder.glob('compaction-*.json'))
        usage=folder/'model-usage.jsonl'
        if usage.exists():call_numbers.extend(json.loads(line).get('call',0) for line in usage.read_text().splitlines())
    model.compactions=max(compactions,default=0);model.calls=max(call_numbers,default=0)
    restored=json.dumps(model.messages,ensure_ascii=False).encode()
    # save_context serializes with ensure_ascii=False; compare source decoded data
    # as well as its original byte hash, without inventing or dropping messages.
    if hashlib.sha256(restored).hexdigest()!=state['context_sha256']:raise ValueError('Restored context bytes differ')
    receipt={k:state[k] for k in ('source_series','source_attempt','context_sha256','library_sha256')}
    receipt.update(library=library,pending_tool_call=model.pending_call,completed_attempts_preserved=len(state['prior']),next_attempt=len(state['prior'])+1,decision_transport=model.metadata['decision_transport'],compactions=model.compactions,model_calls=model.calls,changes=['Bounded transient HTTP retry','Bounded exact-fingerprint startup admission','Image ID pinned to original source image'],policy='Restore completed boundary only; interrupted later-attempt experience excluded from retry context.')
    (directory/'restoration.json').write_text(json.dumps(receipt,indent=2))
    return receipt
