"""Independent provider lanes for the existing frozen cohort campaign.

Adopts existing live series without interrupting solvers. Preflight and replay
occupy a lane. A measured capacity gate permits three Astra lanes and one Qwen
lane without increasing simultaneous requests to the self-hosted Qwen endpoint.
"""
import json
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

from pairing.cohort50 import ROOT, emit, effective_row, preflight_directory
from roboswe.campaign import reference_lease
from roboswe.jobs import now, save
from roboswe.storage import require_disk_headroom

PRIORITIES=['t01','t02','t10','t03','t04','t16','t05','t06','t07','t08','t09']
LANES=('astra','qwen','astra-2','astra-3')


def lane_provider(lane):
    if lane not in LANES:raise ValueError('Unknown lane')
    return lane.split('-')[0]


def provider_pause_reason(error):
    message=error.lower()
    if any(x in message for x in ('usage limit','insufficient_quota','spend control','http 429','rate limit')):
        return 'Provider quota/rate limit; inspect fresh limits before further launches'
    if any(x in message for x in ('http 401','http 403','account differs','account changed','no chatgpt codex login','keychain credential unavailable')):
        return 'Provider authorization/account failure; inspect account before further launches'
    return None


def ordered_rows(manifest):
    return sorted([effective_row(r) for r in manifest['tasks']],key=lambda r:PRIORITIES.index(r['id']) if r['id'] in PRIORITIES else len(PRIORITIES)+int(r['id'][1:]))


def alive(item):
    command=subprocess.run(['ps','-p',str(item['pid']),'-o','command='],capture_output=True,text=True).stdout
    return item['identity'] in command


def source_jobs(campaign):
    values=[]
    for path in Path('jobs').glob('*/job.json'):
        record=json.loads(path.read_text())
        if record.get('campaign_id')==campaign and record.get('kind')=='cohort50-series':values.append(record)
    return values


def pending_rows(rows,provider,launched):
    used={j['cohort_id'] for j in launched if j['provider']==provider}
    return [r for r in rows if r['id'] not in used]


def repair_row(request, rows, launched):
    """Only a documented infrastructure failure can request an automatic rerun."""
    source=next((j for j in launched if j['id']==request['source_series_id']),None)
    row=next(r for r in rows if r['id']==request['task_id'])
    contract_repair=bool(source and request.get('repair_kind')=='native_hand_contract'
        and source['status']=='completed_unscored' and row.get('config',{}).get('robot')=='GR1'
        and source.get('hand_action_contract')!='native-raw-position-v2')
    if source is None or (source['status']!='infrastructure_error' and not contract_repair):
        raise ValueError('Repair request requires a terminal infrastructure-error series')
    if source['provider']!=request['provider'] or source['cohort_id']!=request['task_id']:
        raise ValueError('Repair request provider/task differs from failed source')
    return {**row,'repair_request_id':request['request_id'],'retry_of_series':source['id'],'repair_reason':request['reason'],'repair_kind':request.get('repair_kind','provider_protocol')}


def first_replay_source(campaign):
    series={j['id'] for j in source_jobs(campaign) if j['provider']=='astra' and j['cohort_id']=='t01'}
    for p in sorted(Path('jobs').glob('*/job.json')):
        j=json.loads(p.read_text())
        if j.get('campaign_id') in series and j.get('attempt_number')==1 and j.get('status')=='observed_pass':
            return p.parent
    return None


def main():
    saved=json.loads((ROOT/'active.json').read_text());directory=Path(saved['directory'])
    record=json.loads((directory/'job.json').read_text());campaign=record['id'];deadline=saved['deadline']
    manifest=json.loads((ROOT/'cohort.json').read_text());rows=ordered_rows(manifest)
    image=subprocess.check_output(['docker','image','inspect','roboswe-code:local','--format','{{.Id}}'],text=True).strip()
    prior=json.loads((directory/'progress.json').read_text())
    completed=prior.get('completed',[])
    lanes={}
    for item in prior.get('active',[]):
        item={**item,'stage':item.get('stage','series')}
        item.setdefault('identity',f'pairing.cohort50 series --task-id {item["id"]} --provider {item["provider"]} --parent {campaign}')
        if alive(item):lanes[item.get('lane',item['provider'])]=item
        else:completed.append({**item,'status':'process_already_ended_on_adoption'})
    launched=source_jobs(campaign)
    queues={p:pending_rows(rows,p,launched) for p in ('astra','qwen')}
    failed={(item['id'],item.get('case_revision')) for item in completed if item.get('status')=='preflight_failed'}
    replay_attempted=any(item.get('stage')=='replay' for item in completed) or any(item['stage']=='replay' for item in lanes.values())
    processes={};streams={}
    # Acquire before changing owner metadata, so another coordinator cannot be hidden.
    with reference_lease({'jobs_dir':'jobs','execution':{'memory_mb':6144,'cpu':4,'min_free_disk_gb':5}}):
        emit(ROOT/'active.json',{**saved,'pid':os.getpid(),'scheduler':'independent-provider-lanes-v2','max_simulations':len(LANES)})
        effective=sorted(rows,key=lambda row:row['id'])
        emit(ROOT/'cohort-effective.json',{'base_cohort_sha256':manifest['sha256'],'tasks':effective,
            'effective_tasks_sha256':hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest(),
            'scope':'Frozen task membership with explicit pre-evaluation scene revisions. Original cohort and failed preflights remain preserved.'})
        emit(directory/'execution-order.json',{'per_provider_task_ids':[r['id'] for r in rows],
            'policy':'At most four simulations: three Astra lanes and one Qwen lane after a four-simulator capacity probe and successful live replay window. Preflight and replay consume a lane. Same scenes and per-series budgets; compare matched completed pairs and recorded transport versions.'})
        archive=directory/f'queue-source-{os.getpid()}.zip'
        with ZipFile(archive,'w') as z:
            for folder in ('pairing','roboswe','robot_session'):
                for path in Path(folder).rglob('*.py'):z.write(path)
        record['artifacts'].update(execution_order='execution-order.json',queue_source=archive.name,progress='progress.json')
        record.update(status='running',finished_at=None);save(directory,record)

        def snapshot():
            emit(directory/'progress.json',{'completed':completed,'active':list(lanes.values()),
                'queues':{p:[r['id'] for r in q] for p,q in queues.items()},'deadline':deadline,'scheduler':'independent-provider-lanes-v2','max_simulations':len(LANES)})

        def launch(lane,row,stage,source=None):
            provider=lane_provider(lane)
            if stage=='replay':
                args=['-m','pairing.cohort50_replay',str(source)]
                identity='pairing.cohort50_replay '+str(source)
            else:
                args=['-m','pairing.cohort50',stage,'--task-id',row['id']]
                identity='pairing.cohort50 '+stage+' --task-id '+row['id']
                if stage=='series':
                    args+=['--provider',provider,'--parent',campaign,'--code-image',image]
                    identity+=f' --provider {provider} --parent {campaign}'
                    if row.get('repair_request_id'):
                        args+=['--repair-request-id',row['repair_request_id']]
            log=ROOT/f'{row["id"]}-{provider}-{stage}.log';stream=log.open('ab')
            proc=subprocess.Popen([sys.executable,'-u',*args],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            processes[proc.pid]=proc;streams[proc.pid]=stream
            lanes[lane]={'id':row['id'],'provider':provider,'lane':lane,'stage':stage,'pid':proc.pid,'identity':identity,'started_at':now(),'started_epoch':time.time()}
            if row.get('case_revision'):lanes[lane]['case_revision']=row['case_revision']
            if row.get('repair_request_id'):
                lanes[lane]['repair_request_id']=row['repair_request_id']
                if stage=='series':emit(ROOT/'repair-requests'/f'{row["repair_request_id"]}.receipt.json',{'status':'launched','pid':proc.pid,'task_id':row['id'],'provider':provider,'started_at':now()})
            print(json.dumps({'launched':lanes[lane]}),flush=True)
            snapshot()

        snapshot()
        try:
            while True:
                control=json.loads((ROOT/'control.json').read_text()) if (ROOT/'control.json').exists() else {}
                can_launch=time.time()<deadline-4500 and not control.get('stop') and not control.get('pause_launches')
                for path in sorted((ROOT/'repair-requests').glob('*.request.json')):
                    request=json.loads(path.read_text())
                    receipt=path.with_name(request['request_id']+'.receipt.json')
                    if receipt.exists():continue
                    if any(r.get('repair_request_id')==request['request_id'] for q in queues.values() for r in q):continue
                    existing=source_jobs(campaign)
                    if any(j.get('repair_request_id')==request['request_id'] for j in existing):
                        emit(receipt,{'status':'already_launched'});continue
                    row=repair_row(request,rows,existing)
                    queues[request['provider']].insert(0,row)
                for lane,item in list(lanes.items()):
                    provider=item['provider']
                    proc=processes.get(item['pid'])
                    running=proc.poll() is None if proc else alive(item)
                    if running:
                        # Bound only this owned preflight; solver requests use their task clocks.
                        if proc and item['stage']=='preflight' and time.time()-item['started_epoch']>300:
                            proc.terminate();print(json.dumps({'preflight_timeout':item}),flush=True)
                        continue
                    code=proc.returncode if proc else None
                    if item['pid'] in streams:streams.pop(item['pid']).close()
                    done={**item,'exit_code':code,'finished_at':now()}
                    if item['stage']=='preflight':
                        if code!=0:
                            failed.add((item['id'],item.get('case_revision')));done['status']='preflight_failed'
                        else:done['status']='preflight_passed'
                    elif item['stage']=='replay':
                        done['status']='replay_finished' if code==0 else 'replay_process_failed'
                    else:
                        matches=[j for j in source_jobs(campaign) if j['cohort_id']==item['id'] and j['provider']==provider]
                        done['series_outcomes']=[{'job':j['id'],'status':j['status']} for j in matches]
                        for match in matches:
                            error_path=Path('jobs')/match['id']/'stderr.txt'
                            if match['status']=='infrastructure_error' and error_path.exists():
                                reason=provider_pause_reason(error_path.read_text())
                                if reason:
                                    current=json.loads((ROOT/'control.json').read_text()) if (ROOT/'control.json').exists() else {}
                                    current.setdefault('provider_pauses',{})[provider]={'reason':reason,'source_job':match['id'],'since':now()}
                                    emit(ROOT/'control.json',current);control=current
                                    print(json.dumps({'provider_paused':provider,'reason':reason,'source_job':match['id']}),flush=True)
                    completed.append(done);del lanes[lane]
                    print(json.dumps({'finished':done}),flush=True);snapshot()
                for lane in LANES:
                    provider=lane_provider(lane)
                    if provider in control.get('provider_pauses',{}):continue
                    if lane in lanes or not can_launch or len(lanes)>=int(control.get('max_active',len(LANES))):continue
                    require_disk_headroom('jobs',7)
                    if provider=='astra' and not replay_attempted:
                        source=first_replay_source(campaign)
                        if source is not None:
                            replay_attempted=True;launch(lane,rows[0],'replay',source);continue
                    while queues[provider] and (queues[provider][0]['id'],queues[provider][0].get('case_revision')) in failed:
                        row=queues[provider].pop(0);completed.append({'id':row['id'],'provider':provider,'case_revision':row.get('case_revision'),'status':'preflight_failed'})
                    if not queues[provider]:continue
                    row=queues[provider][0]
                    if (preflight_directory(row)/'receipt.json').exists():
                        queues[provider].pop(0);launch(lane,row,'series')
                    elif any(x['stage']=='preflight' and x['id']==row['id'] for x in lanes.values()):
                        continue
                    else:launch(lane,row,'preflight')
                snapshot()
                if not lanes and (not can_launch or not any(queues.values())):break
                time.sleep(10)
            record['status']='window_complete' if time.time()>=deadline-4500 else 'queue_stopped' if control else 'queue_complete'
        except BaseException:
            record['status']='scheduler_error'
            raise
        finally:
            snapshot();record['finished_at']=now();save(directory,record)
            # Running children remain owned by the campaign and are explicitly
            # recorded for safe adoption. Never kill solvers due to coordinator failure.
            for stream in streams.values():stream.close()


if __name__=='__main__':main()
