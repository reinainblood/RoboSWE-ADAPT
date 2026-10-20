"""Light evidence dashboard and explicit static exports; no robot execution."""
import argparse
import hashlib
import json
import re
import shutil
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from roboswe.server import Handler, JOB_ID, job_records

ASSETS = Path(__file__).parent/'viewer'
FILES = ('model-transcript.jsonl','submissions.jsonl','actions.jsonl','timing.jsonl',
         'reasoning-summaries.jsonl','context-events.jsonl')


def read_json(path, default=None):
    if not path.is_file() or path.is_symlink(): return default
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return default


def rows(path):
    if not path.is_file() or path.is_symlink(): return []
    result=[]
    for line in path.read_text().splitlines():
        try: result.append(json.loads(line))
        except ValueError: continue # In-progress final line may be incomplete.
    return result


def episode(directory):
    job=read_json(directory/'job.json',{})
    report=read_json(directory/'report.json',{})
    transcript=rows(directory/'model-transcript.jsonl')
    turns={}
    for event in transcript:
        n=event.get('turn')
        if n is None: continue
        t=turns.setdefault(n,{'turn':n,'reasoning':[]})
        if event.get('event')=='request':
            try: t['input']=json.loads(event['prompt'])
            except (ValueError,KeyError): t['input']=event.get('prompt')
        elif event.get('event')=='decision': t['decision']=event['decision']
        elif event.get('event')=='provider_thinking': t['reasoning'].append(event.get('trace'))
    # Older direct Anthropic receipts live at series scope. Match exact final
    # decisions, never infer reasoning ownership from line order or timestamps.
    parent_id=job.get('campaign_id','')
    if JOB_ID.fullmatch(parent_id):
        parent=directory.parent/parent_id
        if not parent.is_symlink():
            responses=rows(parent/'provider-responses.jsonl')
            for t in turns.values():
                if t['reasoning'] or 'decision' not in t: continue
                matches=[]
                for event in responses:
                    blocks=event.get('response',{}).get('content',[])
                    for block in blocks:
                        if block.get('type')!='text': continue
                        try: decision=json.loads(block.get('text',''))
                        except ValueError: continue
                        if decision==t['decision']:
                            matches.append({'source_job':parent_id,'call':event.get('call'),
                                'blocks':[b for b in blocks if b.get('type') in ('thinking','redacted_thinking')]})
                if len(matches)==1: t['reasoning']=matches
    # Keep summaries separate: there may be no reliable trial/turn mapping.
    source_events={f:rows(directory/f) for f in FILES if (directory/f).exists()}
    ordered=list(turns.values())
    actions=source_events.get('actions.jsonl',[])
    for index,t in enumerate(ordered):
        following=ordered[index+1].get('input',{}) if index+1<len(ordered) else {}
        current=t.get('input',{})
        if isinstance(following,dict) and 'feedback' in following: t['result']=following['feedback']
        start=current.get('observation',{}).get('control_step') if isinstance(current,dict) else None
        end=following.get('observation',{}).get('control_step',report.get('control_steps')) if isinstance(following,dict) else None
        if isinstance(start,int) and isinstance(end,int):
            t['control_steps_before']=start;t['control_steps_after']=end
            t['robot_actions']=[a for a in actions if start<a.get('control_step',-1)<=end]
    manifest=read_json(directory/'video.json',job.get('video',{}))
    videos=[]
    for v in manifest.get('views',[]):
        name=v.get('simulation','')
        if name and Path(name).name==name and (directory/name).is_file() and not (directory/name).is_symlink():
            videos.append({'name':name,'label':v.get('label',v.get('id',name))})
    audits=[read_json(p) for p in sorted((directory/'predicate-audits').glob('*/audit.json'))]
    audits=[a for a in audits if a]
    from .library.predicate_audit import comparison_status
    for audit in audits:
        # Read-only projection for historical audit schemas; never rewrite evidence.
        audit.setdefault('comparison_status', comparison_status(
            audit.get('answer', {}).get('visible_success'), audit.get('mechanical_success'),
            failed=audit.get('status') == 'judge_failed'))
        audit.setdefault('predicate_disagreement', audit['comparison_status'] == 'visual_mechanical_disagreement')
    return {'schema_version':'roboswe-episode-v1','job':job,'report':report,
            'turns':list(turns.values()),'events':source_events,'videos':videos,'audits':audits,
            'source_hashes':{f:hashlib.sha256((directory/f).read_bytes()).hexdigest() for f in FILES if (directory/f).is_file() and not (directory/f).is_symlink()},
            'reasoning_notice':'Only provider-exposed reasoning is available. Public rationale is not hidden reasoning. Unmapped summaries remain separate; missing traces are not reconstructed.'}


def atif(data):
    steps=[]
    for t in data['turns']:
        if 'input' in t: steps.append({'step_id':len(steps)+1,'source':'user','message':json.dumps(t['input'])})
        if 'decision' in t:
            d=t['decision']; step={'step_id':len(steps)+1,'source':'agent','message':d.get('message',''),
                'tool_calls':[{'tool_call_id':f"turn-{t['turn']}",'function_name':d['operation'],'arguments':{'path':d.get('path',''),'content':d.get('content','')}}],
                'extra':{'roboswe_turn':t['turn']}}
            if 'result' in t: step['observation']={'results':[{'source_call_id':f"turn-{t['turn']}",'content':json.dumps(t['result'])}]}
            step['extra'].update({k:t[k] for k in ('control_steps_before','control_steps_after','robot_actions') if k in t})
            if t['reasoning']: step['reasoning_content']=json.dumps(t['reasoning'])
            steps.append(step)
    return {'schema_version':'ATIF-v1.6','session_id':data['job']['id'],
            'agent':{'name':'RoboSWE','version':'0.1.0'},'steps':steps,
            'notes':'Derived model-turn projection; full tool outcomes/control steps and unmapped reasoning are preserved in roboswe-episode-v1 export. Not a new execution.'}


class ReviewHandler(Handler):
    lock=threading.Lock()
    active=set()
    def do_GET(self):
        route=urlsplit(self.path).path
        if route=='/': return self.file(ASSETS/'review.html')
        if route=='/review.js': return self.file(ASSETS/'review.js')
        if route=='/api/review-jobs':
            records=job_records(self.jobs_dir)
            return self.respond(json.dumps([{'id':j['id'],'name':j['name'],'status':j['status'],'kind':j['kind'],'finished_at':j.get('finished_at')} for j in records]).encode(),'application/json')
        match=re.fullmatch(r'/api/episode/([\w-]+)(/atif)?',route)
        if match and JOB_ID.fullmatch(match[1]):
            directory=self.jobs_dir/match[1]
            if directory.is_symlink() or not (directory/'job.json').is_file(): return self.send_error(404)
            data=episode(directory)
            with self.lock: data['audit_running']=match[1] in self.active
            data['audit_error']=self.server.audit_errors.get(match[1])
            return self.respond(json.dumps(atif(data) if match[2] else data).encode(),'application/json')
        return super().do_GET()
    def do_POST(self):
        # No CORS. A custom header forces cross-origin browser preflight; Host and
        # Origin checks also reject DNS rebinding and cross-site form submissions.
        port=self.server.server_port
        hosts={f'127.0.0.1:{port}',f'localhost:{port}'}
        host=self.headers.get('Host','')
        if host not in hosts or self.headers.get('Origin') not in {None,'http://'+host} or self.headers.get('X-RoboSWE-Review')!='1':
            return self.send_error(403)
        match=re.fullmatch(r'/api/audit/([\w-]+)',urlsplit(self.path).path)
        if not match or not JOB_ID.fullmatch(match[1]): return self.send_error(404)
        directory=self.jobs_dir/match[1]
        if directory.is_symlink(): return self.send_error(404)
        record=read_json(directory/'job.json',{})
        if not record.get('finished_at') or record.get('status') in ('running','starting'): return self.send_error(409,'Episode is not finalized')
        if not self.server.enable_judge: return self.send_error(403,'Judge disabled; start with --enable-judge and server-side OPENROUTER_API_KEY')
        with self.lock:
            if self.active: return self.send_error(409,'An audit is already running; no duplicate request sent')
            self.active.add(match[1])
        def work():
            try:
                from .library.predicate_audit import audit_job
                audit_job(directory, judge_backend=self.server.judge_backend,
                          reasoning_effort='low' if self.server.judge_backend == 'gemini' else None,
                          max_tokens=8192, request_fn=self.server.judge_request)
            except Exception as exc:
                # Error text here is internal validation only; provider errors are sanitized.
                self.server.audit_errors[match[1]]=type(exc).__name__+': '+str(exc)
            finally:
                with self.lock: self.active.discard(match[1])
        threading.Thread(target=work,daemon=True).start()
        return self.respond(b'{"status":"started"}','application/json')


def export(root, output, ids):
    """Explicit selected-job export; never exports the whole workstation."""
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(ASSETS/'review.html',output/'index.html')
    shutil.copy2(ASSETS/'review.js',output/'review.js')
    records=[]
    snapshots={}
    for job_id in ids:
        if not JOB_ID.fullmatch(job_id): raise ValueError('Invalid job ID')
        directory=root/job_id
        if directory.is_symlink(): raise ValueError('Symlinked job')
        data=episode(directory)
        snapshots[job_id]=data
        dest=output/'jobs'/job_id;dest.mkdir(parents=True)
        (dest/'episode.json').write_text(json.dumps(data))
        (dest/'atif.json').write_text(json.dumps(atif(data)))
        for v in data['videos']: shutil.copy2(directory/v['name'],dest/v['name'])
        records.append({k:data['job'].get(k) for k in ('id','name','status','kind','finished_at')})
    (output/'review-jobs.json').write_text(json.dumps(records))
    (output/'snapshot.js').write_text('window.ROBOSWE_SNAPSHOT='+json.dumps({'records':records,'episodes':snapshots})+';')
    index=output/'index.html'
    index.write_text(index.read_text().replace('<script src="./review.js"', '<script src="./snapshot.js"></script><script src="./review.js"'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs-dir',type=Path,default=Path('jobs'))
    p.add_argument('--port',type=int,default=8766)
    p.add_argument('--enable-judge',action='store_true')
    p.add_argument('--judge-backend', choices=['gemini', 'glimpse'], default='gemini')
    p.add_argument('--judge-ledger', type=Path, help='Explicit shared USD spend ledger for paid audits')
    p.add_argument('--judge-spend-cap', type=float, help='Explicit aggregate USD cap for that ledger')
    p.add_argument('--export',type=Path)
    p.add_argument('--job',action='append',default=[])
    args=p.parse_args()
    if args.export:
        if not args.job: p.error('--export requires explicit --job selections')
        export(args.jobs_dir.resolve(),args.export,args.job); return
    judge_request = None
    if args.enable_judge:
        if args.judge_ledger is None or args.judge_spend_cap is None:
            p.error('--enable-judge requires --judge-ledger and --judge-spend-cap')
        from .library.judge_budget import budgeted_request
        try:
            judge_request = budgeted_request(ledger_path=args.judge_ledger,
                limit_usd=args.judge_spend_cap, judge_backend=args.judge_backend,
                receipts=args.judge_ledger.parent / (args.judge_ledger.stem + '-receipts'))
        except ValueError as exc:
            p.error(str(exc))
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(ReviewHandler,jobs_dir=args.jobs_dir))
    server.enable_judge=args.enable_judge;server.audit_errors={}
    server.judge_backend=args.judge_backend;server.judge_request=judge_request
    print(f'RoboSWE review: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()

if __name__=='__main__': main()
