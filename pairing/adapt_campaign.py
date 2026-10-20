"""Bounded sequential RoboSWE-ADAPT supervisor; explicit manifests, no retries."""
import argparse
import fcntl
import json
import os
import time
import traceback
from pathlib import Path


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--config',required=True)
    p.add_argument('--state-dir',required=True);p.add_argument('--max-series',type=int,default=1)
    p.add_argument('--jobs-dir',default='jobs');args=p.parse_args(argv)
    if args.max_series<1:raise ValueError('Positive bounded series count required')
    root=Path(args.state_dir);root.mkdir(parents=True,exist_ok=True)
    with (root/'campaign.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'active.json').exists():raise RuntimeError('Existing campaign receipt; no implicit restart or retry')
        from roboswe.adaptive import build_plan,run
        plan=build_plan(args.dataset,args.config,jobs_dir=args.jobs_dir)
        if len(plan['tasks'])>args.max_series:raise ValueError('Dataset exceeds explicitly bounded series count')
        provider=plan['tasks'][0]['participant']['provider']
        if provider=='openrouter':
            from pairing.muse_model import credential
            key_name='OPENROUTER_API_KEY'
        elif provider=='anthropic':
            from pairing.anthropic_model import credential
            key_name='ANTHROPIC_API_KEY'
        else:raise ValueError('Supervisor supports explicitly authorized OpenRouter/Anthropic lanes only')
        previous_key=os.environ.get(key_name)
        os.environ[key_name]=credential()
        state={'pid':os.getpid(),'started_epoch':time.time(),'status':'running','dataset':str(Path(args.dataset).resolve()),
            'config':str(Path(args.config).resolve()),'max_series':args.max_series,'completed':[],'results':[],'completed_series_count':0,
            'retry_policy':'none; provider/model/infrastructure failures stop lane','provider':provider,
            'model':plan['tasks'][0]['participant']['model'],'context_policy':plan['tasks'][0]['context_policy']}
        write(root/'active.json',state)
        try:
            for number,task in enumerate(plan['tasks'],1):
                if (root/'STOP').exists():
                    state['status']='stopped_by_control';break
                state.update(current_task=task['cohort']['id'],series_number=number);write(root/'active.json',state)
                result=run({**plan,'tasks':[task]})
                if len(result)!=1:raise RuntimeError('Expected exactly one returned series directory')
                child=json.loads((Path(result[0])/'job.json').read_text())
                child_status=child.get('status')
                state['results'].append({'directory':result[0],'job_id':child.get('id'),'status':child_status})
                if child_status!='completed_unscored':
                    state.update(status='stopped_child_result',child_status=child_status,stopped_series=result[0])
                    write(root/'active.json',state)
                    break
                state['completed'].extend(result)
                state['completed_series_count']=len(state['completed'])
                write(root/'active.json',state)
            else:state['status']='completed'
        except BaseException as exc:
            state.update(status=getattr(exc,'status','stopped_error'),error_type=type(exc).__name__,error=str(exc)[:1000])
            (root/'error.txt').write_text(traceback.format_exc())
            raise
        finally:
            state['finished_epoch']=time.time();write(root/'active.json',state)
            if previous_key is None:os.environ.pop(key_name,None)
            else:os.environ[key_name]=previous_key


if __name__=='__main__':main()
