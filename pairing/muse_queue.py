"""Bounded Modal campaign supervisor with explicit failure pauses."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from pairing.muse_cohort import ROOT
from pairing.cohort50 import emit,freeze,effective_row

DISK_PAUSE='Local disk below 8 GiB; active trials may finish, no new launches'

def disk_pause_reason(previous,free_gib):
    # Never clear an infrastructure/auth/user failure because disk space improved.
    if previous and previous!=DISK_PAUSE:return previous
    if free_gib<8:return DISK_PAUSE
    if previous==DISK_PAUSE and free_gib>=8.5:return None
    return previous


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resume_file=ROOT/'resume-requests.json'
    resumes=json.loads(resume_file.read_text()) if resume_file.exists() else {}
    rows=[effective_row(row) for row in freeze()['tasks']];order=['t01','t02','t10','t03']+[r['id'] for r in rows if r['id'] not in ('t01','t02','t10','t03')]
    # Complete interrupted series with the fewest remaining trials first.
    # Membership and all results stay fixed; this is scheduling, not selection.
    def priority(task):
        if task not in resumes:return (1,0)
        source=json.loads((Path('jobs')/resumes[task]/'report.json').read_text())
        return (0,3-len(source['trials']))
    order=sorted(order,key=priority)
    active={};completed=[]
    # t01 is the manually supervised canary. Existing series are never duplicated.
    for task in order:
        pointer=ROOT/(task+'.json')
        if pointer.exists():
            job=json.loads((Path(json.loads(pointer.read_text())['directory'])/'job.json').read_text())
            if job['status'] in ('running','starting'):active[task]={'adopted':True,'job':job['id']}
            elif not (job['status']=='infrastructure_error' and resumes.get(task)==job['id']):completed.append({'task_id':task,'status':job['status']})
    adoption=ROOT/'adopt-processes.json'
    if adoption.exists():
        for task,item in json.loads(adoption.read_text()).items():
            if task in active:continue
            command=subprocess.run(['ps','-p',str(item['pid']),'-o','command='],capture_output=True,text=True).stdout
            if f'pairing.muse_cohort series {task}' in command:active[task]={'adopted_pid':item['pid']}
    quarantine=json.loads((ROOT/'quarantine.json').read_text()) if (ROOT/'quarantine.json').exists() else {}
    pending=[task for task in order if task not in active and task not in quarantine and not any(x['task_id']==task for x in completed)]
    completed=[entry for entry in completed if entry['task_id'] not in quarantine]
    completed.extend({'task_id':task,'status':'quarantined_preflight','reason':reason} for task,reason in quarantine.items())
    emit(ROOT/'manifest.json',{'tasks':rows,'model':'meta/muse-spark-1.3-contributor','attempts_per_task':3,'episodes_planned':150,'concurrency':json.loads((ROOT/'control.json').read_text()).get('concurrency',2) if (ROOT/'control.json').exists() else 2,'backend':'Modal FDR L4 simulator, local isolated code','control_arm':'Not launched; optional cold controls awaiting user preference','scope':'Retained learning on exact within-backend resets; historical Astra cross-backend comparison descriptive.'})
    paused=None
    while pending or active:
        control=json.loads((ROOT/'control.json').read_text()) if (ROOT/'control.json').exists() else {}
        free=shutil.disk_usage('.').free/2**30
        paused=disk_pause_reason(paused,free)
        if control.get('pause_launches'):paused=control.get('reason','User/monitor pause')
        for task,item in list(active.items()):
            if item.get('adopted_pid'):
                command=subprocess.run(['ps','-p',str(item['adopted_pid']),'-o','command='],capture_output=True,text=True).stdout
                if f'pairing.muse_cohort series {task}' in command:continue
                pointer=ROOT/(task+'.json')
                job=json.loads((Path(json.loads(pointer.read_text())['directory'])/'job.json').read_text()) if pointer.exists() else {}
                code=0 if job.get('status')=='completed_unscored' else 1
            elif item.get('adopted'):
                job=json.loads((Path('jobs')/item['job']/'job.json').read_text())
                if job['status'] in ('running','starting'):continue
                code=0 if job['status']=='completed_unscored' else 1
            else:
                code=item['process'].poll()
                if code is None:continue
                item['stream'].close()
            completed.append({'task_id':task,'exit_code':code,'epoch':time.time()});del active[task]
            if code:paused=f'{task} infrastructure/preflight failure; inspect before launching more'
        while pending and len(active)<int(control.get('concurrency',2)) and not paused:
            task=pending.pop(0);stream=(ROOT/(task+'.log')).open('a')
            command=[sys.executable,'-u','-m','pairing.muse_cohort','series',task]
            if task in resumes:command+=['--resume-series',resumes[task]]
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONHASHSEED':'0','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            active[task]={'process':proc,'stream':stream,'pid':proc.pid}
        emit(ROOT/'progress.json',{'supervisor_pid':os.getpid(),'pending':pending,'active':{k:{n:v for n,v in x.items() if n not in ('process','stream')} for k,x in active.items()},'completed':completed,'pause_reason':paused,'free_gib':free,'epoch':time.time()})
        if paused and not active and paused!=DISK_PAUSE:break
        time.sleep(10)

if __name__=='__main__':main()
