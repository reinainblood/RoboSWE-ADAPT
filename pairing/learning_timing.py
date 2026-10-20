"""Separate active model work, passive physics, and end-to-end artifact overhead."""
import json
from datetime import datetime
from pathlib import Path


def measures(report,job):
    active=report.get('elapsed_seconds')
    passive=report.get('passive_wait_wall_seconds',0.)
    wall=active+passive if active is not None else None
    start=job.get('created_at');end=job.get('finished_at')
    job_wall=(datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds() if start and end else None
    overhead=job_wall-wall if job_wall is not None and wall is not None else None
    return {'active_work_seconds':active,'passive_physics_seconds':passive,
        'episode_wall_seconds':wall,'job_wall_seconds':job_wall,
        'startup_and_finalization_seconds':overhead,
        'job_clock_consistent':job_wall>=0 and (overhead is None or overhead>=-2) if job_wall is not None else None,
        'timing_scope':'Episode wall = active work + credited passive physics. Job wall additionally includes startup and artifact finalization. Missing finish timestamps remain null.'}


def main():
    root=Path('artifacts/muse-learning-20261006');rows=[]
    jobs={p.parent:json.loads(p.read_text()) for p in Path('jobs').glob('*/job.json')}
    for pointer in sorted(root.glob('t??.json')):
        series=Path(json.loads(pointer.read_text())['directory']);plan=json.loads((series/'plan.json').read_text())
        originals={Path(p).resolve() for p in plan.get('original_attempt_jobs',[])}
        for path,job in jobs.items():
            if job.get('campaign_id')!=series.name and path.resolve() not in originals:continue
            report_path=path/'report.json'
            if not report_path.exists():
                rows.append({'task_id':pointer.stem,'job':path.name,'attempt':job.get('attempt_number'),'status':job['status'],'official_success':None,'timing':None});continue
            report=json.loads(report_path.read_text())
            rows.append({'task_id':pointer.stem,'job':path.name,'attempt':job.get('attempt_number'),'status':job['status'],'official_success':report['observed_success'],'termination_reason':report['termination_reason'],'control_steps':report['control_steps'],'passive_wait_control_steps':report.get('passive_wait_control_steps',0),'timing':measures(report,job)})
    output={'rows':rows,'scope':'Latest retained series plus its explicitly referenced completed source attempts. Incomplete episodes are not failures. Active-work time alone is not end-to-end completion time.'}
    (root/'learning-timing.json').write_text(json.dumps(output,indent=2));print(json.dumps({'rows':len(rows),'completed_task_reports':sum(r['official_success'] is not None for r in rows)}))

if __name__=='__main__':main()
