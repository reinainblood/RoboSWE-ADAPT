"""Read-only, reproducible coverage and latency audit for the frozen cohort."""
import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    if not path.exists():
        return []
    rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:break  # A live writer may be midway through its final line.
    return rows


def distribution(values):
    return {'count':len(values),'sum':sum(values),'median':statistics.median(values) if values else None,
            'maximum':max(values) if values else None}


def latency(events, *, terminal=False):
    calls=[e for e in events if e['event']=='model_end']
    operations=[e for e in events if e['event']=='operation_end']
    moves=[e for e in operations if (e.get('accepted_control_steps') or 0)>0 and e.get('operation')!='wait']
    starts=[e for e in events if e['event']=='model_start']
    finished={e['turn'] for e in events if e['event'] in ('model_end','model_timeout','model_error')}
    pending=[e for e in starts if e['turn'] not in finished]
    waits=[e for e in operations if e.get('operation')=='wait']
    gaps=[]
    wall_gaps=[]
    for a,b in zip(moves,moves[1:]):
        left,right=a['epoch'],b['epoch']-b['seconds']
        wall_gap=max(0,right-left)
        deliberate_wait=sum(max(0,min(right,w['epoch'])-max(left,w['epoch']-w['seconds'])) for w in waits)
        wall_gaps.append(wall_gap);gaps.append(max(0,wall_gap-deliberate_wait))
    inputs=[e for e in events if e['event']=='input_ready']
    first_start=starts[0]['epoch'] if starts else None
    return {'completed_model_requests_seconds':distribution([e['seconds'] for e in calls]),
        'failed_or_timed_out_model_requests_seconds':distribution([e['seconds'] for e in events if e['event'] in ('model_timeout','model_error')]),
        'execution_seconds':distribution([e['seconds'] for e in operations]),
        'observation_seconds':distribution([e['seconds'] for e in events if e['event']=='observe_end']),
        'movement_batches':len(moves),'completed_physics_ticks_in_movement_batches':sum(e['accepted_control_steps'] for e in moves),
        'time_to_first_movement_start_seconds':max(0,moves[0]['epoch']-moves[0]['seconds']-first_start) if moves and first_start is not None else None,
        'between_movement_idle_seconds':distribution(gaps),
        'between_movement_wall_gap_seconds':distribution(wall_gaps),
        'passive_wait_execution_seconds':distribution([e['seconds'] for e in waits]),
        'long_gaps_over_15_minutes':sum(gap>=900 for gap in gaps),
        'operations_without_movement':dict(Counter(e['operation'] for e in operations if not e.get('accepted_control_steps') and e['operation']!='wait')),
        'input_text_bytes':sum(e['prompt_bytes'] for e in inputs),
        'new_camera_bytes':sum(e['image_bytes'] for e in inputs),
        'requests_reusing_prior_observation':sum(e['images']==0 for e in inputs),
        'inflight_model_request_started_epoch':pending[-1]['epoch'] if pending and not terminal else None,
        'terminal_request_with_missing_end_event':bool(pending and terminal),
        'definitions':'Movement means executed control ticks, not successful object manipulation. Idle gaps exclude the following movement execution duration and deliberate passive-wait execution; wall gaps include passive waiting. Model durations include provider and transport latency. Passive waits excluded from movement batches. In-flight requests are not counted as completed.'}


def learning_comparisons(attempts):
    result=[]
    groups={}
    for attempt in attempts:
        key=(attempt['cohort_id'],attempt['provider'],attempt['series_id'])
        groups.setdefault(key,[]).append(attempt)
    for (cohort,provider,series),trials in groups.items():
        completed=[a for a in trials if a['verdict'] is not None and a['status'] in ('observed_pass','observed_fail')]
        first=[a for a in completed if a['attempt_number']==1]
        second=[a for a in completed if a['attempt_number']==2]
        if len(first)!=1 or len(second)!=1:continue
        a,b=first[0],second[0]
        x,y=a['verdict'],b['verdict']
        same_scene=bool(a.get('initial_fingerprint') and a['initial_fingerprint']==b.get('initial_fingerprint'))
        result.append({'cohort_id':cohort,'provider':provider,'series_id':series,'first_job':a['job_id'],'second_job':b['job_id'],
            'native_control_contract_valid':a.get('native_control_contract_valid',True) and b.get('native_control_contract_valid',True),
            'same_initial_scene':same_scene,
            'outcome_transition':('success' if x['observed_success'] else 'failure')+' → '+('success' if y['observed_success'] else 'failure'),
            'elapsed_seconds_change':y['elapsed_seconds']-x['elapsed_seconds'],
            'control_steps_change':y['control_steps']-x['control_steps'],
            'interpretation':'Descriptive retained-learning pair only. A faster failed attempt is not an improvement; no causal effect estimate.'})
    return result


def observed_completed(attempt):
    """A report can appear before video export and terminal job finalization."""
    return (attempt['status'] in ('observed_pass','observed_fail')
        and attempt['verdict'] is not None
        and attempt['verdict']['termination_reason']!='precondition_invalid')


def audit(root, jobs):
    active=json.loads((root/'active.json').read_text())
    manifest=json.loads((root/'cohort.json').read_text())
    records={p.parent.name:(p.parent,json.loads(p.read_text())) for p in jobs.glob('*/job.json')}
    series_ids={ident for ident,(_,j) in records.items() if j.get('campaign_id')==active['job_id'] and j.get('kind')=='cohort50-series'}
    attempts=[]
    for directory,job in records.values():
        if job.get('campaign_id') not in series_ids or job.get('kind')!='cohort50-attempt':continue
        report_path=directory/'report.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else None
        events=read_jsonl(directory/'timing.jsonl')
        actions=read_jsonl(directory/'actions.jsonl')
        continuity=json.loads((directory/'continuity.json').read_text()) if (directory/'continuity.json').exists() else {}
        series_directory=records[job['campaign_id']][0]
        series_plan=json.loads((series_directory/'plan.json').read_text())
        execution_documentation=series_plan.get('execution_limit_documentation')
        journal_path=directory/'submissions.jsonl'
        if execution_documentation is None and journal_path.exists():
            with journal_path.open() as stream:
                try:
                    initial=json.loads(stream.readline())
                    execution_documentation=initial.get('data',{}).get('spec',{}).get('execution_limit_documentation')
                except json.JSONDecodeError:
                    pass  # A starting job may still be writing initial provenance.
        robot=series_plan['cohort']['config']['robot']
        hand_contract=series_plan.get('hand_action_contract','restricted-normalized-bounds-v1' if robot=='GR1' else 'upstream-standard')
        streak=max((r.get('sustained_steps',0) for r in actions),default=0)
        verdict=None
        if report is not None:
            verdict={'observed_success':report.get('observed_success'),'termination_reason':report.get('termination_reason'),
                'control_steps':report.get('control_steps'),'elapsed_seconds':report.get('elapsed_seconds'),
                'action_log_max_success_streak':streak,
                'success_supported_by_action_log':bool(report.get('observed_success') and streak>=10)}
        attempts.append({'job_id':job['id'],'cohort_id':job['cohort_id'],'provider':job['provider'],
            'series_id':job['campaign_id'],'initial_fingerprint':continuity.get('initial_fingerprint'),
            'transport_version':series_plan.get('transport_optimizations','original-serialization'),
            'observation_feedback_version':series_plan.get('observation_feedback_version','observe-duplicate-v1'),
            'budget_feedback_version':series_plan.get('budget_feedback_version','initial-budget-only-v1'),
            'sensor_documentation':series_plan.get('sensor_documentation','original-documentation'),
            'environment_documentation':series_plan.get('environment_documentation','original-documentation'),
            'execution_limit_documentation':execution_documentation or 'attempt-wording-v1',
            'reset_protocol':series_plan.get('reset_protocol','seeded-rng-plus-hashseed-v2'),
            'hand_action_contract':hand_contract,'native_control_contract_valid':robot!='GR1' or hand_contract=='native-raw-position-v2',
            'repair_request_id':records[job['campaign_id']][1].get('repair_request_id'),
            'case_revision':series_plan['cohort'].get('case_revision'),
            'attempt_number':job['attempt_number'],'status':job['status'],'verdict':verdict,
            'live_control_steps':actions[-1]['control_step'] if actions else 0,'latency':latency(events,terminal=job['status'] not in ('running','starting'))})
    coverage=[]
    review_path=root/'case-review-flags.json'
    review_flags=json.loads(review_path.read_text()) if review_path.exists() else {}
    case_revisions={p.stem:json.loads(p.read_text()) for p in (root/'case-revisions').glob('*.json')}
    for row in manifest['tasks']:
        selected=[a for a in attempts if a['cohort_id']==row['id']]
        preflight=root/'preflight'/row['id']
        revision=case_revisions.get(row['id']);config=revision['config'] if revision else row['config']
        if revision:preflight=preflight/revision['revision_id']
        coverage.append({'id':row['id'],'task':row['task'],'robot':config['robot'],
            'layout':config['env_kwargs']['layout_ids'],'style':config['env_kwargs']['style_ids'],
            'case_revision':revision['revision_id'] if revision else None,
            'review_flags':review_flags.get(row['id'],[]),
            'difficulty_prior':row['difficulty_prior'],
            'preflight':'passed' if (preflight/'receipt.json').exists() else 'failed' if (preflight/'error.txt').exists() else 'not_started',
            'attempt_jobs':[a['job_id'] for a in selected],
            'completed_attempts':sum(observed_completed(a) for a in selected),
            'observed_successes':sum(bool(observed_completed(a) and a['verdict']['observed_success']) for a in selected)})
    models={}
    for model in ('astra','qwen'):
        selected=[a for a in attempts if a['provider']==model]
        observed_trials=[a for a in selected if observed_completed(a)]
        completed=[a for a in observed_trials if a['native_control_contract_valid']]
        models[model]={'started_attempts':len(selected),'completed_valid_attempts':len(completed),
            'completed_observed_attempts':len(observed_trials),
            'restricted_control_completed_attempts':sum(not a['native_control_contract_valid'] for a in observed_trials),
            'observed_successes':sum(bool(a['verdict']['observed_success']) for a in observed_trials),
            'evidence_supported_observed_successes':sum(a['verdict']['success_supported_by_action_log'] for a in observed_trials),
            'native_contract_observed_successes':sum(bool(a['verdict']['observed_success']) for a in completed),
            'infrastructure_failures':sum(a['status']=='infrastructure_error' for a in selected),
            'infrastructure_error_series':sum(records[ident][1].get('provider')==model and records[ident][1]['status']=='infrastructure_error' for ident in series_ids),
            'invalid_preconditions':sum(bool(a['verdict'] and a['verdict']['termination_reason']=='precondition_invalid') for a in selected),
            'status_counts':dict(Counter(a['status'] for a in selected))}
    attempt_ids={a['job_id'] for a in attempts}
    replays=[]
    for directory,job in records.values():
        if job.get('kind')!='cohort50-code-replay' or job.get('source_job') not in attempt_ids:continue
        result=json.loads((directory/'report.json').read_text()) if (directory/'report.json').exists() else {}
        replays.append({'job_id':job['id'],'source_job':job['source_job'],'status':job['status'],
            'reproduced':result.get('reproduced'),'mismatches':result.get('mismatches'),
            'final_fingerprint_matches':result.get('final_fingerprint_matches'),
            'verified_success':job['status']=='reproduced_success' and result.get('reproduced') is True and result.get('final_fingerprint_matches') is True})
    return {'generated_epoch':time.time(),'campaign_id':active['job_id'],'cohort_sha256':manifest['sha256'],
        'resource_history':{
            'capacity_rollout':json.loads((root/'capacity-rollout.json').read_text()) if (root/'capacity-rollout.json').exists() else None,
            'capacity_rollouts':json.loads((root/'capacity-rollouts.json').read_text()) if (root/'capacity-rollouts.json').exists() else None,
            'capacity_probes':[json.loads(p.read_text()) for p in sorted(root.glob('capacity-probe-*/report.json'))],
            'verification_load_windows':json.loads((root/'verification-load-windows.json').read_text()) if (root/'verification-load-windows.json').exists() else None,
            'interpretation':'Wall-clock comparisons must account for recorded transport versions and host-load phases, including the bounded capacity-probe interval.'},
        'unique_tasks_started':sum(bool(c['attempt_jobs']) for c in coverage),'selected_tasks':len(coverage),
        'model_summary':models,'coverage':coverage,'case_revisions':case_revisions,'attempts':attempts,'learning_pairs':learning_comparisons(attempts),'independent_code_replays':replays,
        'interpretation':'Pilot outcomes. completed_valid_attempts excludes invalid preconditions and the known old GR1 raw-hand range mismatch. Restricted-control outcomes remain in completed_observed_attempts and full traces. Only explicitly listed successful independent code replays are replay-verified; no causal ICL claim. Unstarted tasks and infrastructure failures are not model task failures. Live counters are provisional; recorded physics ticks do not imply useful contact or task progress.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('artifacts/cohort50-20261005'));parser.add_argument('--jobs',type=Path,default=Path('jobs'))
    args=parser.parse_args();report=audit(args.root,args.jobs)
    path=args.root/'performance.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
    print(json.dumps({'report':str(path),'tasks_started':report['unique_tasks_started'],'model_summary':report['model_summary']}))


if __name__=='__main__':main()
