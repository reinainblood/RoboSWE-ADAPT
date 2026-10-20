"""Experimental trusted-action replay audit. Never executes actor code or changes grades.

The injected factory receives original metadata and an audit-only output folder.
It must return a FRESH trusted runtime with metadata(), start_task(0), observe(),
bridge(message), cached_predicate() -> bool, finish_task(code_errors=0), and close().
The factory must explicitly certify non-mutating inspection and cached predicate
access through action_audit_capabilities. No extra upstream _check_success call
is permitted. The original trusted bridge predicate-evaluation schedule must be preserved,
including multiple native evaluations per tick. Observation-triggered predicate
evaluation is unsupported; no observe-spam behavior is assumed.
No default factory or command launches a simulator. Hashes bind local receipts;
they do not authenticate a journal supplied by an untrusted coordinator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from roboswe.library.attempts import read_journal


REQUIRED_CAPABILITIES = {
    'contract': 'cached_post_step_predicate_v1',
    'cached_predicate_non_mutating': True,
    'metadata_non_mutating': True,
    'observe_non_mutating': True,
    'start_task_uses_cached_predicate': True,
    'finish_task_uses_cached_predicate': True,
    'original_trusted_bridge_predicate_schedule_preserved': True,
    'predicate_evaluation_schedule': 'same_as_original_trusted_runtime',
    'observation_calls_do_not_evaluate_predicate': True,
}


def validate_runtime_capabilities(runtime):
    capabilities = getattr(runtime, 'action_audit_capabilities', None)
    if not isinstance(capabilities, dict):
        raise UnsupportedActionAudit('Trusted cached-predicate and non-mutating inspection capability certification required')
    for key, expected in REQUIRED_CAPABILITIES.items():
        value = capabilities.get(key)
        if value != expected or (expected is True and value is not True):
            raise UnsupportedActionAudit('Unsupported action-audit capability: ' + key)
    if not callable(getattr(runtime, 'cached_predicate', None)):
        raise UnsupportedActionAudit('Explicit non-mutating cached_predicate accessor required')
    return dict(capabilities)


class UnsupportedActionAudit(ValueError):
    pass


def _read(path):
    return json.loads(path.read_text())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def inspect_source(directory):
    """Validate finalized evidence without a simulator or authored-code import."""
    directory = Path(directory)
    job, report = _read(directory/'job.json'), _read(directory/'report.json')
    if job.get('artifact_errors') or not job.get('finished_at') or job.get('status') not in {'verified_pass', 'verified_fail', 'verification_invalid', 'observed_pass', 'observed_fail'}:
        raise UnsupportedActionAudit('A finalized original attempt is required')
    entries = read_journal(directory/'submissions.jsonl')
    if not entries or entries[0]['kind'] != 'initial':
        raise UnsupportedActionAudit('Initial provenance missing')
    initial = entries[0]['data']; spec = initial.get('spec', {})
    if spec.get('mode') != 'serial' or len(spec.get('tasks', [])) != 1:
        raise UnsupportedActionAudit('Only single-task serial episodes supported')
    for key in ('config', 'initial_fingerprint', 'spec_sha256', 'bindings', 'code_image'):
        if key not in initial:
            raise UnsupportedActionAudit('Missing provenance: '+key)
    if initial['spec_sha256'] != spec.get('sha256') or not initial['code_image'].startswith('sha256:'):
        raise UnsupportedActionAudit('Unpinned spec or authored-code image')
    image = spec.get('docker_simulator_image') or spec.get('modal_image_id') or initial.get('simulator_image')
    if not isinstance(image, str) or not image.startswith(('sha256:', 'im-')):
        raise UnsupportedActionAudit('Pinned simulator image required')
    settle, limit = spec.get('settle_steps'), spec.get('max_control_steps_per_task')
    if type(settle) is not int or settle < 1 or type(limit) is not int or limit < 1:
        raise UnsupportedActionAudit('Explicit settle and control budgets required')
    starts = [e for e in entries if e['kind'] == 'start_task_request']
    finishes = [e for e in entries if e['kind'] == 'finish_task']
    if len(starts) != 1 or starts[0]['data'].get('index') != 0 or len(finishes) != 1:
        raise UnsupportedActionAudit('Exactly one completed journal task required')
    if any(e['kind'] == 'reset_code_workspace' for e in entries):
        raise UnsupportedActionAudit('Workspace reset journal unsupported')
    actions = [json.loads(line) for line in (directory/'actions.jsonl').read_text().splitlines()]
    if len(actions) > limit or report.get('control_steps') != len(actions):
        raise UnsupportedActionAudit('Action count disagrees with budget/report')
    digest = hashlib.sha256(); streak = 0
    for index, row in enumerate(actions, 1):
        if type(row.get('control_step')) is not int or row['control_step'] != index:
            raise UnsupportedActionAudit('Action steps must be contiguous starting at one')
        action = row.get('action')
        if not isinstance(action, list) or not action or not all(_finite(x) for x in action):
            raise UnsupportedActionAudit('Native actions must be finite numeric vectors')
        if type(row.get('upstream_success')) is not bool or type(row.get('sustained_steps')) is not int:
            raise UnsupportedActionAudit('Original predicate/streak observations missing')
        streak = streak+1 if row['upstream_success'] else 0
        if row['sustained_steps'] != streak:
            raise UnsupportedActionAudit('Original streak log inconsistent')
        digest.update(json.dumps(action, allow_nan=False, separators=(',', ':')).encode()+b'\n')
    final = finishes[0]['data']
    if report.get('precondition_valid') is not True or type(report.get('observed_success')) is not bool:
        raise UnsupportedActionAudit('Original precondition or physical verdict unavailable')
    for key in ('control_steps','observed_success','precondition_valid','start_fingerprint','end_fingerprint','verdict'):
        if final.get(key) != report.get(key):
            raise UnsupportedActionAudit('Journal finish disagrees with report: '+key)
    for original in (report, final):
        if original.get('cumulative_action_sha256') != digest.hexdigest() or original.get('start_fingerprint') != initial['initial_fingerprint']:
            raise UnsupportedActionAudit('Action digest or initial fingerprint disagrees')
    wait_ticks = 0
    for e in entries:
        if e['kind'] == 'operation_request' and e['data'].get('decision', {}).get('operation') == 'wait' and e['data'].get('policy') != 'passive_wait_v1':
            raise UnsupportedActionAudit('Unknown passive-wait policy')
        if e['kind'] == 'operation_result' and e['data'].get('operation') == 'wait':
            wait = e['data'].get('result', {}).get('passive_wait', {})
            advanced, requested = wait.get('advanced_steps'), wait.get('requested_steps')
            if type(advanced) is not int or type(requested) is not int or not 0 <= advanced <= requested <= 600:
                raise UnsupportedActionAudit('Invalid passive-wait receipt')
            wait_ticks += advanced
    if wait_ticks != report.get('passive_wait_control_steps', 0) or wait_ticks > min(1200, len(actions)):
        raise UnsupportedActionAudit('Passive wait ticks disagree with original report')
    checkpoints = {}
    transcript = directory/'model-transcript.jsonl'
    if transcript.exists():
        for line in transcript.read_text().splitlines():
            row = json.loads(line)
            if row.get('event') != 'request':
                continue
            prompt = json.loads(row['prompt']); observation = prompt.get('observation', {})
            step = observation.get('control_step')
            if type(step) is int and 0 <= step <= len(actions):
                projection = {k:observation[k] for k in ('sensors','camera_calibration','action_schema','simulator_phase','task_success') if k in observation}
                checkpoints.setdefault(step, []).append(projection)
    files = ('job.json','report.json','submissions.jsonl','actions.jsonl','model-transcript.jsonl','verification.json')
    hashes = {name:_sha(directory/name) for name in files if (directory/name).exists()}
    source_snapshot = directory.parent / str(job.get('campaign_id', '')) / 'source.zip'
    snapshot_hash = _sha(source_snapshot) if job.get('campaign_id') and source_snapshot.is_file() else None
    return {'initial':initial, 'report':report, 'actions':actions, 'checkpoints':checkpoints,
            'hashes':hashes, 'campaign_source_snapshot_sha256':snapshot_hash, 'passive_wait_ticks':wait_ticks, 'settle_steps':settle}


def audit(directory, runtime_factory, output):
    """Audit recorded native actions; output must be new and outside source job."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or directory in output.parents:
        raise ValueError('Audit output must be outside the original job')
    source = inspect_source(directory)
    output.mkdir(parents=True, exist_ok=False)
    result = {'kind':'optional_trusted_action_replay_v1', 'changes_official_score':False,
              'actor_code_executed':False, 'runtime_replay_performed':False,
              'source_sha256':source['hashes'], 'campaign_source_snapshot_sha256':source['campaign_source_snapshot_sha256'], 'trust':'Trusted coordinator receipt consistency only; hashes do not authenticate untrusted uploads. Capability declarations are a trusted adapter contract, not independent proof of non-mutation.',
              'timing_scope':'Audit clock is not scored. No original model delays replayed; physics is paused between recorded ticks. All passive-wait ticks are already in the action log; no extra wait or settle ticks are synthesized.',
              'source_passive_wait_ticks':source['passive_wait_ticks'], 'mismatches':[]}
    runtime = None
    try:
        runtime = runtime_factory(copy.deepcopy(source['initial']), output/'runtime')
        result['runtime_capability_certification'] = validate_runtime_capabilities(runtime)
        actual = runtime.metadata()
        for key in ('config','initial_fingerprint','spec_sha256','spec','bindings','simulator_image','development_runtime'):
            if actual.get(key) != source['initial'].get(key):
                raise UnsupportedActionAudit('Runtime provenance mismatch: '+key)
        start = runtime.start_task(0)
        if start.get('precondition_valid') is not True or runtime.cached_predicate() is not False:
            raise UnsupportedActionAudit('Initial predicate must be false')
        observation = start['observation']; schema = observation.get('action_schema', {})
        shape, low, high = schema.get('shape'), schema.get('low'), schema.get('high')
        if schema.get('format') != 'native_vector' or not isinstance(shape,list) or len(shape)!=1 or type(shape[0]) is not int or shape[0]<1:
            raise UnsupportedActionAudit('Explicit native-vector schema required')
        dimension = shape[0]
        if not isinstance(low,list) or not isinstance(high,list) or len(low)!=dimension or len(high)!=dimension or not all(_finite(v) for v in low+high) or any(a>b for a,b in zip(low,high)):
            raise UnsupportedActionAudit('Finite action bounds required')
        for row in source['actions']:
            if len(row['action']) != dimension or any(not lo <= value <= hi for value,lo,hi in zip(row['action'],low,high)):
                raise UnsupportedActionAudit('Recorded action outside native dimension/bounds')
        def compare_observation(step, obs):
            if obs.get('control_step') != step:
                raise UnsupportedActionAudit('Runtime observation step mismatch')
            for expected in source['checkpoints'].get(step, []):
                if any(obs.get(k) != v for k,v in expected.items()):
                    result['mismatches'].append({'step':step,'reason':'observation_checkpoint_diverged'})
        compare_observation(0, observation)
        streak = 0
        for row in source['actions']:
            observation = runtime.bridge({'operation':'step','action':row['action']})
            predicate = runtime.cached_predicate()
            if type(predicate) is not bool:
                raise UnsupportedActionAudit('Cached predicate accessor must return boolean')
            streak = streak+1 if predicate else 0
            if predicate != row['upstream_success'] or streak != row['sustained_steps']:
                result['mismatches'].append({'step':row['control_step'],'reason':'predicate_trajectory_diverged'})
            compare_observation(row['control_step'], observation)
        replay = runtime.finish_task(code_errors=0)
        recomputed = streak >= source['settle_steps']
        for key in ('start_fingerprint','end_fingerprint','control_steps','cumulative_action_sha256','precondition_valid','observed_success','verdict'):
            if replay.get(key) != source['report'].get(key):
                result['mismatches'].append({'field':key,'reason':'final_report_diverged'})
        if replay.get('observed_success') is not recomputed:
            result['mismatches'].append({'reason':'runtime_verdict_disagrees_with_recomputed_streak'})
        result.update(runtime_replay_performed=True, replay=replay, recomputed_success=recomputed,
                      status='consistent' if not result['mismatches'] else 'diverged')
    except Exception as exc:
        result.update(status='unsupported_or_incomplete', error_type=type(exc).__name__, error=str(exc)[:1000])
    finally:
        if runtime is not None:
            try:runtime.close()
            except Exception as exc:result.update(status='cleanup_failed', cleanup_error=type(exc).__name__)
        if any(not (directory/name).is_file() or _sha(directory/name)!=sha for name,sha in source['hashes'].items()):
            result.update(status='source_changed_during_audit')
        (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
