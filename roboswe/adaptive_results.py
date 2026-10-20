"""Read-only, replay-gated export of public adaptive series.

Run ``python -m roboswe.adaptive_results JOBS_DIR`` to emit JSON to stdout.
This exporter never treats a live job record as proof of process liveness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import hashlib
import math
import sqlite3
from pathlib import Path

PROTOCOL = 'adaptive-public-sqlite-v1'


def _read(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def _receipt_rows(path):
    if not path.exists():
        return [], False
    rows = []
    partial = False
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            partial = True
    return rows, partial


def _service_timing(directory):
    files = ('token-counts.jsonl', 'context-estimates.jsonl',
             'model-usage.jsonl', 'provider-responses.jsonl', 'provider-request-events.jsonl')
    loaded = {name: _receipt_rows(directory / name) for name in files}
    counts = loaded['token-counts.jsonl'][0]
    mapped = [r for r in loaded['context-estimates.jsonl'][0] if r.get('event') == 'count']
    # Mapped-count receipts time the complete logical service call, including
    # count-only retries/backoff. A native counter receipt can mirror that call.
    # Never add native and mapped durations or cumulative failure durations.
    mirrored = [r for r in counts if r.get('method') == 'google-native-countTokens-v1'] if mapped else []
    count_values = [r.get('count_seconds') for r in mapped]
    count_values += [r.get('seconds') for r in counts if r not in mirrored]
    timed_counts = [_number(v) for v in count_values if _number(v) is not None]
    request_events = loaded['provider-request-events.jsonl'][0]
    count_failures = sum(r.get('event') == 'count_transport_failure' for r in loaded['context-estimates.jsonl'][0])
    count_failures += sum(r.get('operation') == 'count_tokens' and bool(r.get('error')) for r in request_events)

    # Anthropic duplicates _record() into usage and response logs. Prefer its
    # usage elapsed_seconds; otherwise use the provider response roundtrip.
    calls = {}
    unidentifiable = 0
    failed_response_rows = 0
    for source in ('model-usage.jsonl', 'provider-responses.jsonl'):
        for r in loaded[source][0]:
            response = r.get('response') or {}
            if r.get('error') or response.get('error'):
                failed_response_rows += 1
                continue
            identity = ('call', r['call']) if r.get('call') is not None else ('id', r.get('id') or response.get('id'))
            if identity[1] is None:
                unidentifiable += 1
                continue
            item = calls.setdefault(identity, {'seconds': None, 'source': None})
            value = _number(r.get('elapsed_seconds'))
            if source == 'provider-responses.jsonl' and value is None:
                value = _number(r.get('seconds'))
            if value is not None and item['seconds'] is None:
                item.update(seconds=value, source=source)
    inference_values = [r['seconds'] for r in calls.values() if r['seconds'] is not None]
    starts = {r.get('reservation') for r in request_events if r.get('operation') == 'inference' and r.get('phase') == 'transmission_started' and r.get('reservation')}
    settled = {r.get('reservation') for r in request_events if r.get('operation') == 'inference' and r.get('phase') == 'settled' and r.get('reservation')}
    failed = {r.get('reservation') for r in request_events if r.get('operation') == 'inference' and r.get('phase') in {'unsettled_failure', 'unsettled_usage'} and r.get('reservation')}
    partial_files = [name for name, (_, partial) in loaded.items() if partial]
    return {
        'count_service_completed_roundtrip_seconds': sum(timed_counts) if timed_counts else None,
        'count_service_coverage': {
            'completed_logical_call_receipts': len(count_values),
            'timed_completed_logical_calls': len(timed_counts),
            'missing_duration_receipts': len(count_values) - len(timed_counts),
            'mirrored_native_receipts_excluded': len(mirrored),
            'failure_event_rows_excluded': count_failures,
            'unfinished_call_count': None,
            'status': 'partial_or_unknown',
            'scope': 'Completed logical count-service calls only; mapped/Anthropic totals may include retries and backoff. Failure rows have cumulative elapsed times and are excluded. No count-start log establishes a complete denominator.',
        },
        'inference_provider_completed_roundtrip_seconds': sum(inference_values) if inference_values else None,
        'inference_provider_coverage': {
            'completed_call_receipts': len(calls), 'timed_completed_calls': len(inference_values),
            'missing_duration_receipts': len(calls) - len(inference_values),
            'unidentifiable_rows_excluded': unidentifiable,
            'failed_response_rows_excluded': failed_response_rows,
            'recorded_transmissions_started': len(starts) if starts else None,
            'recorded_unsettled_failures': len(failed),
            'recorded_open_transmissions': len(starts - settled - failed) if starts else None,
            'status': 'complete_for_recorded_transmissions' if starts and not failed and starts == settled and len(starts) == len(inference_values) == len(calls) and not partial_files and not unidentifiable and not failed_response_rows else 'partial_or_unknown',
            'scope': 'Completed provider-request roundtrips, including transport. Anthropic model-usage elapsed_seconds preferred; otherwise provider-responses elapsed_seconds/seconds. Mirrored logs deduplicated by call/id. Failed and unfinished requests are excluded; no pure model compute is inferred.',
        },
        'partial_service_receipt_files': partial_files,
    }


def _timing(directory, report):
    rows, _ = _receipt_rows(directory / 'timing.jsonl')
    def total(events):
        values = [_number(r.get('seconds')) for r in rows if r.get('event') in events]
        known = [v for v in values if v is not None]
        return sum(known) if known else None
    return {
        'last_model_call_error_type': next((r.get('error_type') for r in reversed(rows) if r.get('event') == 'model_error'), None),
        'active_seconds': _number(report.get('elapsed_seconds')),
        'wall_seconds': _number(report.get('wall_elapsed_seconds')),
        'passive_wait_seconds': _number(report.get('passive_wait_wall_seconds')),
        'model_roundtrip_seconds': total({'model_end', 'model_timeout', 'model_error'}),
        'operation_roundtrip_seconds': total({'operation_end'}),
        'observation_roundtrip_seconds': total({'observe_end'}),
        'model_compute_seconds': None,
        'transport_only_seconds': None,
        'code_only_seconds': None,
        **_service_timing(directory),
        'scope': 'Original attempt only; replay and setup excluded. model_roundtrip_seconds is the model wrapper including context counting, inference, and orchestration, including timed wrapper errors. Service-specific totals are components of that wrapper, not additional elapsed time. Operation roundtrips include transport/orchestration, not pure code execution. Missing/unfinished service latency remains unknown.',
    }


def validated_failure_review(directory, job=None):
    """Validate an additive review of a stopped historical OpenRouter refusal.

    This is local evidence binding, not authentication of arbitrary submissions.
    """
    directory = Path(directory)
    path = directory / 'failure-review.json'
    if not path.exists():
        return {}
    review = _read(path)
    fields = {'schema_version', 'job_id', 'job_sha256', 'response_file_sha256',
              'response_call', 'choice_index', 'finish_reason',
              'native_finish_reason', 'classification'}
    if not isinstance(review, dict) or set(review) != fields:
        raise ValueError('Invalid failure-review schema')
    original = _read(directory / 'job.json')
    if job is not None and job != original:
        raise ValueError('Failure review job differs from original file')
    if (review['schema_version'] != 1 or review['classification'] != 'model_refusal'
            or review['job_id'] != original.get('id')
            or original.get('status') not in {'infrastructure_error', 'model_refusal', 'model_error', 'model_output_limit'}):
        raise ValueError('Failure review requires a stopped unverified model attempt')
    for filename, field in [('job.json','job_sha256'), ('provider-responses.jsonl','response_file_sha256')]:
        target = directory / filename
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != review[field]:
            raise ValueError('Failure review source hash mismatch')
    rows, partial = _receipt_rows(directory / 'provider-responses.jsonl')
    if partial or not rows or rows[-1].get('call') != review['response_call']:
        raise ValueError('Failure review must identify the final complete provider response')
    choices = rows[-1].get('response', {}).get('choices', [])
    if type(review['choice_index']) is not int or review['choice_index'] != 0 or len(choices) != 1:
        raise ValueError('Failure review requires exactly one choice at index zero')
    choice = choices[0]
    if any(review[key] != choice.get(key) for key in ('finish_reason','native_finish_reason')):
        raise ValueError('Failure review finish reason mismatch')
    if not (choice.get('finish_reason') == 'content_filter'
            or choice.get('native_finish_reason') == 'SAFETY'
            or (choice.get('message') or {}).get('refusal')):
        raise ValueError('Failure review response does not establish refusal')
    return {**review, 'review_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _attempt(directory, job):
    report = _read(directory / 'report.json')
    verification = _read(directory / 'verification.json')
    trials = verification.get('trials', [])
    trial = trials[0] if len(trials) == 1 else {}
    valid = (
        verification.get('kind') == 'independent_code_replay'
        and verification.get('reproduced') is True
        and verification.get('infrastructure_valid', True) is True
        and trial.get('reproduced') is True
        and trial.get('infrastructure_valid') is True
        and trial.get('precondition_valid') is True
        and trial.get('replay', {}).get('precondition_valid') is True
    )
    official = None
    if valid:
        # Contradictory success evidence is invalid, not a failed attempt.
        verdicts = [verification.get('verified_success'), trial.get('verified_success'),
                    trial.get('observed_success'), trial.get('replay', {}).get('observed_success')]
        if all(v is True for v in verdicts):
            official = True
        elif all(v is False for v in verdicts):
            official = False
        else:
            valid = False
    timing = _timing(directory, report)
    status = job.get('status', 'unknown')
    reason = report.get('termination_reason')
    failure_review = validated_failure_review(directory, job)
    if valid:
        category = 'verified_pass' if official else 'verified_fail'
    elif verification:
        category = 'verification_invalid'
    elif failure_review:
        category = 'provider_model_failure'
    elif status in {'infrastructure_error', 'invalid_precondition', 'verification_invalid'}:
        category = 'infrastructure_failure'
    elif timing['last_model_call_error_type'] or status in {'model_refusal', 'model_output_limit', 'provider_error', 'model_error', 'budget_exhausted', 'funding_blocked'}:
        category = 'provider_model_failure'
    elif report:
        category = 'awaiting_verification'
    else:
        category = 'incomplete'
    return {
        'attempt_number': job['attempt_number'], 'job_id': job['id'],
        'recorded_status': status, 'category': category,
        'failure_review': failure_review,
        'failure_classification': failure_review.get('classification') if failure_review else (status if category == 'provider_model_failure' else None),
        'official_success': official, 'observed_success': report.get('observed_success'),
        'termination_reason': reason,
        'budget_censored': reason in {'task_deadline', 'model_deadline', 'control_step_budget'},
        'control_steps': _number(report.get('control_steps')),
        'passive_wait_control_steps': _number(report.get('passive_wait_control_steps')),
        'code_errors': _number(report.get('code_errors')),
        'timing': timing,
        'evidence': {'report': report != {}, 'verification': verification != {}, 'replay_valid': valid,
                     'files': {name: {'path': str(directory / name), 'sha256': hashlib.sha256((directory / name).read_bytes()).hexdigest()}
                               for name in ('job.json', 'report.json', 'verification.json') if (directory / name).exists()}},
    }


def _context_audit(directory, children):
    slides = []
    path = directory / 'history.sqlite'
    if path.exists():
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            slides = [dict(row) for row in db.execute(
                'SELECT trial, epoch, removed_messages, estimated_removed_tokens, estimated_before_tokens, estimated_after_tokens FROM slides ORDER BY id')]
    counts = []
    ratios = []
    for child, _ in children:
        for filename in ('model-usage.jsonl', 'context-estimates.jsonl'):
            path = child / filename
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if filename == 'model-usage.jsonl':
                    usage = row.get('usage', {})
                    value = usage.get('prompt_tokens', usage.get('input_tokens', usage.get('promptTokenCount')))
                    if 'cache_creation_input_tokens' in usage or 'cache_read_input_tokens' in usage:
                        parts = [usage.get(key, 0) for key in
                                 ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')]
                        value = sum(parts) if all(_number(part) is not None for part in parts) else None
                    if _number(value) is not None:
                        counts.append(value)
                elif _number(row.get('actual_prompt_tokens')) and _number(row.get('estimated_input_units')) is not None:
                    ratios.append(row['estimated_input_units'] / row['actual_prompt_tokens'])
    return {'slide_count': len(slides), 'slides': slides,
            'reported_prompt_tokens_range': [min(counts), max(counts)] if counts else None,
            'proxy_to_reported_prompt_ratio_range': [min(ratios), max(ratios)] if ratios else None,
            'scope': 'Original provider usage and SQLite slide receipts. Estimated slide units are policy-dependent, not necessarily native tokens. Early sliding can confound comparisons across context policies.'}


def export_series(directory, children):
    job = _read(directory / 'job.json')
    plan = _read(directory / 'plan.json')
    metadata = _read(directory / 'model.json')
    review_path = directory / 'protocol-review.json'
    review = _read(review_path)
    if not isinstance(review, dict):
        raise ValueError('Protocol review must be an object')
    public_review = {key: review[key] for key in
                    ('primary_leaderboard_eligible', 'classification', 'reason',
                     'mechanical_result_preserved', 'series_stop', 'future_protocol')
                    if key in review}
    if review:
        public_review['evidence_sha256'] = hashlib.sha256(review_path.read_bytes()).hexdigest()
    if plan.get('protocol') != PROTOCOL or plan.get('trials') != 5:
        raise ValueError(f'{directory.name}: public five-trial protocol required')
    grouped = {}
    for child_dir, child in children:
        number = child.get('attempt_number')
        if type(number) is not int or number not in range(1, 6) or number in grouped:
            raise ValueError(f'{directory.name}: duplicate or invalid attempt number')
        grouped[number] = _attempt(child_dir, child)
    attempts = [grouped.get(n, {'attempt_number': n, 'job_id': None,
        'category': 'not_started', 'official_success': None}) for n in range(1, 6)]
    curve = [a['official_success'] for a in attempts]
    complete = all(v is not None for v in curve)
    successes = sum(v is True for v in curve)
    verified = sum(v is not None for v in curve)
    suffix = None
    if complete:
        suffix = 0
        for success in reversed(curve):
            if not success:
                break
            suffix += 1
    pairs = []
    for a, b in zip(attempts, attempts[1:]):
        if a['official_success'] is True and b['official_success'] is True:
            def delta(field):
                x, y = a.get(field), b.get(field)
                return y - x if x is not None and y is not None else None
            x, y = a['timing']['active_seconds'], b['timing']['active_seconds']
            pairs.append({'from_trial': a['attempt_number'], 'to_trial': b['attempt_number'],
                          'control_step_change': delta('control_steps'),
                          'active_second_change': y - x if x is not None and y is not None else None})
    participant = plan.get('participant', {})
    return {
        'series_id': job['id'], 'task': job.get('task'), 'protocol': PROTOCOL,
        'protocol_review': public_review,
        'recorded_status': job.get('status'), 'liveness_verified': False,
        'participant': {k: participant.get(k) for k in ('provider', 'model', 'effort', 'context_window', 'max_output_tokens')},
        'context_policy': metadata.get('context_policy', participant.get('context_policy')),
        'context_audit': _context_audit(directory, children),
        'limits': plan.get('limits'), 'execution': plan.get('execution'),
        'planned_trials': 5, 'verified_trials': verified, 'incomplete_trials': 5 - verified,
        'complete': complete, 'attempts': attempts, 'success_curve': curve,
        'category_counts': dict(Counter(a['category'] for a in attempts)),
        'descriptive_metrics': {
            'verified_success_count': successes,
            'verified_success_fraction_of_planned': successes / 5,
            'eventual_success_fraction_bounds': [successes / 5, (successes + 5 - verified) / 5],
            'complete_series_success_fraction': successes / 5 if complete else None,
            'terminal_success_streak': suffix,
            'adjacent_success_pairs': pairs,
            'uncertainty': 'Bounds reflect incomplete outcomes, not a confidence interval. Retained trials are dependent; no causal learning estimate or independent-trial binomial interval.',
            'composite_improvement_score': None,
        },
    }


def export_results(jobs_dir):
    root = Path(jobs_dir)
    records = [(p.parent, _read(p)) for p in sorted(root.glob('*/job.json'))]
    series = []
    for directory, job in records:
        if job.get('kind') == 'adaptive-direct-series' and job.get('protocol') == PROTOCOL:
            children = [(p, j) for p, j in records if j.get('kind') == 'adaptive-direct-attempt' and j.get('campaign_id') == job['id']]
            series.append(export_series(directory, children))
    return {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
            'protocol': PROTOCOL, 'series': series,
            'scope': 'Only discovered public five-trial series. Unlaunched dataset tasks are not represented; this is not a full-dataset denominator or ranked leaderboard.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs_dir', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(export_results(args.jobs_dir), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
