import json

import pytest

from roboswe.adaptive_results import PROTOCOL, export_results


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def series(root):
    put(root / 'series/job.json', {'id': 'series', 'kind': 'adaptive-direct-series', 'protocol': PROTOCOL, 'task': 'Example', 'status': 'running'})
    put(root / 'series/plan.json', {'protocol': PROTOCOL, 'trials': 5, 'participant': {'model': 'test', 'provider': 'test'}})


def attempt(root, number, success, verified=False, steps=100):
    directory = root / str(number)
    put(directory / 'job.json', {'id': str(number), 'kind': 'adaptive-direct-attempt', 'campaign_id': 'series', 'attempt_number': number, 'status': 'observed_pass' if success else 'observed_fail'})
    put(directory / 'report.json', {'observed_success': success, 'control_steps': steps, 'elapsed_seconds': 12, 'wall_elapsed_seconds': 15, 'passive_wait_wall_seconds': 3})
    if verified:
        put(directory / 'verification.json', {'kind': 'independent_code_replay', 'reproduced': True, 'verified_success': success, 'trials': [{'reproduced': True, 'infrastructure_valid': True, 'precondition_valid': True, 'observed_success': success, 'verified_success': success, 'replay': {'precondition_valid': True, 'observed_success': success}}]})
    return directory


def test_observed_success_is_not_official_and_denominator_stays_five(tmp_path):
    series(tmp_path)
    attempt(tmp_path, 1, True)
    result = export_results(tmp_path)['series'][0]
    assert result['verified_trials'] == 0
    assert result['incomplete_trials'] == 5
    assert result['success_curve'] == [None] * 5
    assert result['attempts'][0]['category'] == 'awaiting_verification'
    assert result['descriptive_metrics']['complete_series_success_fraction'] is None
    assert result['descriptive_metrics']['eventual_success_fraction_bounds'] == [0, 1]


def test_protocol_review_does_not_erase_verified_outcome(tmp_path):
    series(tmp_path)
    attempt(tmp_path, 1, False, True)
    put(tmp_path / 'series/protocol-review.json', {
        'primary_leaderboard_eligible': False,
        'classification': 'development_context_policy',
        'reason': 'Premature sliding',
    })
    result = export_results(tmp_path)['series'][0]
    assert result['success_curve'] == [False, None, None, None, None]
    assert result['verified_trials'] == 1
    assert result['protocol_review']['primary_leaderboard_eligible'] is False
    assert len(result['protocol_review']['evidence_sha256']) == 64


def test_infrastructure_status_overrides_model_wrapper_error(tmp_path):
    series(tmp_path)
    directory = attempt(tmp_path, 3, False)
    path = directory / 'job.json'
    job = json.loads(path.read_text())
    job['status'] = 'infrastructure_error'
    put(path, job)
    (directory / 'report.json').unlink()
    (directory / 'timing.jsonl').write_text(
        '{"event":"model_error","error_type":"RuntimeError","seconds":30.366}\n')
    result = export_results(tmp_path)['series'][0]
    trial = result['attempts'][2]
    assert trial['category'] == 'infrastructure_failure'
    assert trial['official_success'] is None
    assert result['verified_trials'] == 0
    assert trial['timing']['last_model_call_error_type'] == 'RuntimeError'


def test_anthropic_prompt_range_includes_cached_input(tmp_path):
    series(tmp_path)
    directory = attempt(tmp_path, 1, False)
    (directory/'model-usage.jsonl').write_text(json.dumps({'usage': {
        'input_tokens': 4, 'cache_creation_input_tokens': 100,
        'cache_read_input_tokens': 10000, 'output_tokens': 50,
    }}) + '\n')
    result = export_results(tmp_path)['series'][0]
    assert result['context_audit']['reported_prompt_tokens_range'] == [10104, 10104]


def test_partial_replayed_series_and_timing_do_not_invent_transport(tmp_path):
    series(tmp_path)
    attempt(tmp_path, 1, False, True)
    directory = attempt(tmp_path, 2, True, True)
    (directory / 'timing.jsonl').write_text('{"event":"model_end","seconds":4}\n{"event":"operation_end","seconds":2}\n{"event":')
    result = export_results(tmp_path)['series'][0]
    assert result['success_curve'] == [False, True, None, None, None]
    assert result['descriptive_metrics']['eventual_success_fraction_bounds'] == [0.2, 0.8]
    assert result['descriptive_metrics']['terminal_success_streak'] is None
    timing = result['attempts'][1]['timing']
    assert timing['model_roundtrip_seconds'] == 4
    assert timing['operation_roundtrip_seconds'] == 2
    assert timing['transport_only_seconds'] is None
    assert timing['code_only_seconds'] is None


@pytest.mark.parametrize('fault', ['infrastructure', 'precondition', 'contradiction', 'reproduced', 'missing_trial'])
def test_invalid_replay_never_becomes_pass_or_fail(tmp_path, fault):
    series(tmp_path)
    directory = attempt(tmp_path, 1, True, True)
    path = directory / 'verification.json'
    verification = json.loads(path.read_text())
    if fault == 'infrastructure':
        verification['trials'][0]['infrastructure_valid'] = False
    elif fault == 'precondition':
        verification['trials'][0]['replay']['precondition_valid'] = False
    elif fault == 'contradiction':
        verification['verified_success'] = False
    elif fault == 'reproduced':
        verification['reproduced'] = False
    else:
        verification['trials'] = []
    put(path, verification)
    result = export_results(tmp_path)['series'][0]
    assert result['verified_trials'] == 0
    assert result['attempts'][0]['category'] == 'verification_invalid'


def test_complete_series_descriptive_success_streak_and_pairs(tmp_path):
    series(tmp_path)
    for n, success in enumerate([False, True, True, True, True], 1):
        attempt(tmp_path, n, success, True, steps=200 - 10 * n)
    result = export_results(tmp_path)['series'][0]
    assert result['complete']
    assert result['descriptive_metrics']['terminal_success_streak'] == 4
    assert result['descriptive_metrics']['complete_series_success_fraction'] == .8
    assert len(result['descriptive_metrics']['adjacent_success_pairs']) == 3
    assert result['descriptive_metrics']['adjacent_success_pairs'][0]['control_step_change'] == -10
    assert result['descriptive_metrics']['composite_improvement_score'] is None


def test_duplicate_attempts_rejected(tmp_path):
    series(tmp_path)
    attempt(tmp_path, 1, True, True)
    put(tmp_path / 'duplicate/job.json', {'id': 'duplicate', 'kind': 'adaptive-direct-attempt', 'campaign_id': 'series', 'attempt_number': 1})
    with pytest.raises(ValueError, match='duplicate'):
        export_results(tmp_path)


def test_model_refusal_retained_as_non_score(tmp_path):
    series(tmp_path)
    put(tmp_path / '1/job.json', {'id': '1', 'kind': 'adaptive-direct-attempt', 'campaign_id': 'series', 'attempt_number': 1, 'status': 'model_refusal'})
    result = export_results(tmp_path)['series'][0]
    assert result['attempts'][0]['category'] == 'provider_model_failure'
    assert result['success_curve'][0] is None


def test_context_slides_expose_proxy_native_difference(tmp_path):
    import sqlite3
    series(tmp_path)
    directory = attempt(tmp_path, 1, False)
    with sqlite3.connect(tmp_path / 'series/history.sqlite') as db:
        db.execute('CREATE TABLE slides(id INTEGER, trial TEXT, epoch REAL, removed_messages INTEGER, estimated_removed_tokens INTEGER, estimated_before_tokens INTEGER, estimated_after_tokens INTEGER)')
        db.execute('INSERT INTO slides VALUES(1,"1",1.0,12,200000,1000000,800000)')
    (directory / 'model-usage.jsonl').write_text('{"usage":{"prompt_tokens":150000}}\n')
    (directory / 'context-estimates.jsonl').write_text('{"estimated_input_units":900000,"actual_prompt_tokens":150000}\n')
    audit = export_results(tmp_path)['series'][0]['context_audit']
    assert audit['slide_count'] == 1
    assert audit['reported_prompt_tokens_range'] == [150000, 150000]
    assert audit['proxy_to_reported_prompt_ratio_range'] == [6, 6]
    assert audit['slides'][0]['removed_messages'] == 12


def lines(path, rows, partial=''):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows)+partial)


def test_anthropic_service_latency_deduplicates_response_and_usage(tmp_path):
    from roboswe.adaptive_results import _timing
    receipt={'call':3,'elapsed_seconds':12,'id':'msg3'}
    lines(tmp_path/'model-usage.jsonl',[receipt])
    lines(tmp_path/'provider-responses.jsonl',[{**receipt,'response':{'id':'msg3'}}])
    lines(tmp_path/'token-counts.jsonl',[{'input_tokens':100,'seconds':4,'attempts':2}])
    lines(tmp_path/'provider-request-events.jsonl',[
        {'operation':'count_tokens','attempt':1,'seconds':2,'will_retry':True,'error':{'type':'timeout'}},
        {'operation':'inference','phase':'transmission_started','reservation':'a'},
        {'operation':'inference','phase':'settled','reservation':'a'}])
    lines(tmp_path/'timing.jsonl',[{'event':'model_end','seconds':17}])
    value=_timing(tmp_path,{})
    assert value['model_roundtrip_seconds']==17
    assert value['count_service_completed_roundtrip_seconds']==4
    assert value['count_service_coverage']['failure_event_rows_excluded']==1
    assert value['inference_provider_completed_roundtrip_seconds']==12
    assert value['inference_provider_coverage']['timed_completed_calls']==1
    assert value['inference_provider_coverage']['status']=='complete_for_recorded_transmissions'


def test_mapped_google_count_does_not_add_native_duplicate_or_retry_failures(tmp_path):
    from roboswe.adaptive_results import _timing
    lines(tmp_path/'context-estimates.jsonl',[
        {'event':'count_transport_failure','attempt':1,'seconds':90},
        {'event':'count','count_seconds':95,'transformed_request_sha256':'same'},
        # Identical input is a second real service call, not a duplicate.
        {'event':'count','count_seconds':3,'transformed_request_sha256':'same'},
        {'event':'calibration','actual_prompt_tokens':100},
        {'event':'count_transport_failure','attempt':1,'seconds':90},
        {'event':'count_transport_failure','attempt':2,'seconds':181}], partial='{"event":')
    lines(tmp_path/'token-counts.jsonl',[
        {'method':'google-native-countTokens-v1','seconds':4,'request_sha256':'native'},
        {'method':'google-native-countTokens-v1','seconds':3,'request_sha256':'native'}])
    lines(tmp_path/'model-usage.jsonl',[{'call':1,'usage':{}},{'call':2,'usage':{}}])
    lines(tmp_path/'provider-responses.jsonl',[{'call':1,'seconds':8,'response':{'id':'x'}},{'call':2,'seconds':9,'response':{'id':'y'}}])
    value=_timing(tmp_path,{})
    assert value['count_service_completed_roundtrip_seconds']==98
    assert value['count_service_coverage']['completed_logical_call_receipts']==2
    assert value['count_service_coverage']['mirrored_native_receipts_excluded']==2
    assert value['count_service_coverage']['failure_event_rows_excluded']==3
    assert value['inference_provider_completed_roundtrip_seconds']==17
    assert value['inference_provider_coverage']['status']=='partial_or_unknown'
    assert value['partial_service_receipt_files']==['context-estimates.jsonl']


def test_missing_and_unfinished_service_durations_remain_unknown(tmp_path):
    from roboswe.adaptive_results import _timing
    lines(tmp_path/'token-counts.jsonl',[{'input_tokens':100}])
    lines(tmp_path/'provider-responses.jsonl',[{'call':1,'response':{'id':'x'}}],partial='{"call":2')
    lines(tmp_path/'provider-request-events.jsonl',[
        {'operation':'inference','phase':'transmission_started','reservation':'a'},
        {'operation':'inference','phase':'unsettled_failure','reservation':'a','seconds':50},
        {'operation':'inference','phase':'transmission_started','reservation':'b'}])
    value=_timing(tmp_path,{})
    assert value['count_service_completed_roundtrip_seconds'] is None
    assert value['inference_provider_completed_roundtrip_seconds'] is None
    assert value['count_service_coverage']['missing_duration_receipts']==1
    assert value['inference_provider_coverage']['missing_duration_receipts']==1
    assert value['inference_provider_coverage']['recorded_open_transmissions']==1
    assert value['inference_provider_coverage']['recorded_unsettled_failures']==1
    for key in ('model_compute_seconds','transport_only_seconds','code_only_seconds'):assert value[key] is None


def test_latency_ignores_replay_and_rejects_middle_corruption(tmp_path):
    from roboswe.adaptive_results import _timing
    (tmp_path/'verification').mkdir()
    lines(tmp_path/'verification/model-usage.jsonl',[{'call':1,'elapsed_seconds':1000}])
    assert _timing(tmp_path,{})['inference_provider_completed_roundtrip_seconds'] is None
    (tmp_path/'model-usage.jsonl').write_text('{bad\n{"call":1,"elapsed_seconds":3}\n')
    with pytest.raises(json.JSONDecodeError):_timing(tmp_path,{})


def test_failed_response_duration_is_not_a_completed_inference(tmp_path):
    from roboswe.adaptive_results import _timing
    lines(tmp_path/'provider-responses.jsonl',[{'call':1,'seconds':90,'response':{'error':{'message':'timeout'}}}])
    value=_timing(tmp_path,{})
    assert value['inference_provider_completed_roundtrip_seconds'] is None
    assert value['inference_provider_coverage']['failed_response_rows_excluded']==1
    assert value['inference_provider_coverage']['status']=='partial_or_unknown'


def refusal_review_fixture(tmp_path):
    import hashlib
    job={'id':'child','attempt_number':2,'status':'infrastructure_error'}
    (tmp_path/'job.json').write_text(json.dumps(job))
    response={'call':3,'response':{'model':'google/gemini-4-argon','choices':[{'finish_reason':'content_filter','native_finish_reason':'SAFETY','message':{'content':'','refusal':None}}]}}
    (tmp_path/'provider-responses.jsonl').write_text(json.dumps(response)+'\n')
    review={'schema_version':1,'job_id':'child','job_sha256':hashlib.sha256((tmp_path/'job.json').read_bytes()).hexdigest(),
            'response_file_sha256':hashlib.sha256((tmp_path/'provider-responses.jsonl').read_bytes()).hexdigest(),
            'response_call':3,'choice_index':0,'finish_reason':'content_filter','native_finish_reason':'SAFETY','classification':'model_refusal'}
    (tmp_path/'failure-review.json').write_text(json.dumps(review))
    return job,review


def test_bound_refusal_review_projects_without_rewriting_outcomes(tmp_path):
    from roboswe.adaptive_results import _attempt,validated_failure_review
    job,review=refusal_review_fixture(tmp_path)
    original=(tmp_path/'job.json').read_bytes()
    result=_attempt(tmp_path,job)
    assert result['recorded_status']=='infrastructure_error'
    assert result['category']=='provider_model_failure'
    assert result['failure_classification']=='model_refusal'
    assert result['official_success'] is None
    assert validated_failure_review(tmp_path,job)['response_call']==3
    assert (tmp_path/'job.json').read_bytes()==original


@pytest.mark.parametrize('field,value',[
    ('job_id','different'),('job_sha256','fake'),('response_file_sha256','fake'),
    ('response_call',2),('finish_reason','stop'),('native_finish_reason','OTHER'),
    ('classification','verified_pass'),('choice_index',1),
])
def test_failure_review_rejects_false_binding(tmp_path,field,value):
    from roboswe.adaptive_results import validated_failure_review
    job,review=refusal_review_fixture(tmp_path)
    review[field]=value
    (tmp_path/'failure-review.json').write_text(json.dumps(review))
    with pytest.raises(ValueError):validated_failure_review(tmp_path,job)
