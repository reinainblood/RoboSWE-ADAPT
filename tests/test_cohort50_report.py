from pairing.cohort50_report import latency, read_jsonl


def test_latency_excludes_execution_and_passive_wait_from_idle():
    events=[{'event':'model_start','turn':0,'epoch':100},
        {'event':'model_end','turn':0,'epoch':120,'seconds':20},
        {'event':'operation_end','turn':0,'epoch':130,'seconds':10,'accepted_control_steps':20,'operation':'execute'},
        {'event':'operation_end','turn':1,'epoch':150,'seconds':15,'accepted_control_steps':None,'operation':'wait'},
        {'event':'operation_end','turn':2,'epoch':1180,'seconds':40,'accepted_control_steps':50,'operation':'write_execute'},
        {'event':'model_start','turn':3,'epoch':1181}]
    r=latency(events)
    assert r['movement_batches']==2
    assert r['time_to_first_movement_start_seconds']==20
    assert r['between_movement_idle_seconds']['maximum']==995
    assert r['between_movement_wall_gap_seconds']['maximum']==1010
    assert r['passive_wait_execution_seconds']['sum']==15
    assert r['long_gaps_over_15_minutes']==1
    assert r['completed_model_requests_seconds']['sum']==20
    assert r['inflight_model_request_started_epoch']==1181
    assert r['operations_without_movement']=={}


def test_live_partial_line_and_no_movement(tmp_path):
    p=tmp_path/'timing.jsonl';p.write_text('{"event":"a"}\n{"event":')
    assert read_jsonl(p)==[{'event':'a'}]
    report=latency([])
    assert report['time_to_first_movement_start_seconds'] is None
    assert report['between_movement_idle_seconds']['maximum'] is None


def test_learning_report_keeps_failures_and_series_separate():
    from pairing.cohort50_report import learning_comparisons
    def trial(number,success,seconds,series='s1'):
        return {'cohort_id':'t01','provider':'astra','series_id':series,'attempt_number':number,
            'job_id':f'{series}-{number}','status':'observed_pass' if success else 'observed_fail','initial_fingerprint':'same',
            'verdict':{'observed_success':success,'elapsed_seconds':seconds,'control_steps':number*100}}
    rows=[trial(1,True,100),trial(2,False,80),trial(1,False,1800,'s2')]
    pairs=learning_comparisons(rows)
    assert len(pairs)==1
    assert pairs[0]['outcome_transition']=='success → failure'
    assert pairs[0]['elapsed_seconds_change']==-20
    assert pairs[0]['same_initial_scene']
    assert 'not an improvement' in pairs[0]['interpretation']


def test_terminal_failure_is_not_reported_as_still_waiting():
    events=[{'event':'model_start','turn':0,'epoch':100}]
    assert latency(events)['inflight_model_request_started_epoch']==100
    final=latency(events,terminal=True)
    assert final['inflight_model_request_started_epoch'] is None
    assert final['terminal_request_with_missing_end_event']
    events.append({'event':'model_error','turn':0,'epoch':120,'seconds':20})
    final=latency(events,terminal=True)
    assert final['failed_or_timed_out_model_requests_seconds']['sum']==20
    assert not final['terminal_request_with_missing_end_event']
def test_report_written_before_job_finalized_is_not_completed():
    from pairing.cohort50_report import observed_completed
    attempt={'status':'running','verdict':{'termination_reason':'sustained_success','observed_success':True}}
    assert not observed_completed(attempt)
    attempt['status']='infrastructure_error'
    assert not observed_completed(attempt)
    attempt['status']='observed_pass'
    assert observed_completed(attempt)
