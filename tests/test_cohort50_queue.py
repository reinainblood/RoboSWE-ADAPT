from pairing.cohort50_queue import ordered_rows, pending_rows


def test_two_robots_early_without_changing_membership():
    rows=[{'id':f't{i:02d}'} for i in range(1,51)]
    ordered=ordered_rows({'tasks':rows})
    assert [r['id'] for r in ordered[:3]]==['t01','t02','t10']
    assert {r['id'] for r in ordered}=={r['id'] for r in rows}


def test_lanes_skip_only_their_own_launched_series():
    rows=[{'id':'t01'},{'id':'t02'}]
    launched=[{'provider':'astra','cohort_id':'t01'}]
    assert pending_rows(rows,'astra',launched)==[{'id':'t02'}]
    assert pending_rows(rows,'qwen',launched)==rows


def test_repair_queue_does_not_silently_rerun_model_failures():
    import pytest
    from pairing.cohort50_queue import repair_row
    request={'request_id':'repair-1','source_series_id':'failed-series','provider':'qwen','task_id':'t01','reason':'Invalid final response recovery'}
    rows=[{'id':'t01','task':'CloseMicrowave'}]
    source={'id':'failed-series','provider':'qwen','cohort_id':'t01','status':'infrastructure_error'}
    result=repair_row(request,rows,[source])
    assert result['retry_of_series']=='failed-series'
    assert result['repair_request_id']=='repair-1'
    assert rows==[{'id':'t01','task':'CloseMicrowave'}]
    with pytest.raises(ValueError):repair_row(request,rows,[{**source,'status':'completed_unscored'}])
    with pytest.raises(ValueError):repair_row({**request,'provider':'astra'},rows,[source])


def test_capacity_lanes_do_not_add_qwen_request_concurrency():
    from collections import Counter
    from pairing.cohort50_queue import LANES, lane_provider
    assert Counter(lane_provider(lane) for lane in LANES)=={'astra':3,'qwen':1}


def test_only_provider_quota_and_auth_failures_pause_launches():
    from pairing.cohort50_queue import provider_pause_reason
    assert provider_pause_reason('You have hit your usage limit')
    assert provider_pause_reason('Qwen endpoint HTTP 429; credentials suppressed')
    assert provider_pause_reason('Active Astra account differs from authorized account')
    assert provider_pause_reason('Qwen endpoint HTTP 401; credentials suppressed')
    assert provider_pause_reason('Qwen final content is not valid JSON') is None
    assert provider_pause_reason('RuntimeError: initial scene differs') is None
    assert provider_pause_reason('Qwen endpoint HTTP 413') is None


def test_native_hand_repair_requires_the_known_old_gr1_contract():
    import pytest
    from pairing.cohort50_queue import repair_row
    request={'request_id':'hand-repair','source_series_id':'old','provider':'astra','task_id':'t10','reason':'Wrapper rejected native hand targets','repair_kind':'native_hand_contract'}
    rows=[{'id':'t10','config':{'robot':'GR1'}}]
    source={'id':'old','provider':'astra','cohort_id':'t10','status':'completed_unscored'}
    assert repair_row(request,rows,[source])['repair_kind']=='native_hand_contract'
    with pytest.raises(ValueError):repair_row(request,rows,[{**source,'hand_action_contract':'native-raw-position-v2'}])
    with pytest.raises(ValueError):repair_row(request,[{'id':'t10','config':{'robot':'PandaOmron'}}],[source])


def test_keychain_failure_pauses_provider_launches():
    from pairing.cohort50_queue import provider_pause_reason
    assert provider_pause_reason('Existing Modal FDR Keychain credential unavailable; output suppressed')
