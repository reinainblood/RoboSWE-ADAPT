from pairing.learning_timing import measures


def test_passive_time_and_artifact_overhead_are_not_hidden():
    result=measures({'elapsed_seconds':100,'passive_wait_wall_seconds':40},{'created_at':'2026-10-06T00:00:00+00:00','finished_at':'2026-10-06T00:03:00+00:00'})
    assert result['active_work_seconds']==100
    assert result['episode_wall_seconds']==140
    assert result['job_wall_seconds']==180
    assert result['startup_and_finalization_seconds']==40


def test_missing_finish_is_not_zero_overhead_or_completion():
    result=measures({'elapsed_seconds':100},{'created_at':'2026-10-06T00:00:00+00:00'})
    assert result['job_wall_seconds'] is None
    assert result['startup_and_finalization_seconds'] is None
