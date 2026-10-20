import json

from pairing.opus55_cohort import completed_cohort_ids


def test_completed_cohort_ids_excludes_provider_interrupted_partial_series(tmp_path):
    complete = tmp_path / "complete"
    partial = tmp_path / "partial"
    malformed = tmp_path / "malformed"
    for directory in (complete, partial, malformed):
        directory.mkdir()
    (complete / "report.json").write_text(
        json.dumps({"trials": [{"attempt_number": 1}, {"attempt_number": 2}]})
    )
    (partial / "report.json").write_text(json.dumps({"trials": []}))
    (malformed / "report.json").write_text("not-json")
    series = [
        {"id": "complete", "cohort_id": "t18", "status": "completed_unscored"},
        {"id": "partial", "cohort_id": "t19", "status": "infrastructure_error"},
        {"id": "missing", "cohort_id": "t20", "status": "infrastructure_error"},
        {"id": "malformed", "cohort_id": "t21", "status": "infrastructure_error"},
    ]
    assert completed_cohort_ids(series, jobs_root=tmp_path) == {"t18"}
