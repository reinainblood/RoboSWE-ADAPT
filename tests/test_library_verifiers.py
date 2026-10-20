import math

import pytest

from roboswe.library.metrics import score_library
from roboswe.library.spec import GOALS, library_set
from roboswe.library.verifiers import (
    FixtureBinding,
    JointBinding,
    TaskVerifier,
    evaluate,
)


def binding(role, low=0, high=1):
    return FixtureBinding(role, role, (JointBinding("joint", low, high),))


@pytest.mark.parametrize("task", list(GOALS))
def test_individual_goal_positive_negative(task):
    goal = GOALS[task]
    b = binding(goal.fixture, high=math.pi / 2 if goal.fixture == "sink" else 1)
    on = task.startswith(("Open", "TurnOn"))
    assert evaluate(task, b, {"joint": 1.0 if on else 0.0})["success"]
    assert not evaluate(task, b, {"joint": 0.0 if on else 1.0})["success"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -3, 100])
def test_invalid_measurements_cannot_pass(value):
    with pytest.raises(ValueError):
        evaluate("OpenOven", binding("oven"), {"joint": value})


def test_negative_hinge_and_empty_binding():
    b = binding("microwave", -math.pi / 2, 0)
    assert evaluate("OpenMicrowave", b, {"joint": -math.pi / 2})["success"]
    assert evaluate("CloseMicrowave", b, {"joint": 0})["success"]
    with pytest.raises(ValueError):
        evaluate("OpenOven", FixtureBinding("oven", "bad", ()), {})


def test_drawer_positive_overshoot_uses_upstream_absolute_displacement():
    drawer = FixtureBinding(
        "drawer", "shallow", (JointBinding("joint", -0.1, 0, -0.2, 0),)
    )
    result = evaluate("CloseDrawer", drawer, {"joint": 0.009})
    assert result["measurements"][0]["normalized"] == pytest.approx(0.09)
    assert not result["success"]


def test_sustained_real_transition_not_free_pass_or_duplicate_samples():
    v = TaskVerifier("OpenDrawer", binding("drawer"), {"joint": 0}, settle_steps=3)
    assert not v.update({"joint": 1}, 1)["success"]
    assert not v.update({"joint": 0.2}, 2)["success"]
    assert not v.update({"joint": 1}, 3)["success"]
    with pytest.raises(ValueError):
        v.update({"joint": 1}, 3)
    assert not v.update({"joint": 1}, 4)["success"]
    assert v.update({"joint": 1}, 5)["success"]
    assert not v.update({"joint": 0}, 6)["success"]
    free = TaskVerifier("CloseOven", binding("oven"), {"joint": 0}, settle_steps=1)
    assert free.update({"joint": 0}, 1)["status"] == "precondition_invalid"


def trials(spec, early=100, late=50):
    return [
        {
            "round": t["round"],
            "pair_key": t["pair_key"],
            "elapsed_seconds": early if t["round"] == 1 else late,
            "code_errors": 4 if t["round"] == 1 else 1,
            "execution_attempts": 10,
            "execution_errors": 4 if t["round"] == 1 else 1,
            "control_steps": 100,
            "verified_success": True,
            "precondition_valid": True,
        }
        for t in spec["tasks"]
    ]


def test_invalid_sensor_sample_revokes_prior_success_and_consumes_its_step():
    verifier = TaskVerifier("OpenOven", binding("oven"), {"joint": 0}, settle_steps=2)
    verifier.update({"joint": 1}, 1)
    assert verifier.update({"joint": 1}, 2)["success"]
    with pytest.raises(ValueError):
        verifier.update({"joint": float("nan")}, 3)
    assert not verifier.verdict()["success"]
    assert verifier.verdict()["status"] == "invalid_measurement"
    assert not verifier.update({"joint": 1}, 4)["success"]
    assert verifier.update({"joint": 1}, 5)["success"]


def test_learning_pairs_and_missing_denominator():
    spec = library_set()
    rows = trials(spec)
    score = score_library(spec, rows)
    assert score["expected_trials"] == 16 and score["matched_pairs"] == 8
    assert score["task_success_rate"] == 1
    assert score["library_score"] is None
    assert score["paired_successful_time_gain"] == pytest.approx(0.5)
    assert score["pairs"][0]["error_rate_change"] == pytest.approx(-0.3)
    missing = score_library(spec, rows[:-1])
    assert (
        missing["missing_trials"] == 1
        and missing["paired_successful_time_gain"]
        < score["paired_successful_time_gain"]
    )


def test_late_failure_cannot_game_time_by_exiting_early():
    spec = library_set()
    rows = trials(spec)
    for row in rows:
        if row["round"] == 2:
            row.update(verified_success=False, elapsed_seconds=0, code_errors=0)
    score = score_library(spec, rows)
    assert score["library_score"] is None
    assert all(p["time_gain"] is None for p in score["pairs"])


def test_reproduced_infrastructure_interruption_is_not_a_participant_failure():
    spec = library_set()
    rows = trials(spec)
    cold = controls(spec, rows)
    # Even legacy records whose generic provenance set this flag true must not
    # hide the recorded termination cause.
    rows[8].update(
        verified_success=False, termination_reason="infrastructure_interruption"
    )
    for controls_arg in (None, cold):
        result = score_library(spec, rows, cold_controls=controls_arg)
        assert result["library_score"] is None
        assert result["task_success_rate"] is None
        assert result["success_rate_change"] is None
        assert result["infrastructure_invalid_trials"] == 1
        assert result["pairs"][0]["reason"] == "infrastructure_not_verified"
        assert result["pairs"][0]["comparable"] is False
        assert result["score_status"] == "unverified_warm_sequence"


def test_early_failure_is_not_a_fake_slow_baseline():
    spec = library_set()
    rows = trials(spec)
    for row in rows:
        if row["round"] == 1:
            row.update(verified_success=False, elapsed_seconds=0)
    score = score_library(spec, rows)
    assert score["paired_successful_time_gain"] == 0
    assert score["success_rate_change"] == 1
    assert all(p["reason"] == "late_recovered" for p in score["pairs"])


def controls(spec, rows):
    cold = []
    for row in rows:
        row.update(
            start_fingerprint="same-" + row["pair_key"],
            config_sha256="same-config",
            protocol_sha256="same-protocol",
            model="same-model",
            reasoning_effort="medium",
            task_timeout_seconds=spec["task_timeout_seconds"],
            retained_context=True,
            retained_library=True,
            infrastructure_valid=True,
            reproduced=True,
        )
        if row["round"] == 2:
            cold.append(
                {
                    **row,
                    "elapsed_seconds": 200,
                    "retained_context": False,
                    "retained_library": False,
                }
            )
    return cold


def test_controlled_score_ignores_early_sandbagging_and_error_denominator_games():
    spec = library_set()
    rows = trials(spec)
    cold = controls(spec, rows)
    before = score_library(spec, rows, cold_controls=cold)
    for row in rows:
        if row["round"] == 1:
            row["elapsed_seconds"] = 700
        row["execution_attempts"] = 100000
    after = score_library(spec, rows, cold_controls=cold)
    assert (
        before["library_score"]
        == after["library_score"]
        == pytest.approx(100 * 150 / 720)
    )


def test_cold_controls_require_matched_states_and_complete_coverage():
    spec = library_set()
    rows = trials(spec)
    cold = controls(spec, rows)
    assert score_library(spec, rows, cold_controls=cold[:-1])["library_score"] is None
    cold[0]["start_fingerprint"] = "different-physical-state"
    with pytest.raises(ValueError, match="start_fingerprint"):
        score_library(spec, rows, cold_controls=cold)


def test_cold_context_policy_must_match_the_warm_participant():
    spec = library_set()
    rows = trials(spec)
    cold = controls(spec, rows)
    rows[8]["context_policy"] = {"auto_compact_tokens": 120000}
    cold[0]["context_policy"] = {"auto_compact_tokens": 800000}
    with pytest.raises(ValueError, match="context_policy"):
        score_library(spec, rows, cold_controls=cold)


def test_unreproduced_control_is_not_disguised_as_zero_credit_precondition():
    spec = library_set()
    rows = trials(spec)
    cold = controls(spec, rows)
    cold[0].update(
        precondition_valid=False,
        infrastructure_valid=False,
        status="warm_trial_not_comparable",
    )
    result = score_library(spec, rows, cold_controls=cold)
    assert result["library_score"] is None
    assert (
        result["controlled_comparison"]["pairs"][0]["reason"]
        == "infrastructure_not_verified"
    )
    cold = controls(spec, rows)
    rows[8]["reproduced"] = False
    assert score_library(spec, rows, cold_controls=cold)["library_score"] is None


def test_partial_warm_sequence_cannot_receive_a_library_score():
    spec = library_set()
    rows = trials(spec)
    cold = controls(spec, rows)
    result = score_library(spec, rows[1:], cold_controls=cold)
    assert (
        result["library_score"] is None
        and result["score_status"] == "incomplete_warm_sequence"
    )


@pytest.mark.parametrize("field", ["reproduced", "infrastructure_valid"])
def test_unverified_early_round_withholds_overall_library_score(field):
    spec = library_set()
    rows = trials(spec)
    cold = controls(spec, rows)
    rows[0].update(verified_success=False)
    rows[0][field] = False
    result = score_library(spec, rows, cold_controls=cold)
    assert result["library_score"] is None
    assert result["score_status"] == "unverified_warm_sequence"


def test_invalid_trial_logs_fail_closed():
    spec = library_set()
    rows = trials(spec)
    with pytest.raises(ValueError):
        score_library(spec, rows + [rows[0]])
    rows[0]["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError):
        score_library(spec, rows)
    rows = trials(spec)
    rows[0]["precondition_valid"] = False
    with pytest.raises(ValueError):
        score_library(spec, rows)
