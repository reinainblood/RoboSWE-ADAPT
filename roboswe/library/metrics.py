"""Learning diagnostics and a separately controlled library-performance score.

Agent-controlled early-round slowness/errors never set the graded baseline.
Within-run changes are descriptive; scoring requires cold comparisons beginning
at the same trusted physical state as each retained-context late task.
"""

import math


def _infrastructure_invalid(trial):
    return (
        trial.get("infrastructure_valid") is False
        or trial.get("reproduced") is False
        or trial.get("termination_reason") == "infrastructure_interruption"
    )


def _nonnegative(value, field):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid {field}")


def _error_rate(trial):
    attempts, errors = trial.get("execution_attempts"), trial.get("execution_errors")
    if attempts is None or errors is None:
        return None
    if (
        type(attempts) is not int
        or type(errors) is not int
        or not 0 <= errors <= attempts
    ):
        raise ValueError("Invalid execution error counts")
    return errors / attempts if attempts else None


def score_library(spec, trials, *, cold_controls=None):
    expected = {(t["round"], t["pair_key"]): t for t in spec["tasks"]}
    if len(expected) != len(spec["tasks"]) or {r for r, _ in expected} != {1, 2}:
        raise ValueError(
            "Learning metrics require two matched rounds without duplicate tasks"
        )
    tasks = list(dict.fromkeys(t["pair_key"] for t in spec["tasks"]))
    if any((r, task) not in expected for task in tasks for r in (1, 2)):
        raise ValueError("Every task must appear in both rounds")
    rows = {}
    for trial in trials:
        key = (trial["round"], trial["pair_key"])
        if key not in expected or key in rows:
            raise ValueError("Unexpected or duplicate trial")
        for field in ("elapsed_seconds", "code_errors", "control_steps"):
            _nonnegative(trial[field], field)
        if type(trial["verified_success"]) is not bool:
            raise ValueError("verified_success must be a verifier-owned boolean")
        if trial.get("precondition_valid") is not True and trial["verified_success"]:
            raise ValueError("Precondition-invalid trials cannot pass")
        _error_rate(trial)
        rows[key] = trial
    pairs = []
    for task in tasks:
        early, late = rows.get((1, task)), rows.get((2, task))
        pair = {
            "task": task,
            "comparable": False,
            "time_gain": None,
            "error_rate_change": None,
            "control_step_gain": None,
        }
        if early is None or late is None:
            pair["reason"] = "missing_trial"
        elif _infrastructure_invalid(early) or _infrastructure_invalid(late):
            pair["reason"] = "infrastructure_not_verified"
        elif not early.get("precondition_valid") or not late.get("precondition_valid"):
            pair["reason"] = "precondition_invalid"
        else:
            pair["comparable"] = True
            both_passed = early["verified_success"] and late["verified_success"]
            pair["reason"] = (
                "both_passed"
                if both_passed
                else "late_recovered"
                if late["verified_success"]
                else "late_failed"
            )
            # Failed early trials have censored completion times. Do not invent a
            # slow successful baseline or award speed credit for deliberate failure.
            if both_passed:
                pair["time_gain"] = max(
                    -1,
                    min(
                        1,
                        (early["elapsed_seconds"] - late["elapsed_seconds"])
                        / max(early["elapsed_seconds"], 1),
                    ),
                )
                pair["control_step_gain"] = (
                    early["control_steps"] - late["control_steps"]
                ) / max(1, early["control_steps"], late["control_steps"])
            er, lr = _error_rate(early), _error_rate(late)
            pair.update(early_execution_error_rate=er, late_execution_error_rate=lr)
            if er is not None and lr is not None:
                # Percentage-point change: negative means fewer failed executions.
                pair["error_rate_change"] = lr - er
        pairs.append(pair)
    invalid_count = sum(_infrastructure_invalid(t) for t in rows.values())
    complete_valid = len(rows) == len(expected) and invalid_count == 0
    passed = sum(
        t["verified_success"] and not _infrastructure_invalid(t) for t in rows.values()
    )
    completion = passed / len(expected)
    early_rate = sum(
        t["verified_success"] for (r, _), t in rows.items() if r == 1
    ) / len(tasks)
    late_rate = sum(
        t["verified_success"] for (r, _), t in rows.items() if r == 2
    ) / len(tasks)
    result = {
        "schema_version": 2,
        "metric_revision": "controlled-library-v2.2",
        "score_status": "incomplete_warm_sequence"
        if len(rows) != len(expected)
        else "unverified_warm_sequence"
        if invalid_count
        else "requires_cold_controls",
        "library_score": None,
        "expected_trials": len(expected),
        "observed_trials": len(rows),
        "missing_trials": len(expected) - len(rows),
        "infrastructure_invalid_trials": invalid_count,
        "verified_successes": passed,
        "rate_status": "complete" if complete_valid else "incomplete_or_unverified",
        "task_success_rate": completion if complete_valid else None,
        "early_success_rate": early_rate if complete_valid else None,
        "late_success_rate": late_rate if complete_valid else None,
        "success_rate_change": late_rate - early_rate if complete_valid else None,
        "matched_pairs": sum(p["comparable"] for p in pairs),
        "expected_pairs": len(tasks),
        "paired_successful_time_gain": sum(p["time_gain"] or 0 for p in pairs)
        / len(tasks),
        "interpretation": "Within-run changes are descriptive, not a grade or causal learning estimate. Failed completion times are censored. Error rates have measured attempt denominators and are not rewarded directly.",
        "pairs": pairs,
    }
    if cold_controls is not None:
        result["controlled_comparison"] = score_controls(spec, rows, cold_controls)
        result["score_status"] = result["controlled_comparison"]["status"]
        result["library_score"] = result["controlled_comparison"]["library_score"]
    return result


def score_controls(spec, warm_rows, controls):
    """Caller must supply verifier-owned warm and cold trial records.

    No score from missing, mismatched, or infrastructure-invalid controls. Code
    failures/task deadlines remain legitimate failures; infrastructure failures do not.
    """
    tasks = list(dict.fromkeys(t["pair_key"] for t in spec["tasks"]))
    cold = {}
    for trial in controls:
        task = trial["pair_key"]
        if task not in tasks or task in cold:
            raise ValueError("Unknown or duplicate cold control")
        cold[task] = trial
    pairs = []
    budget = spec["task_timeout_seconds"]
    for task in tasks:
        warm, baseline = warm_rows.get((2, task)), cold.get(task)
        pair = {"task": task, "valid": False, "time_gain": None}
        if warm is None:
            pair["reason"] = "missing_control_or_warm_trial"
        elif warm.get("reproduced") is not True or _infrastructure_invalid(warm):
            pair["reason"] = "infrastructure_not_verified"
        elif warm.get("precondition_valid") is False:
            pair["reason"] = "precondition_invalid"
        elif baseline is None:
            pair["reason"] = "missing_control_or_warm_trial"
        elif (
            warm.get("precondition_valid") is not True
            or baseline.get("precondition_valid") is not True
            or baseline.get("reproduced") is not True
            or warm.get("infrastructure_valid") is not True
            or baseline.get("infrastructure_valid") is not True
            or _infrastructure_invalid(baseline)
        ):
            pair["reason"] = "infrastructure_not_verified"
        else:
            # These fields are copied by the trusted coordinator, never the agent.
            if warm.get("context_policy") != baseline.get("context_policy"):
                raise ValueError(f"Cold comparison mismatch: {task} context_policy")
            for key in (
                "start_fingerprint",
                "config_sha256",
                "protocol_sha256",
                "model",
                "reasoning_effort",
                "task_timeout_seconds",
            ):
                if warm.get(key) is None or baseline.get(key) != warm[key]:
                    raise ValueError(f"Cold comparison mismatch: {task} {key}")
            if warm["task_timeout_seconds"] != budget:
                raise ValueError("Control budget differs from library protocol")
            if (
                baseline.get("retained_context") is not False
                or baseline.get("retained_library") is not False
            ):
                raise ValueError(
                    "Cold control must start without retained context or library"
                )
            if (
                warm.get("retained_context") is not True
                or warm.get("retained_library") is not True
            ):
                raise ValueError("Warm comparison must retain both context and library")
            for trial in (warm, baseline):
                _nonnegative(trial["elapsed_seconds"], "elapsed_seconds")
                if type(trial["verified_success"]) is not bool:
                    raise ValueError("Control verdict must be a verified boolean")
            warm_cost = (
                min(budget, warm["elapsed_seconds"])
                if warm["verified_success"]
                else budget
            )
            cold_cost = (
                min(budget, baseline["elapsed_seconds"])
                if baseline["verified_success"]
                else budget
            )
            pair.update(
                valid=True,
                reason="matched_state_control",
                time_gain=(cold_cost - warm_cost) / budget,
                warm_success=warm["verified_success"],
                cold_success=baseline["verified_success"],
                warm_execution_error_rate=_error_rate(warm),
                cold_execution_error_rate=_error_rate(baseline),
            )
        pairs.append(pair)
    complete_warm = len(warm_rows) == len(spec["tasks"])
    verified_warm = all(
        row.get("reproduced") is True
        and row.get("infrastructure_valid") is True
        and not _infrastructure_invalid(row)
        for row in warm_rows.values()
    )
    valid = (
        complete_warm
        and verified_warm
        and all(p["valid"] or p["reason"] == "precondition_invalid" for p in pairs)
    )
    completion = sum(t["verified_success"] for t in warm_rows.values()) / len(
        spec["tasks"]
    )
    gain = sum(p["time_gain"] or 0 for p in pairs) / len(pairs) if valid else None
    return {
        "status": "incomplete_warm_sequence"
        if not complete_warm
        else "unverified_warm_sequence"
        if not verified_warm
        else "controlled_pilot_estimate"
        if valid
        else "incomplete_cold_controls",
        "library_score": 100 * completion * max(0, gain) if valid else None,
        "mean_signed_time_gain": gain,
        "valid_comparisons": sum(p["valid"] for p in pairs),
        "precondition_invalid_zero_credit": sum(
            p["reason"] == "precondition_invalid" for p in pairs
        ),
        "pairs": pairs,
        "formula": "100 * warm_task_success_rate * max(0, mean((cold_censored_time - warm_censored_time) / task_budget))",
        "interpretation": "Matched-state comparison of retained context plus files against a cold participant. One sample per condition; not a confidence interval or isolation of library files from model context. Error rates are reported separately to avoid rewarding denominator manipulation.",
    }
