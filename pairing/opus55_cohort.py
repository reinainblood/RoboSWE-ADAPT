"""Matched Opus 5.5 run over the frozen Astra retained-learning cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from zipfile import ZipFile

from pairing.anthropic_model import (
    MODEL,
    AnthropicLearningModel,
    SpendLedger,
)
from pairing.cohort50 import (
    ROOT as COHORT_ROOT,
    SCHEMA,
    effective_row,
    emit,
    freeze,
    preflight_directory,
    prompt_for,
)
from pairing.cohort50_queue import PRIORITIES
from roboswe.campaign import reference_lease
from roboswe.jobs import create, now, save
from roboswe.storage import require_disk_headroom


ROOT = Path("artifacts/opus55-cohort-20261009")
POINTER = ROOT / "campaign.json"
ACTIVE = ROOT / "active.json"
PROGRESS = ROOT / "progress.json"
BUDGET = ROOT / "budget.json"
PREFLIGHT = ROOT / "cache-preflight"
REFERENCE = COHORT_ROOT / "astra-50-results.json"
PROVIDER = "opus55"
LANES = ("opus55-1",)
SPEND_LIMIT_USD = 195.0
REQUEST_RESERVATION_USD = 5.0 + 32_768 * 20.0 / 1_000_000


def ordered_rows(manifest):
    return sorted(
        [effective_row(row) for row in manifest["tasks"]],
        key=lambda row: PRIORITIES.index(row["id"])
        if row["id"] in PRIORITIES
        else len(PRIORITIES) + int(row["id"][1:]),
    )


def source_jobs(campaign_id: str) -> list[dict]:
    values = []
    for path in Path("jobs").glob("*/job.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            record.get("campaign_id") == campaign_id
            and record.get("kind") == "cohort50-series"
            and record.get("provider") == PROVIDER
        ):
            values.append(record)
    return values


def completed_cohort_ids(series: list[dict], *, jobs_root: Path = Path("jobs")) -> set[str]:
    """Return only cohorts with a complete, auditable two-attempt result.

    Provider-paused series may contain paid partial inference and a report shell,
    but they must be retried from a fresh model context and exact initial scene.
    Treating every source job as used would silently skip the interrupted task on
    campaign resume.
    """
    completed = set()
    for job in series:
        report_path = jobs_root / job["id"] / "report.json"
        if not report_path.exists():
            continue
        try:
            trials = json.loads(report_path.read_text()).get("trials") or []
        except (OSError, json.JSONDecodeError):
            continue
        if len(trials) == 2:
            completed.add(job["cohort_id"])
    return completed


def reconcile_interrupted_series(campaign_id: str) -> list[dict]:
    """Close dead pilot jobs and preserve any paid partial inference separately."""
    commands = subprocess.run(
        ["ps", "-axo", "command="], capture_output=True, text=True
    ).stdout
    if f"--provider {PROVIDER} --parent {campaign_id}" in commands:
        raise RuntimeError("An Opus series from the failed campaign is still alive")
    excluded = []
    for series in source_jobs(campaign_id):
        directory = Path("jobs") / series["id"]
        usage_path = directory / "model-usage.jsonl"
        usage = [json.loads(line) for line in usage_path.read_text().splitlines()] if usage_path.exists() else []
        cost = sum(float(row.get("cost_usd") or 0) for row in usage)
        if series.get("status") == "running":
            series.update(
                status="infrastructure_error",
                finished_at=now(),
                reconciliation={
                    "reason": "owned scheduler interruption during storage-headroom recovery",
                    "reconciled_at": now(),
                    "model_requests": len(usage),
                    "recorded_cost_usd": cost,
                    "excluded_from_head_to_head": True,
                },
            )
            save(directory, series)
        for path in Path("jobs").glob("*/job.json"):
            attempt = json.loads(path.read_text())
            if attempt.get("campaign_id") != series["id"] or attempt.get("status") != "running":
                continue
            attempt.update(
                status="infrastructure_error",
                finished_at=now(),
                reconciliation={
                    "reason": "owned scheduler interruption during storage-headroom recovery",
                    "reconciled_at": now(),
                    "model_requests": len(usage),
                    "recorded_cost_usd": cost,
                    "excluded_from_head_to_head": True,
                },
            )
            if (path.parent / "partial-library-interrupted.zip").exists():
                attempt["artifacts"]["partial_library_interrupted"] = "partial-library-interrupted.zip"
            save(path.parent, attempt)
        excluded.append({
            "series_id": series["id"],
            "cohort_id": series["cohort_id"],
            "model_requests": len(usage),
            "recorded_cost_usd": cost,
            "reason": "storage-headroom pilot interrupted; no attempt report; excluded",
        })
    return excluded


def live_cache_validated(campaign_id: str) -> bool:
    """Require two real decision receipts and a cache hit before scaling lanes."""
    for series in source_jobs(campaign_id):
        path = Path("jobs") / series["id"] / "model-usage.jsonl"
        if not path.exists():
            continue
        try:
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if json.loads(line).get("purpose") == "decision"
            ]
        except (OSError, json.JSONDecodeError):
            continue
        if len(rows) >= 2 and any(
            row.get("usage", {}).get("cache_read_input_tokens", 0) > 0
            for row in rows[1:]
        ):
            return True
    return False


def alive(item: dict) -> bool:
    command = subprocess.run(
        ["ps", "-p", str(item["pid"]), "-o", "command="],
        capture_output=True,
        text=True,
    ).stdout
    return item["identity"] in command


def cache_preflight() -> dict:
    """Authenticate, pin the model, and prove a billed cache hit before launch."""
    ROOT.mkdir(parents=True, exist_ok=True)
    receipt_path = PREFLIGHT / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("status") == "passed"
            and receipt.get("model") == MODEL
            and receipt.get("second_usage", {}).get("cache_read_input_tokens", 0) > 0
        ):
            return receipt
        raise RuntimeError("Saved Opus cache preflight is not valid")
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    ledger = SpendLedger(BUDGET, limit_usd=SPEND_LIMIT_USD)
    del ledger
    stable = prompt_for("CloseMicrowave")
    stable += "\nNon-executing cache validation context follows. " * 1_200
    with AnthropicLearningModel(
        stable,
        PREFLIGHT,
        context_policy="legacy",
        effort="high",
        cache_ttl="5m",
        max_tokens=256,
        ledger_path=BUDGET,
        spend_limit_usd=SPEND_LIMIT_USD,
    ) as model:
        decision_one = model.ask(
            json.dumps(
                {
                    "preflight": True,
                    "instruction": "Return operation finish with empty path and content.",
                }
            ),
            schema=SCHEMA,
            timeout=180,
        )
        decision_two = model.ask(
            json.dumps(
                {
                    "preflight": True,
                    "instruction": "Again return operation finish with empty path and content.",
                }
            ),
            schema=SCHEMA,
            timeout=180,
        )
        metadata = dict(model.metadata)
    usage_rows = [
        json.loads(line)
        for line in (PREFLIGHT / "model-usage.jsonl").read_text().splitlines()
    ]
    if len(usage_rows) != 2:
        raise RuntimeError("Opus cache preflight produced an unexpected request count")
    first, second = usage_rows
    if first["usage"]["cache_creation_input_tokens"] <= 0:
        raise RuntimeError("Opus cache preflight did not create a prompt cache entry")
    if second["usage"]["cache_read_input_tokens"] <= 0:
        raise RuntimeError("Opus cache preflight did not read from prompt cache")
    if decision_one["operation"] != "finish" or decision_two["operation"] != "finish":
        raise RuntimeError("Opus structured-output preflight returned an unexpected operation")
    receipt = {
        "status": "passed",
        "completed_at": now(),
        "model": metadata["model"],
        "provider": metadata["model_provider"],
        "effort": metadata["reasoning_effort"],
        "cache": metadata["prompt_caching"],
        "structured_output": metadata["structured_output"],
        "credential_source": metadata["credential_source"],
        "workspace": metadata["workspace"],
        "first_response_id": first.get("id"),
        "second_response_id": second.get("id"),
        "first_usage": first["usage"],
        "second_usage": second["usage"],
        "preflight_cost_usd": first["cost_usd"] + second["cost_usd"],
        "secret_recorded": False,
    }
    emit(receipt_path, receipt)
    return receipt


def comparison(campaign_id: str) -> dict:
    reference = json.loads(REFERENCE.read_text())
    astra = {pair["cohort_id"]: pair for pair in reference["pairs"]}
    rows = []
    for series in source_jobs(campaign_id):
        report_path = Path("jobs") / series["id"] / "report.json"
        if not report_path.exists():
            continue
        trials = json.loads(report_path.read_text()).get("trials") or []
        if len(trials) != 2:
            continue
        pair = astra[series["cohort_id"]]
        transition = pair["outcome_transition"]
        astra_first = transition.startswith("success")
        astra_second = transition.endswith("success")
        rows.append(
            {
                "cohort_id": series["cohort_id"],
                "task": series["task"],
                "opus_series": series["id"],
                "astra_series": pair["series_id"],
                "astra": [astra_first, astra_second],
                "opus55": [bool(trials[0]["observed_success"]), bool(trials[1]["observed_success"])],
                "opus_transition": (
                    ("success" if trials[0]["observed_success"] else "failure")
                    + " → "
                    + ("success" if trials[1]["observed_success"] else "failure")
                ),
            }
        )
    return {
        "generated_at": now(),
        "campaign_id": campaign_id,
        "scope": (
            "Matched completed two-attempt pairs only. Same frozen tasks/scenes and solver budgets; "
            "descriptive model comparison, not a causal ICL estimate or certified benchmark score."
        ),
        "matched_tasks": len(rows),
        "astra_first_successes": sum(row["astra"][0] for row in rows),
        "astra_second_successes": sum(row["astra"][1] for row in rows),
        "opus55_first_successes": sum(row["opus55"][0] for row in rows),
        "opus55_second_successes": sum(row["opus55"][1] for row in rows),
        "opus55_tasks_with_any_success": sum(any(row["opus55"]) for row in rows),
        "rows": sorted(rows, key=lambda row: int(row["cohort_id"][1:])),
    }


def update_comparison(directory: Path, record: dict):
    result = comparison(record["id"])
    emit(directory / "head-to-head.json", result)
    record["artifacts"]["head_to_head"] = "head-to-head.json"


def create_campaign(manifest: dict, preflight: dict) -> tuple[Path, dict, bool]:
    replaces = None
    excluded_pilots = []
    if POINTER.exists():
        pointer = json.loads(POINTER.read_text())
        directory = Path(pointer["directory"])
        existing = json.loads((directory / "job.json").read_text())
        if existing.get("status") != "scheduler_error":
            return directory, existing, False
        excluded_pilots = reconcile_interrupted_series(existing["id"])
        series = source_jobs(existing["id"])
        completed = []
        for job in series:
            report_path = Path("jobs") / job["id"] / "report.json"
            if report_path.exists() and len(json.loads(report_path.read_text()).get("trials") or []) == 2:
                completed.append(job)
        if completed or any(job.get("status") != "infrastructure_error" for job in series):
            raise RuntimeError("Refusing to replace a failed campaign containing a completed result")
        replaces = existing["id"]
        history_path = ROOT / "campaign-history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        history.append({**pointer, "status": existing.get("status"), "preserved_at": now()})
        emit(history_path, history)
    directory, record = create(
        "jobs",
        "Opus 5.5 high · frozen Astra 50-task retained-learning head-to-head",
        "opus55-cohort-campaign",
    )
    reference_hash = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    record.update(
        status="starting",
        provider=PROVIDER,
        model=MODEL,
        benchmark_score=None,
        spend_limit_usd=SPEND_LIMIT_USD,
        cohort_sha256=manifest["sha256"],
        astra_reference_sha256=reference_hash,
        replaces_campaign=replaces,
        excluded_pilots=excluded_pilots if replaces else [],
    )
    (directory / "cohort.json").write_bytes((COHORT_ROOT / "cohort.json").read_bytes())
    (directory / "astra-reference.json").write_bytes(REFERENCE.read_bytes())
    (directory / "cache-preflight.json").write_text(json.dumps(preflight, indent=2))
    with ZipFile(directory / "source.zip", "w") as archive:
        for folder in ("pairing", "roboswe", "robot_session"):
            for path in Path(folder).rglob("*.py"):
                archive.write(path)
    record["artifacts"].update(
        cohort="cohort.json",
        astra_reference="astra-reference.json",
        cache_preflight="cache-preflight.json",
        source="source.zip",
        progress="progress.json",
        budget="budget-snapshot.json",
    )
    save(directory, record)
    emit(POINTER, {"id": record["id"], "directory": str(directory)})
    return directory, record, True


def run():
    preflight = cache_preflight()
    manifest = freeze()
    rows = ordered_rows(manifest)
    directory, record, created = create_campaign(manifest, preflight)
    if record.get("status") in {"completed_unscored", "budget_limit_reached"}:
        return
    image = subprocess.check_output(
        ["docker", "image", "inspect", "roboswe-code:local", "--format", "{{.Id}}"],
        text=True,
    ).strip()
    existing = source_jobs(record["id"])
    used = completed_cohort_ids(existing)
    pending = [row for row in rows if row["id"] not in used]
    completed = []
    lanes = {}
    processes = {}
    streams = {}
    if PROGRESS.exists() and not created:
        saved = json.loads(PROGRESS.read_text())
        completed = saved.get("completed", [])
        for item in saved.get("active", []):
            if alive(item):
                lanes[item["lane"]] = item
            else:
                completed.append({**item, "status": "process_ended_before_adoption"})
    ledger = SpendLedger(BUDGET, limit_usd=SPEND_LIMIT_USD)

    def snapshot():
        budget = ledger.snapshot()
        emit(
            PROGRESS,
            {
                "campaign_id": record["id"],
                "completed": completed,
                "active": list(lanes.values()),
                "pending": [row["id"] for row in pending],
                "budget": budget,
                "lanes": list(LANES),
                "updated_at": now(),
            },
        )
        emit(directory / "progress.json", json.loads(PROGRESS.read_text()))
        emit(directory / "budget-snapshot.json", budget)
        update_comparison(directory, record)
        save(directory, record)
        return budget

    def launch(lane: str, row: dict):
        require_disk_headroom("jobs", 7)
        expected = preflight_directory(row) / "receipt.json"
        if not expected.exists():
            raise RuntimeError(f"Frozen preflight missing for {row['id']}")
        log = ROOT / "logs" / f"{row['id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("ab")
        args = [
            sys.executable,
            "-u",
            "-m",
            "pairing.cohort50",
            "series",
            "--task-id",
            row["id"],
            "--provider",
            PROVIDER,
            "--parent",
            record["id"],
            "--code-image",
            image,
        ]
        env = os.environ.copy()
        env["ROBOSWE_ANTHROPIC_BUDGET_PATH"] = str(BUDGET.resolve())
        env["ROBOSWE_ANTHROPIC_MAX_USD"] = str(SPEND_LIMIT_USD)
        env["ROBOSWE_ANTHROPIC_WORKSPACE_ID"] = preflight["workspace"]["id"]
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
        )
        identity = (
            f"pairing.cohort50 series --task-id {row['id']} --provider {PROVIDER} "
            f"--parent {record['id']}"
        )
        item = {
            "id": row["id"],
            "task": row["task"],
            "lane": lane,
            "pid": proc.pid,
            "identity": identity,
            "started_at": now(),
            "started_epoch": time.time(),
            "log": str(log),
        }
        processes[proc.pid] = proc
        streams[proc.pid] = stream
        lanes[lane] = item
        print(json.dumps({"launched": item}), flush=True)

    if record.get("status") == "provider_paused":
        history = record.setdefault("provider_pause_history", [])
        history.append(
            {
                "paused_at": record.get("finished_at"),
                "reason": record.get("provider_pause_reason"),
            }
        )
        record.pop("finished_at", None)
        record.pop("provider_pause_reason", None)
    record.update(status="running", execution_started_at=record.get("execution_started_at") or now())
    save(directory, record)
    emit(
        ACTIVE,
        {
            "campaign_id": record["id"],
            "directory": str(directory),
            "pid": os.getpid(),
            "model": MODEL,
            "spend_limit_usd": SPEND_LIMIT_USD,
            "lanes": list(LANES),
            "started_at": now(),
        },
    )
    budget_stop = False
    provider_stop = None
    lease = reference_lease(
        {"jobs_dir": "jobs", "execution": {"memory_mb": 6144, "cpu": 4, "min_free_disk_gb": 7}}
    )
    try:
        with lease:
            snapshot()
            while lanes or pending:
                for lane, item in list(lanes.items()):
                    proc = processes.get(item["pid"])
                    running = proc.poll() is None if proc else alive(item)
                    if running:
                        continue
                    code = proc.returncode if proc else None
                    if item["pid"] in streams:
                        streams.pop(item["pid"]).close()
                    matches = [
                        job
                        for job in source_jobs(record["id"])
                        if job["cohort_id"] == item["id"]
                    ]
                    outcome = matches[-1]["status"] if matches else "missing_series_record"
                    done = {**item, "exit_code": code, "status": outcome, "finished_at": now()}
                    completed.append(done)
                    del lanes[lane]
                    if outcome == "budget_limit_reached":
                        budget_stop = True
                    elif outcome == "infrastructure_error":
                        errors = []
                        for match in matches:
                            path = Path("jobs") / match["id"] / "stderr.txt"
                            if path.exists():
                                errors.append(path.read_text().lower())
                        combined = "\n".join(errors)
                        if any(token in combined for token in ("http 401", "http 403", "http 429", "credit balance", "rate limit")):
                            provider_stop = "Anthropic authorization, quota, or rate-limit failure"
                        elif not live_cache_validated(record["id"]):
                            provider_stop = "Unvalidated pilot encountered an infrastructure error"
                    print(json.dumps({"finished": done}), flush=True)
                    snapshot()
                budget = ledger.snapshot()
                can_reserve = budget["remaining_unreserved_usd"] + 1e-9 >= REQUEST_RESERVATION_USD
                if not can_reserve:
                    budget_stop = True
                scaled = live_cache_validated(record["id"])
                for lane in LANES:
                    if lane in lanes or not pending or budget_stop or provider_stop:
                        continue
                    if not scaled and (lanes or lane != LANES[0]):
                        continue
                    require_disk_headroom("jobs", 7)
                    row = pending.pop(0)
                    launch(lane, row)
                    snapshot()
                if not lanes and (not pending or budget_stop or provider_stop):
                    break
                time.sleep(10)
        if provider_stop:
            record.update(status="provider_paused", provider_pause_reason=provider_stop)
        elif budget_stop and pending:
            record.update(status="budget_limit_reached", unstarted_task_ids=[row["id"] for row in pending])
        elif pending:
            record.update(status="stopped_with_pending", unstarted_task_ids=[row["id"] for row in pending])
        else:
            record["status"] = "completed_unscored"
    except BaseException:
        record["status"] = "scheduler_error"
        (directory / "stderr.txt").write_text(traceback.format_exc())
        record["artifacts"]["stderr"] = "stderr.txt"
        raise
    finally:
        for stream in streams.values():
            stream.close()
        snapshot()
        record["finished_at"] = now()
        save(directory, record)


def status():
    if not POINTER.exists():
        print(json.dumps({"status": "not_started"}, indent=2))
        return
    pointer = json.loads(POINTER.read_text())
    directory = Path(pointer["directory"])
    record = json.loads((directory / "job.json").read_text())
    value = {
        "campaign": record,
        "progress": json.loads(PROGRESS.read_text()) if PROGRESS.exists() else None,
        "head_to_head": json.loads((directory / "head-to-head.json").read_text())
        if (directory / "head-to-head.json").exists()
        else None,
    }
    print(json.dumps(value, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "run", "status"])
    args = parser.parse_args()
    if args.mode == "preflight":
        print(json.dumps(cache_preflight(), indent=2))
    elif args.mode == "run":
        run()
    else:
        status()


if __name__ == "__main__":
    main()
