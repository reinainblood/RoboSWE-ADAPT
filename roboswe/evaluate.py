"""Re-verify completed campaign code without modifying the original records."""

import json
import traceback
from pathlib import Path

from .campaign import (
    publish_video,
    reference_lease,
    require_execution,
    trial_provenance,
)
from .execution.sandbox import CodeSandbox
from .jobs import create, now, save
from .library.attempts import read_journal, replay_journal
from .library.metrics import score_library
from .library.simulator import SimulatorProcess
from .provenance import snapshot_harness


def evaluate_campaign(job, jobs_dir="jobs", *, memory_mb=None):
    source = Path(job)
    if not source.exists():
        source = Path(jobs_dir) / job
    source = source.resolve()
    original = json.loads((source / "job.json").read_text())
    if original["status"] == "running":
        raise ValueError(
            "Cannot evaluate an active campaign; its live task records remain available"
        )
    plan = json.loads((source / "plan.json").read_text())
    require_execution(plan)
    # Replay can use an available allocation smaller than an old overlarge cap;
    # never silently alter a saved plan here. The operator must choose a valid run.
    sequences = sorted(source.glob("sequence-*/submissions.jsonl"))
    if not sequences:
        raise ValueError("No submitted-code sequences in this campaign")
    entries = [read_journal(path) for path in sequences]
    if any(
        not events or events[0]["kind"] != "initial" or "spec" not in events[0]["data"]
        for events in entries
    ):
        raise ValueError("Campaign lacks complete simulator/spec provenance")
    # Do not construct a simulator merely to discover an obviously unfinished log.
    for events in entries:
        active = pending = False
        for event in events:
            kind = event["kind"]
            if kind == "start_task_request":
                active = True
            elif kind == "finish_task":
                active = False
            elif kind == "operation_request":
                pending = True
            elif kind == "operation_result":
                pending = False
        if active or pending:
            raise ValueError(
                "Submission sequence is incomplete and cannot be certified"
            )
    execution = dict(plan["execution"])
    if memory_mb is not None:
        if type(memory_mb) is not int or memory_mb <= 0:
            raise ValueError("Verifier memory limit must be positive")
        execution["memory_mb"] = memory_mb
    with reference_lease({**plan, "execution": execution}) as capacity:
        return _evaluate_sequences(
            original, plan, sequences, entries, jobs_dir, execution, capacity
        )


def _evaluate_sequences(
    original, plan, sequences, entries, jobs_dir, execution, capacity
):
    directory, record = create(
        jobs_dir,
        f"Independent verification · {original['name']}",
        "independent-verification",
    )
    record["original_campaign"] = original["id"]
    record["verification_resources"] = {
        "cpu": execution["cpu"],
        "memory_mb": execution["memory_mb"],
        "capacity": capacity,
    }
    record["coordinator"] = snapshot_harness(directory)
    record["artifacts"].update(
        coordinator="coordinator.json", coordinator_source="coordinator-source.zip"
    )
    reports = []
    try:
        for index, (journal, events) in enumerate(zip(sequences, entries, strict=True)):
            initial = events[0]["data"]
            config, spec = initial["config"], initial["spec"]
            if (
                initial["code_image"] != execution["code_image"]
                or initial.get("simulator_image") != execution["image"]
            ):
                raise ValueError(
                    "Original image provenance differs from the saved plan"
                )
            output = directory / f"sequence-{index}"
            output.mkdir()
            with (
                SimulatorProcess(
                    config,
                    spec,
                    output / "simulator",
                    image=execution["image"],
                    role="verifier",
                    timeout=execution["timeout_seconds"],
                    max_steps=execution["max_control_steps"],
                    lifetime_seconds=execution["lifetime_seconds"],
                    cpu=execution["cpu"],
                    memory_mb=execution["memory_mb"],
                ) as runtime,
                CodeSandbox(execution["code_image"]) as sandbox,
            ):
                runtime.record_start(
                    "verification",
                    width=plan["environment"]["camera_width"],
                    height=plan["environment"]["camera_height"],
                )
                try:
                    report = replay_journal(
                        journal, runtime, sandbox, output / "replay.jsonl"
                    )
                finally:
                    runtime.record_stop("verification")
            model = json.loads((journal.parent / "model.json").read_text())
            for trial in report["trials"]:
                trial.update(
                    trial_provenance(
                        plan,
                        config,
                        model,
                        retained=plan["mode"] == "library" and trial["index"] > 0,
                    )
                )
            if plan["mode"] == "library":
                # Never borrow a score or unaudited cold result from the original.
                report["metrics"] = score_library(spec, report["trials"])
            reports.append(report)
            video_dir, video_record = create(
                jobs_dir,
                f"Verification video · {original['name']} · sequence {index}",
                "verification-video",
            )
            publish_video(
                output / "simulator/recordings/verification",
                video_dir,
                video_record,
                phase="replay",
            )
            video_record.update(
                status="reproduced" if report["reproduced"] else "diverged",
                video_provenance="independent code replay",
                finished_at=now(),
            )
            (video_dir / "report.json").write_text(json.dumps(report, indent=2))
            video_record["artifacts"]["report"] = "report.json"
            save(video_dir, video_record)
            (directory / "report.json").write_text(
                json.dumps({"sequences": reports}, indent=2)
            )
            record["artifacts"]["report"] = "report.json"
            save(directory, record)
        reproduced = all(r["reproduced"] for r in reports)
        record.update(
            status="reproduced" if reproduced else "diverged",
            exit_code=0 if reproduced else 1,
        )
    except BaseException:
        record.update(status="error", exit_code=1)
        (directory / "stderr.txt").write_text(traceback.format_exc())
        record["artifacts"]["stderr"] = "stderr.txt"
        raise
    finally:
        record["finished_at"] = now()
        save(directory, record)
    return directory
