"""Cold participants at replay-verified task boundaries, without state teleporting."""

import hashlib
import json
import time
from pathlib import Path

from roboswe.execution.sandbox import CodeSandbox
from roboswe.storage import require_disk_headroom

from .attempts import AttemptRunner, Journal, read_journal, replay_journal
from .video_judge import publish_visual_audit


def prefix_before_task(original, index, output):
    """Preserve the original hash chain and all partial failed executions."""
    entries = read_journal(original)
    boundary = next(
        (
            i
            for i, e in enumerate(entries)
            if e["kind"] == "start_task_request" and e["data"]["index"] == index
        ),
        None,
    )
    if boundary is None:
        raise ValueError("Missing requested task boundary")
    lines = Path(original).read_text().splitlines(keepends=True)
    with Path(output).open("x") as target:
        target.writelines(lines[:boundary])
    return entries[0]["data"]


def combined_journal(prefix, cold, output):
    """Replay warm prefix, discard its files, then rerun actual cold submissions."""
    before, after = read_journal(prefix), read_journal(cold)
    if before[0]["kind"] != "initial" or after[0]["kind"] != "initial":
        raise ValueError("Missing journal provenance")
    for key in (
        "config",
        "initial_fingerprint",
        "spec_sha256",
        "code_image",
        "simulator_image",
    ):
        if before[0]["data"].get(key) != after[0]["data"].get(key):
            raise ValueError(f"Cold journal provenance differs: {key}")
    journal = Journal(output)
    try:
        initial = dict(before[0]["data"])
        initial["composition"] = {
            "prefix_sha256": hashlib.sha256(Path(prefix).read_bytes()).hexdigest(),
            "cold_sha256": hashlib.sha256(Path(cold).read_bytes()).hexdigest(),
        }
        journal.append("initial", initial)
        for event in before[1:]:
            journal.append(event["kind"], event["data"])
        journal.append("reset_code_workspace", {})
        for event in after[1:]:
            journal.append(event["kind"], event["data"])
    finally:
        journal.close()


def run_controls(
    plan,
    spec,
    config,
    sequence_dir,
    warm_report,
    model_metadata,
    *,
    model_factory,
    simulator_factory,
    progress=None,
):
    """Each control and its verifier begin by rerunning the same original prefix.

    This is intentionally more expensive than restoring only qpos: controller/Python
    state and failed partial executions are part of the physical setup too.
    """
    from roboswe.campaign import drive_task, publish_video, trial_provenance
    from roboswe.jobs import create, now, save

    execution = plan["execution"]
    simulator_args = {
        "image": execution["image"],
        "timeout": execution["timeout_seconds"],
        "max_steps": execution["max_control_steps"],
        "lifetime_seconds": execution["lifetime_seconds"],
        "cpu": execution.get("cpu", 4),
        "memory_mb": execution.get("memory_mb", 16384),
    }
    root = sequence_dir / "cold-controls"
    root.mkdir(exist_ok=False)
    controls = []
    warm = {trial["index"]: trial for trial in warm_report["trials"]}
    for task in spec["tasks"]:
        if task["round"] != 2:
            continue
        require_disk_headroom(sequence_dir)
        index = task["index"]
        if progress:
            progress(f"Cold comparison: {task['task']} (original task {index + 1})")
        trial = warm.get(index)
        if (
            not trial
            or not trial.get("precondition_valid")
            or not trial.get("reproduced")
            or trial.get("infrastructure_valid") is not True
        ):
            controls.append(
                {
                    "pair_key": task["pair_key"],
                    "precondition_valid": trial.get("precondition_valid")
                    if trial
                    else None,
                    "infrastructure_valid": False,
                    "status": "warm_trial_not_comparable",
                }
            )
            continue
        if not trial.get("start_fingerprint"):
            raise ValueError(
                "Warm run lacks boundary fingerprints; it cannot support matched-state control scoring"
            )
        directory = root / f"task-{index:03d}"
        directory.mkdir()
        prefix = directory / "prefix.jsonl"
        prefix_before_task(sequence_dir / "submissions.jsonl", index, prefix)
        job_dir, record = create(
            plan["jobs_dir"],
            f"Cold control · {task['task']} · seed {config['seed']}",
            "learning-control",
        )
        record.update(task=task["task"], verifier_status="awaiting_independent_replay")
        cold_journal = directory / "cold.jsonl"
        recording_id = f"cold-{index:03d}"
        started = time.monotonic()
        try:
            with simulator_factory(
                config, spec, directory / "original", **simulator_args
            ) as runtime:
                with CodeSandbox(execution["code_image"]) as prefix_sandbox:
                    prefix_result = replay_journal(
                        prefix,
                        runtime,
                        prefix_sandbox,
                        directory / "prefix-replay.jsonl",
                    )
                if (
                    not prefix_result["reproduced"]
                    or runtime.fingerprint() != trial["start_fingerprint"]
                ):
                    raise ValueError(
                        "Cold setup did not reproduce the warm task's exact initial physical state"
                    )
                # The prefix sandbox is gone. No original library or prior model
                # messages enter either this fresh workspace or this conversation.
                with (
                    CodeSandbox(execution["code_image"]) as sandbox,
                    model_factory(
                        model=model_metadata["model"],
                        effort=model_metadata["reasoning_effort"],
                        diagnostics=directory / "model-events.jsonl",
                        persistent=True,
                    ) as model,
                ):
                    journal = Journal(cold_journal)
                    runner = AttemptRunner(runtime, sandbox, journal)
                    runtime.record_start(
                        recording_id,
                        width=plan["environment"]["camera_width"],
                        height=plan["environment"]["camera_height"],
                    )
                    try:
                        with (directory / "model-transcript.jsonl").open(
                            "x"
                        ) as transcript:
                            cold_trial, _ = drive_task(
                                runner,
                                model,
                                index,
                                execution,
                                directory,
                                transcript,
                                started + execution["lifetime_seconds"],
                            )
                        record["library"] = sandbox.archive(job_dir / "library.zip")
                        record["artifacts"]["library"] = "library.zip"
                    finally:
                        if runner.task_active:
                            runner.finish_task(reason="infrastructure_interruption")
                        runtime.record_stop(recording_id)
                        journal.close()
            publish_video(
                directory / "original/recordings" / recording_id,
                job_dir,
                record,
                phase="original",
            )
            save(job_dir, record)
            combined = directory / "combined.jsonl"
            combined_journal(prefix, cold_journal, combined)
            with (
                simulator_factory(
                    config,
                    spec,
                    directory / "verification",
                    role="verifier",
                    **simulator_args,
                ) as runtime,
                CodeSandbox(execution["code_image"]) as sandbox,
            ):

                def on_start(
                    task_index,
                    target=index,
                    boundary=trial["start_fingerprint"],
                    video_id=recording_id,
                    sim=runtime,
                ):
                    if task_index == target:
                        if sim.fingerprint() != boundary:
                            raise ValueError(
                                "Cold verification start state differs from warm trial"
                            )
                        sim.record_start(
                            video_id,
                            width=plan["environment"]["camera_width"],
                            height=plan["environment"]["camera_height"],
                        )

                def on_finish(
                    replayed, target=index, video_id=recording_id, sim=runtime
                ):
                    if replayed["index"] == target:
                        sim.record_stop(video_id)

                replay = replay_journal(
                    combined,
                    runtime,
                    sandbox,
                    directory / "verification.jsonl",
                    on_start=on_start,
                    on_finish=on_finish,
                )
            cold_trial = replay["trials"][-1]
            cold_trial.update(
                trial_provenance(plan, config, model.metadata, retained=False)
            )
            cold_trial["infrastructure_valid"] = (
                cold_trial["infrastructure_valid"] and replay["reproduced"]
            )
            record.update(
                status="verified_pass"
                if cold_trial["verified_success"]
                else "verified_fail"
                if replay["reproduced"]
                else "replay_diverged",
                verifier_status="reproduced" if replay["reproduced"] else "diverged",
                video_provenance="independent cold code replay",
            )
            publish_video(
                directory / "verification/recordings" / recording_id,
                job_dir,
                record,
                phase="replay",
            )
            (job_dir / "report.json").write_text(json.dumps(cold_trial, indent=2))
            record["artifacts"]["report"] = "report.json"
            if execution.get("video_judge"):
                publish_visual_audit(
                    job_dir,
                    record,
                    cold_trial,
                    model=plan["agent"]["model"],
                    backend=execution.get("video_judge_backend", "codex"),
                )
            controls.append(cold_trial)
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            invalid = {
                "pair_key": task["pair_key"],
                "precondition_valid": trial.get("precondition_valid", False),
                "infrastructure_valid": False,
                "status": "control_infrastructure_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            controls.append(invalid)
            record.update(
                status="error", verifier_status="not_verified", error=invalid["error"]
            )
        finally:
            # Preserve original footage even if code replay or model contact failed.
            if "original" not in record.get("video_runs", {}):
                try:
                    publish_video(
                        directory / "original/recordings" / recording_id,
                        job_dir,
                        record,
                        phase="original",
                    )
                except (OSError, ValueError) as exc:
                    record["artifact_errors"].append(
                        {"stage": "recording_files", "error": str(exc)}
                    )
            record["finished_at"] = now()
            save(job_dir, record)
            (root / "controls.json").write_text(json.dumps(controls, indent=2))
    return controls
