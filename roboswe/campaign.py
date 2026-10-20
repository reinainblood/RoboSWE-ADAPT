"""The fixed RoboSWE participant loop and independent verification coordinator."""

import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from .codex_login import CodexLogin
from .execution.sandbox import CodeSandbox, docker_path
from .jobs import create, now, save
from .library.attempts import AttemptRunner, Journal, canonical, replay_journal
from .library.cold_controls import run_controls
from .library.metrics import score_library
from .library.serial import serial_spec
from .library.simulator import SimulatorProcess
from .library.video_judge import publish_visual_audit
from .storage import clone_or_copy, require_disk_headroom


def trial_provenance(plan, config, model, *, retained):
    import hashlib

    return {
        "config_sha256": hashlib.sha256(canonical(config)).hexdigest(),
        "protocol_sha256": hashlib.sha256(canonical(plan["execution"])).hexdigest(),
        "model": model["model"],
        "reasoning_effort": model.get("reasoning_effort", "unrecorded"),
        "context_policy": model.get("context_policy"),
        "task_timeout_seconds": plan["execution"]["timeout_seconds"],
        "retained_context": retained,
        "retained_library": retained,
    }


def require_execution(plan):
    execution = plan["execution"]
    if execution["backend"] != "docker":
        raise ValueError(
            "Campaign execution currently supports Docker only; Modal remains plan-only"
        )
    if execution["concurrency"] != 1:
        raise ValueError(
            "The reference campaign runner currently requires --concurrency 1"
        )
    for field in ("image", "code_image"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", execution.get(field) or ""):
            raise ValueError(
                f"--{field.replace('_', '-')} must be an existing immutable Docker image ID"
            )
    if plan["agent"]["api_base_url"] or plan["agent"]["api_key_env"]:
        raise ValueError(
            "This reference connection uses the local Codex login; API options are reserved"
        )


def check_capacity(execution, info, running_simulators):
    required = (execution["memory_mb"] + 1024 + 512) * 1024 * 1024
    if required > info["MemTotal"]:
        raise ValueError(
            f"Requested simulator/code/workspace limits total {required / 2**30:.1f} GiB, but Docker has {info['MemTotal'] / 2**30:.1f} GiB"
        )
    if execution["cpu"] > info["NCPU"]:
        raise ValueError(
            "Requested simulator CPU limit exceeds Docker's available CPUs"
        )
    if running_simulators:
        raise ValueError(
            "A RoboSWE simulator is already running; reference campaigns must run serially on this Docker host"
        )


def ensure_docker_capacity(plan):
    storage = require_disk_headroom(
        plan.get("jobs_dir", "jobs"), plan["execution"].get("min_free_disk_gb", 10)
    )
    docker = docker_path()
    info = json.loads(
        subprocess.check_output(
            [docker, "info", "--format", "{{json .}}"], text=True, timeout=20
        )
    )
    active = subprocess.check_output(
        [docker, "ps", "--filter", "label=roboswe=simulator", "--format", "{{.Names}}"],
        text=True,
        timeout=20,
    ).splitlines()
    check_capacity(plan["execution"], info, active)
    return {
        "docker_memory_bytes": info["MemTotal"],
        "docker_cpus": info["NCPU"],
        "concurrent_simulators": 0,
        "host_storage": storage,
    }


@contextmanager
def reference_lease(plan):
    """Reserve this host for a reference campaign across replay/control gaps."""
    import fcntl

    path = Path(tempfile.gettempdir()) / "roboswe-reference-campaign.lock"
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "Another reference campaign or verification is active on this host"
            ) from exc
        try:
            yield ensure_docker_capacity(plan)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def model_input(feedback, observation, directory, index):
    """Decode only coordinator-owned images; never accept agent-selected paths."""
    directory.mkdir(parents=True, exist_ok=True)
    images = []
    for camera_index, (_name, data) in enumerate(
        observation["images_png_base64"].items()
    ):
        path = directory / f"turn-{index:05d}-camera-{camera_index}.png"
        path.write_bytes(base64.b64decode(data, validate=True))
        images.append(path)
    state = {
        k: v
        for k, v in observation.items()
        if k not in {"images_png_base64", "depth_npz_base64"}
    }
    if isinstance(feedback, dict):
        feedback = {
            k: v
            for k, v in feedback.items()
            if k not in {"images_png_base64", "depth_npz_base64"}
        }
    return json.dumps(
        {
            "feedback": feedback,
            "observation": state,
            "attached_cameras_in_order": list(observation["images_png_base64"]),
            "api": "import robot; robot.observe() or robot.step(native_action_vector). Sensor vectors are JSON lists; RGB is base64 PNG. depth_arrays is metadata ONLY. Actual depth is in the top-level depth_npz_base64 returned to Python by robot.observe(); the encoded payload is omitted from this text prompt. Decode it with base64.b64decode and numpy.load(BytesIO(...), allow_pickle=False), then select the named array using camera_calibration[image_key]['depth_key']. Use calibration from the same observation, not cached transforms from before robot motion. No prebuilt motion tools. Code persists in /workspace. write only saves source; execute separately runs it. File reads/writes support relative .py files and NOTES.md; execute supports .py only. Every robot.step is one control tick, not confirmation of reaching a target. A code error does not reset the robot.",
        },
        allow_nan=False,
    ), images


def publish_video(source, directory, record, *, phase=None):
    if phase not in (None, "original", "replay"):
        raise ValueError("Unknown recording phase")
    manifest = json.loads((source / "video.json").read_text())
    for filename in manifest["files"]:
        if filename not in {"video.json", "video-timeline.jsonl"} and not re.fullmatch(
            r"video-(overview|camera-\d+)-(wall|simulation)\.mp4", filename
        ):
            raise ValueError("Invalid trusted recording filename")
    record["video"] = manifest
    for filename in manifest["files"]:
        clone_or_copy(source / filename, directory / filename)
        record["artifacts"][filename] = filename
        if phase:
            saved = f"{phase}-{filename}"
            clone_or_copy(source / filename, directory / saved)
            record["artifacts"][saved] = saved
    if phase:
        variant = json.loads(json.dumps(manifest))
        variant["files"] = [f"{phase}-{name}" for name in manifest["files"]]
        variant["timeline"] = f"{phase}-video-timeline.jsonl"
        for view in variant["views"]:
            for clock in ("wall", "simulation"):
                if view.get(clock):
                    view[clock] = f"{phase}-{view[clock]}"
        (directory / f"{phase}-video.json").write_text(json.dumps(variant, indent=2))
        record.setdefault("video_runs", {})[phase] = variant
        record["default_video_run"] = phase
    for error in manifest.get("errors", []):
        record["artifact_errors"].append({"stage": "video_recording", "error": error})


def drive_task(
    runner, model, index, execution, sequence_dir, transcript, deadline, turn=0
):
    """Identical bounded participant loop for retained and cold conditions."""
    runtime = runner.runtime
    feedback = runner.start_task(index)
    observation = feedback.pop("observation")
    feedback["attempt_limits"] = {
        "max_control_steps": execution.get("max_attempt_steps", 50),
        "wall_seconds": execution.get("attempt_timeout_seconds", 60),
    }
    start_step = observation["control_step"]
    stop_reason = "precondition_invalid"
    first_request = True
    if feedback["precondition_valid"]:
        while runner.task_active:
            require_disk_headroom(sequence_dir)
            remaining = min(
                runtime.timeout - (time.monotonic() - runtime.started),
                deadline - time.monotonic(),
            )
            if remaining <= 0:
                stop_reason = "task_deadline"
                break
            if (
                observation["control_step"] - start_step
                >= execution["max_control_steps"]
            ):
                stop_reason = "control_step_budget"
                break
            prompt, images = model_input(
                feedback, observation, sequence_dir / "model-images", turn
            )
            transcript.write(
                json.dumps(
                    {
                        "event": "request",
                        "task_index": index,
                        "turn": turn,
                        "prompt": prompt,
                        "images": [p.name for p in images],
                        "input_kind": "user_instruction"
                        if first_request
                        else "tool_output",
                    }
                )
                + "\n"
            )
            transcript.flush()
            try:
                decision = model.ask(
                    prompt,
                    images=images,
                    timeout=remaining,
                    as_tool_output=not first_request,
                )
                first_request = False
                require_disk_headroom(sequence_dir)
            except TimeoutError:
                stop_reason = "model_deadline"
                break
            transcript.write(
                json.dumps(
                    {
                        "event": "decision",
                        "task_index": index,
                        "turn": turn,
                        "decision": decision,
                    }
                )
                + "\n"
            )
            transcript.flush()
            turn += 1
            try:
                feedback = runner.apply(
                    decision,
                    timeout=min(
                        execution.get("attempt_timeout_seconds", 60), remaining
                    ),
                    max_control_steps=execution.get("max_attempt_steps", 50),
                )
            except ValueError as exc:
                feedback = {"operation": decision["operation"], "path": decision["path"], "protocol_error": str(exc)}
            except TimeoutError:
                stop_reason = "task_deadline"
                break
            if not runner.task_active:
                return feedback, turn
            observation = runtime.observe()
            if observation.get("simulator_phase") != "active":
                raise RuntimeError("Simulator faulted; this trial cannot be certified")
            if observation.get("task_success"):
                stop_reason = "sustained_success"
                break
    return runner.finish_task(reason=stop_reason), turn


def run_sequence(
    plan,
    spec,
    config,
    campaign_dir,
    *,
    model_factory=CodexLogin,
    simulator_factory=SimulatorProcess,
    progress=None,
):
    """One persistent kitchen, workspace and model conversation; then fresh replay."""
    execution = plan["execution"]
    sequence_dir = (
        campaign_dir
        / f"sequence-{config['seed']}-{len(list(campaign_dir.glob('sequence-*')))}"
    )
    sequence_dir.mkdir()
    journal = Journal(sequence_dir / "submissions.jsonl")
    tasks = []
    trials = []
    transcript = (sequence_dir / "model-transcript.jsonl").open("x")
    started = time.monotonic()
    simulator_args = {
        "image": execution["image"],
        "timeout": execution["timeout_seconds"],
        "max_steps": execution["max_control_steps"],
        "lifetime_seconds": execution["lifetime_seconds"],
        "cpu": execution.get("cpu", 4),
        "memory_mb": execution.get("memory_mb", 16384),
    }
    try:
        with (
            simulator_factory(
                config, spec, sequence_dir / "original", **simulator_args
            ) as runtime,
            CodeSandbox(
                execution["code_image"], lifetime_seconds=execution["lifetime_seconds"]
            ) as sandbox,
            model_factory(
                model=plan["agent"]["model"],
                effort=plan["agent"].get("reasoning_effort", "medium"),
                diagnostics=sequence_dir / "model-events.jsonl",
                persistent=True,
            ) as model,
        ):
            (sequence_dir / "model.json").write_text(
                json.dumps(model.metadata, indent=2)
            )
            runner = AttemptRunner(runtime, sandbox, journal)
            turn = 0
            for task in spec["tasks"]:
                if time.monotonic() - started >= execution["lifetime_seconds"]:
                    raise TimeoutError("Sequence container lifetime exhausted")
                index = task["index"]
                if progress:
                    progress(
                        f"Task {index + 1}/{len(spec['tasks'])}: {task['task']} (round {task['round']}, seed {config['seed']})"
                    )
                directory, record = create(
                    plan["jobs_dir"],
                    f"{plan['mode']} · {task['task']} · round {task['round']} · seed {config['seed']}",
                    "benchmark-task",
                )
                tasks.append((directory, record))
                record.update(
                    task=task["task"],
                    campaign_id=campaign_dir.name,
                    task_index=index,
                    verifier_status="awaiting_independent_replay",
                )
                save(directory, record)
                recording_id = f"task-{index:03d}"
                active = False
                recording_started = False
                completed = False
                try:
                    runtime.record_start(
                        recording_id,
                        width=plan["environment"]["camera_width"],
                        height=plan["environment"]["camera_height"],
                    )
                    recording_started = True
                    active = True
                    trial, turn = drive_task(
                        runner,
                        model,
                        index,
                        execution,
                        sequence_dir,
                        transcript,
                        started + execution["lifetime_seconds"],
                        turn,
                    )
                    trials.append(trial)
                    active = False
                    record["library"] = sandbox.archive(directory / "library.zip")
                    record["artifacts"]["library"] = "library.zip"
                    record.update(
                        status="awaiting_verification",
                        observed_success=trials[-1]["observed_success"],
                        trial_summary={
                            key: trials[-1][key]
                            for key in (
                                "control_steps",
                                "elapsed_seconds",
                                "code_errors",
                                "termination_reason",
                            )
                        },
                    )
                    (directory / "stdout.txt").write_text(
                        json.dumps(trials[-1], indent=2)
                    )
                    record["artifacts"]["stdout"] = "stdout.txt"
                    completed = True
                    if progress:
                        trial = trials[-1]
                        outcome = (
                            "observed success"
                            if trial["observed_success"]
                            else "precondition invalid"
                            if not trial["precondition_valid"]
                            else "not completed"
                        )
                        progress(
                            f"Task {index + 1}: {outcome}; {trial['control_steps']} control ticks, {trial['elapsed_seconds']:.1f}s live elapsed; awaiting code replay"
                        )
                finally:
                    if not completed:
                        record["status"] = "interrupted" if active else "error"
                    if active and runner.task_active:
                        try:
                            runner.finish_task(reason="infrastructure_interruption")
                        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                            record["artifact_errors"].append(
                                {"stage": "task_finalization", "error": str(exc)}
                            )
                        record["status"] = "interrupted"
                    try:
                        if recording_started:
                            runtime.record_stop(recording_id)
                    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                        record["artifact_errors"].append(
                            {"stage": "recording_finalization", "error": str(exc)}
                        )
                    try:
                        publish_video(
                            sequence_dir / "original/recordings" / recording_id,
                            directory,
                            record,
                            phase="original",
                        )
                    except (OSError, ValueError) as exc:
                        record["artifact_errors"].append(
                            {"stage": "recording_files", "error": str(exc)}
                        )
                    record["finished_at"] = now()
                    save(directory, record)
    finally:
        transcript.close()
        journal.close()
    # Never replay inside the live agent simulator or reuse its writable volume.
    require_disk_headroom(sequence_dir)
    if progress:
        progress(f"Starting independent code replay for seed {config['seed']}")
    with (
        simulator_factory(
            config,
            spec,
            sequence_dir / "verification",
            role="verifier",
            **simulator_args,
        ) as runtime,
        CodeSandbox(
            execution["code_image"], lifetime_seconds=execution["lifetime_seconds"]
        ) as sandbox,
    ):

        def start_replay(index):
            runtime.record_start(
                f"task-{index:03d}",
                width=plan["environment"]["camera_width"],
                height=plan["environment"]["camera_height"],
            )

        def finish_replay(trial):
            index = trial["index"]
            runtime.record_stop(f"task-{index:03d}")
            directory, record = tasks[index]
            trial.update(
                trial_provenance(
                    plan,
                    config,
                    model.metadata,
                    retained=plan["mode"] == "library" and trial["index"] > 0,
                )
            )
            # Keep original videos in the sequence archive; show trusted replay
            # footage as the benchmark task's default review video.
            publish_video(
                sequence_dir / "verification/recordings" / f"task-{index:03d}",
                directory,
                record,
                phase="replay",
            )
            record.update(
                status="verified_pass"
                if trial["verified_success"]
                else "verified_fail"
                if trial["reproduced"]
                else "replay_diverged",
                verified_success=trial["verified_success"],
                verifier_status="reproduced" if trial["reproduced"] else "diverged",
                video_provenance="independent code replay",
            )
            (directory / "report.json").write_text(json.dumps(trial, indent=2))
            record["artifacts"]["report"] = "report.json"
            save(directory, record)
            if progress:
                outcome = (
                    "verified pass"
                    if trial["verified_success"]
                    else "verified fail"
                    if trial["reproduced"]
                    else "unverified: replay diverged"
                )
                progress(f"Replay task {index + 1}: {outcome}")

        report = replay_journal(
            sequence_dir / "submissions.jsonl",
            runtime,
            sandbox,
            sequence_dir / "replay.jsonl",
            on_start=start_replay,
            on_finish=finish_replay,
        )
    if execution.get("video_judge"):
        if progress:
            progress("Starting optional visual audits of replay recordings")
        for trial, (directory, record) in zip(report["trials"], tasks, strict=True):
            publish_visual_audit(
                directory,
                record,
                trial,
                model=plan["agent"]["model"],
                backend=execution.get("video_judge_backend", "codex"),
            )
            save(directory, record)
    if plan["mode"] == "library":
        controls = None
        if execution.get("learning_controls"):
            controls = run_controls(
                plan,
                spec,
                config,
                sequence_dir,
                report,
                model.metadata,
                model_factory=model_factory,
                simulator_factory=simulator_factory,
                progress=progress,
            )
        report["metrics"] = score_library(
            spec, report["trials"], cold_controls=controls
        )
    (sequence_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def run_campaign(plan):
    require_execution(plan)
    with reference_lease(plan) as capacity:
        return _run_campaign(plan, capacity)


def _run_campaign(plan, capacity):
    from .provenance import snapshot_harness

    directory, record = create(
        plan["jobs_dir"],
        plan["name"] or f"RoboSWE {plan['mode']} campaign",
        "benchmark-campaign",
    )
    (directory / "plan.json").write_text(json.dumps(plan, indent=2))
    record["artifacts"]["plan"] = "plan.json"
    record["capacity_at_start"] = capacity
    record["coordinator"] = snapshot_harness(directory)
    record["artifacts"].update(
        coordinator="coordinator.json", coordinator_source="coordinator-source.zip"
    )
    save(directory, record)

    def progress(message):
        print(f"[{now()}] {message}", file=sys.stderr, flush=True)

    progress(f"Campaign {record['id']} · {directory}")
    reports = []
    try:
        environment = plan["environment"]
        execution = plan["execution"]
        for seed in plan["episode_seeds"]:
            selected = (
                [None] if plan["mode"] == "library" else plan["selection"]["tasks"]
            )
            for task in selected:
                spec = (
                    deepcopy(plan["library"])
                    if task is None
                    else serial_spec(
                        task.removeprefix("robocasa/"),
                        timeout=execution["timeout_seconds"],
                        max_steps=execution["max_control_steps"],
                    )
                )
                config = {
                    "robot": environment["robot"],
                    "task": task or "robocasa/Kitchen",
                    "split": environment["split"],
                    "seed": seed,
                    "env_kwargs": {
                        "camera_widths": environment["camera_width"],
                        "camera_heights": environment["camera_height"],
                        "camera_depths": environment["camera_depths"],
                    },
                }
                for key in ("layout_ids", "style_ids"):
                    if environment[key] is not None:
                        config["env_kwargs"][key] = environment[key]
                if task is None:
                    config["env_kwargs"]["init_robot_base_ref"] = spec[
                        "initial_robot_base_ref"
                    ]
                reports.append(
                    run_sequence(plan, spec, config, directory, progress=progress)
                )
                (directory / "report.json").write_text(
                    json.dumps({"sequences": reports}, indent=2)
                )
                record["artifacts"]["report"] = "report.json"
                save(directory, record)
        record.update(status="completed", exit_code=0)
    except BaseException as exc:
        record.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
            exit_code=1,
        )
        (directory / "stderr.txt").write_text(traceback.format_exc())
        record["artifacts"]["stderr"] = "stderr.txt"
        raise
    finally:
        record["finished_at"] = now()
        save(directory, record)
    return directory
