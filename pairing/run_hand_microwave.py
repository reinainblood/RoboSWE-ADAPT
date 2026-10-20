"""Two-task, unscored Inspire-hand trial using the normal RoboSWE coding loop.

The trusted simulator runs locally. Generated Python runs only in CodeSandbox.
This is an interactive embodiment trial, not a controlled benchmark comparison.
"""

import argparse
import hashlib
import json
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pairing.panda_inspire import register_panda_inspire_hand
from roboswe.campaign import drive_task, publish_video, reference_lease
from roboswe.codex_login import INSTRUCTIONS, CodexLogin
from roboswe.execution.sandbox import CodeSandbox
from roboswe.jobs import create, now, save
from roboswe.library.attempts import AttemptRunner, Journal, canonical
from roboswe.library.runtime import LibraryRuntime
from roboswe.library.serial import SerialRuntime, serial_spec
from roboswe.library.spec import GOALS, library_set
from robot_session.recording import MP4Writer
from robot_session.session import RobotSession


class CoachedCodexLogin(CodexLogin):
    """Trial-only attachment of a human demonstration to the first model turn."""

    coaching_image = None
    coaching_receipt = None

    def ask(self, text, *, images=(), **kwargs):
        if self.coaching_image is not None:
            images = (*images, self.coaching_image)
            text += "\nThe final attached image is Kirsten's real-hand bagel demonstration, not a simulator observation. Follow the human coaching in your instructions; infer simulator geometry from the robot cameras."
            self.coaching_receipt.write_text(json.dumps({
                "prompt": text,
                "images": [str(p) for p in images],
                "demonstration_sha256": hashlib.sha256(self.coaching_image.read_bytes()).hexdigest(),
            }, indent=2))
            answer = super().ask(text, images=images, **kwargs)
            self.coaching_image = None
            return answer
        return super().ask(text, images=images, **kwargs)


class HandSession(RobotSession):
    def __init__(self, config):
        import mujoco

        super().__init__(config)
        self.hand_writer = None
        self.mujoco = mujoco
        self.renderer = mujoco.Renderer(
            self.raw.sim.model._model, height=480, width=640
        )
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance, self.camera.elevation, self.camera.azimuth = 0.48, -15, 0
        self.options = mujoco.MjvOption()
        self.options.geomgroup[0] = 0
        self.options.sitegroup[:] = 0
        prefix = self.raw.robots[0].gripper["right"].naming_prefix
        self.hand_geoms = [
            i
            for i in range(self.raw.sim.model._model.ngeom)
            if self.raw.sim.model._model.geom(i).name.startswith(prefix)
        ]

    def hand_frame(self):
        model, data = self.raw.sim.model._model, self.raw.sim.data._data
        self.camera.lookat[:] = np.mean(data.geom_xpos[self.hand_geoms], axis=0)
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.options)
        for geom in self.renderer.scene.geoms[: self.renderer.scene.ngeom]:
            if geom.objtype == self.mujoco.mjtObj.mjOBJ_GEOM:
                name = model.geom(geom.objid).name
                if not name.startswith(("robot0_", "gripper0_", "mobilebase0_")):
                    geom.rgba[3] = 0
        return self.renderer.render().copy()

    def start_recording(self, payload):
        manifest = super().start_recording(payload)
        self.hand_start_step = self.steps
        self.hand_writer = MP4Writer(
            self.recording.directory / "video-camera-3-simulation.mp4", 640, 480, 20
        )
        self.hand_writer.append(self.hand_frame(), 0)
        return manifest

    def record_event(self, event):
        super().record_event(event)
        if event == "step" and self.hand_writer is not None:
            self.hand_writer.append(
                self.hand_frame(), (self.steps - self.hand_start_step) / 20
            )

    def stop_recording(self, payload):
        if self.hand_writer is not None:
            self.hand_writer.close()
            self.hand_writer = None
        manifest = super().stop_recording(payload)
        if not any(v["id"] == "camera-3" for v in manifest["views"]):
            filename = "video-camera-3-simulation.mp4"
            manifest["views"].append(
                {
                    "id": "camera-3",
                    "label": "Hand close-up · surrounding fixtures hidden",
                    "width": 640,
                    "height": 480,
                    "simulation": filename,
                }
            )
            manifest["files"].append(filename)
            (self.recording.directory / "video.json").write_text(
                json.dumps(manifest, indent=2)
            )
        return manifest

    def close(self):
        if self.hand_writer is not None:
            self.hand_writer.close()
            self.hand_writer = None
        self.renderer.close()
        super().close()


def hand_observation(result):
    result["hand_interface"] = {
        "model": "PandaInspireRightHand",
        "commands": [
            "pinky curl",
            "ring curl",
            "middle curl",
            "index curl",
            "thumb curl",
            "thumb opposition",
        ],
        "range": [-1, 1],
        "semantics": "Six absolute normalized targets in the right_gripper action slice. -1 is each actuator's minimum/open position; +1 is its maximum flexion/opposition. Commands are coupled into twelve physical joint actuators. This is not the scalar Panda parallel-jaw gripper. The eef/grip-site pose is at the hand mount, not the fingertips.",
    }
    return result


class HandRuntime(LibraryRuntime):
    def observe(self):
        return hand_observation(super().observe())

    def metadata(self):
        return {
            **super().metadata(),
            "development_runtime": True,
            "simulator_image": None,
        }


class HandSerialRuntime(SerialRuntime):
    def observe(self):
        return hand_observation(super().observe())

    def metadata(self):
        return {
            **super().metadata(),
            "development_runtime": True,
            "simulator_image": None,
        }


@contextmanager
def trial_slot(plan, wait):
    deadline = time.monotonic() + 3600
    while True:
        lease = reference_lease(plan)
        try:
            capacity = lease.__enter__()
        except ValueError as exc:
            if (
                not wait
                or "Another reference campaign or verification is active"
                not in str(exc)
            ):
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the simulator slot") from exc
            time.sleep(2)
            continue
        break
    try:
        yield capacity
    finally:
        lease.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-image", required=True)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--timeout", type=int, default=720)
    parser.add_argument(
        "--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max")
    )
    parser.add_argument("--detailed-explanations", action="store_true")
    parser.add_argument(
        "--official-task",
        choices=("GetToastedBread", "ToastBagel", "BlendIngredients"),
    )
    parser.add_argument("--wait-for-slot", action="store_true")
    parser.add_argument("--coaching-file", type=Path)
    parser.add_argument("--coaching-image", type=Path)
    args = parser.parse_args()
    if bool(args.coaching_file) != bool(args.coaching_image):
        parser.error("Provide both --coaching-file and --coaching-image")
    coaching = args.coaching_file.read_text() if args.coaching_file else None
    if args.coaching_image:
        args.coaching_image.read_bytes()  # Validate before starting a simulator.

    execution = {
        "timeout_seconds": args.timeout,
        "max_control_steps": 6000,
        "max_attempt_steps": 50,
        "attempt_timeout_seconds": 60,
        "lifetime_seconds": 2 * args.timeout + 300,
        "code_image": args.code_image,
        "memory_mb": 6144,
        "cpu": 4,
        "min_free_disk_gb": 10,
    }
    register_panda_inspire_hand()
    spec = library_set()
    spec.update(id="inspire-microwave-hand-trial", rounds=1)
    spec["tasks"] = [
        {"index": i, "round": 1, "pair_key": name, **asdict(GOALS[name])}
        for i, name in enumerate(("OpenMicrowave", "CloseMicrowave"))
    ]
    spec.pop("sha256")
    spec["sha256"] = hashlib.sha256(canonical(spec)).hexdigest()
    if args.official_task:
        spec = serial_spec(args.official_task, timeout=args.timeout, max_steps=6000)
        execution["lifetime_seconds"] = args.timeout + 300
    config = {
        "robot": "PandaOmron",
        "task": "robocasa/Kitchen",
        "split": None,
        "seed": 0,
        "env_kwargs": {
            "layout_ids": 2,
            "style_ids": 1,
            "init_robot_base_ref": "microwave_main_group",
            "gripper_types": "PandaInspireRightHand",
            "camera_widths": 640,
            "camera_heights": 480,
            "camera_depths": True,
        },
    }
    if args.official_task:
        config["task"] = "robocasa/" + args.official_task
        config["env_kwargs"].pop("init_robot_base_ref")
    label = {"gpt-6-astra": "Astra", "gpt-5.6-sol": "5.6 Sol"}.get(
        args.model, args.model
    )
    instructions = INSTRUCTIONS
    if args.detailed_explanations:
        instructions = instructions.replace(
            "message is a brief explanation, not\nan execution claim.",
            "message is a public action rationale of 2–5 sentences, not an execution claim. "
            "Explain the observation supporting your choice, the intended physical contact "
            "and movement, the expected observable effect, and what feedback would make "
            "you revise the approach. Clearly distinguish observations from hypotheses. "
            "When revising, state what changed relative to the previous attempt.",
        )
    if coaching:
        instructions += "\n\nHUMAN-COACHED ATTEMPT — KIRSTEN'S DEMONSTRATION\n" + coaching
        label += " · Kirsten-coached"
    task_label = args.official_task or "open then close microwave"
    directory, record = create(
        "jobs", f"{label} + Inspire hand · {task_label}", "hand-model-trial"
    )
    record.update(
        benchmark_score=None, verification_status="interactive_unscored_trial"
    )
    if args.wait_for_slot:
        record["status"] = "queued"
    trial_config = {
        "config": config,
        "spec": spec,
        "execution": execution,
        "simulator_runtime": "local macOS",
        "generated_code_runtime": "isolated Docker",
        "purpose": "Human-requested hand trial; no causal hand comparison or learning score",
        "model": args.model,
        "reasoning_effort": args.effort,
        "explanation_policy": "detailed_action_rationale_v1"
        if args.detailed_explanations
        else "brief",
        "participant_instructions": instructions,
    }
    if coaching:
        demonstration = directory / ("human-demonstration" + args.coaching_image.suffix)
        demonstration.write_bytes(args.coaching_image.read_bytes())
        (directory / "human-coaching.txt").write_text(coaching)
        trial_config["human_coaching"] = {
            "author": "Kirsten",
            "text": "human-coaching.txt",
            "image": demonstration.name,
            "image_sha256": hashlib.sha256(demonstration.read_bytes()).hexdigest(),
            "comparison": "Separate coached attempt; original unaided trial preserved",
        }
        record["human_coached"] = True
    (directory / "plan.json").write_text(json.dumps(trial_config, indent=2))
    record["artifacts"]["plan"] = "plan.json"
    for file in ("panda_inspire.py", "run_hand_microwave.py"):
        (directory / file).write_bytes((Path(__file__).parent / file).read_bytes())
    save(directory, record)
    print(f"Trial {record['id']}", flush=True)
    session = journal = None
    trials = []
    try:
        with trial_slot(
            {"execution": execution, "jobs_dir": "jobs"}, args.wait_for_slot
        ):
            record["status"] = "running"
            save(directory, record)
            session = HandSession(config)
            session.recording_root = directory / "recordings"
            with (
                (directory / "actions.jsonl").open("x") as evidence,
                (directory / "model-transcript.jsonl").open("x") as transcript,
                CodeSandbox(
                    args.code_image, lifetime_seconds=execution["lifetime_seconds"]
                ) as sandbox,
                CoachedCodexLogin(
                    model=args.model,
                    effort=args.effort,
                    instructions=instructions,
                    reasoning_summary="detailed"
                    if args.detailed_explanations
                    else None,
                    verbosity="high" if args.detailed_explanations else None,
                    summary_log=directory / "reasoning-summaries.jsonl"
                    if args.detailed_explanations
                    else None,
                    persistent=True,
                    diagnostics=directory / "model-events.jsonl",
                ) as model,
            ):
                if coaching:
                    model.coaching_image = demonstration
                    model.coaching_receipt = directory / "human-coaching-input.json"
                runtime_type = HandSerialRuntime if args.official_task else HandRuntime
                runtime = runtime_type(
                    session,
                    spec,
                    timeout=args.timeout,
                    max_steps=6000,
                    evidence=evidence,
                )
                journal = Journal(directory / "submissions.jsonl")
                runner = AttemptRunner(runtime, sandbox, journal)
                (directory / "model.json").write_text(
                    json.dumps(model.metadata, indent=2)
                )
                deadline = time.monotonic() + execution["lifetime_seconds"]
                turn = 0
                for task in spec["tasks"]:
                    index = task["index"]
                    task_dir, task_record = create(
                        "jobs",
                        f"{label} + Inspire hand · {task['task']}",
                        "hand-model-task",
                    )
                    task_record.update(
                        task=task["task"],
                        campaign_id=record["id"],
                        task_index=index,
                        video_provenance="Original model hand trial",
                        verifier_status="observed_only",
                        human_coached=bool(coaching),
                    )
                    save(task_dir, task_record)
                    video_id = f"task-{index:03d}"
                    session.start_recording(
                        {"recording_id": video_id, "width": 640, "height": 480}
                    )
                    print(f"Starting {task['task']}", flush=True)
                    try:
                        trial, turn = drive_task(
                            runner,
                            model,
                            index,
                            execution,
                            directory,
                            transcript,
                            deadline,
                            turn,
                        )
                        trial.update(
                            retained_context=index > 0, retained_library=index > 0
                        )
                        trials.append(trial)
                        task_record["status"] = (
                            "observed_pass"
                            if trial["observed_success"]
                            else "observed_fail"
                        )
                        (task_dir / "report.json").write_text(
                            json.dumps(trial, indent=2)
                        )
                        task_record["artifacts"]["report"] = "report.json"
                        task_record["library"] = sandbox.archive(
                            task_dir / "library.zip"
                        )
                        task_record["artifacts"]["library"] = "library.zip"
                        print(
                            json.dumps(
                                {
                                    k: trial[k]
                                    for k in (
                                        "task",
                                        "observed_success",
                                        "control_steps",
                                        "elapsed_seconds",
                                        "termination_reason",
                                    )
                                }
                            ),
                            flush=True,
                        )
                    finally:
                        if runner.task_active:
                            runner.finish_task(reason="infrastructure_interruption")
                            task_record["status"] = "interrupted"
                        session.stop_recording({"recording_id": video_id})
                        publish_video(
                            directory / "recordings" / video_id,
                            task_dir,
                            task_record,
                            phase="original",
                        )
                        # Make the model-authored rationales readable beside the
                        # video, aligned to the observation each decision saw.
                        explanations = []
                        current = None
                        for line in (
                            (directory / "model-transcript.jsonl")
                            .read_text()
                            .splitlines()
                        ):
                            entry = json.loads(line)
                            if entry["event"] == "request":
                                current = entry
                            elif current and current["task_index"] == index:
                                observation = json.loads(current["prompt"])[
                                    "observation"
                                ]
                                step = observation["control_step"]
                                relative = (
                                    step - session.recording.manifest["start_step"]
                                ) / session.raw.control_freq
                                decision = entry["decision"]
                                explanations.append(
                                    f"Video {relative:.2f}s · control step {step} · "
                                    f"{decision['operation']} {decision['path']}\n{decision['message']}\n"
                                )
                        (task_dir / "stdout.txt").write_text(
                            "Model-authored action rationales. These explain intended behavior; "
                            "they do not prove contact or success. Times use simulation playback.\n\n"
                            + "\n".join(explanations)
                        )
                        task_record["artifacts"]["stdout"] = "stdout.txt"
                        task_record["finished_at"] = now()
                        save(task_dir, task_record)
                record["status"] = "completed_unscored"
                record["exit_code"] = 0
    except BaseException:
        record.update(status="error", exit_code=1)
        (directory / "stderr.txt").write_text(traceback.format_exc())
        record["artifacts"]["stderr"] = "stderr.txt"
        raise
    finally:
        if journal is not None:
            journal.close()
        if session is not None:
            (directory / "final-snapshot.zip").write_bytes(session.snapshot())
            session.close()
        (directory / "report.json").write_text(
            json.dumps({"trials": trials, "benchmark_score": None}, indent=2)
        )
        record["artifacts"]["report"] = "report.json"
        record["finished_at"] = now()
        save(directory, record)


if __name__ == "__main__":
    main()
