"""Interactive geometry-informed physical control; never a candidate library.

JSON commands on stdin select bounded movement tests. Every command is written
and executed in the isolated code sandbox. No joint-state writes, task demos or
privileged simulator calls occur in submitted code. Geometry targets are supplied
by the human/test coordinator, so this is explicitly an oracle control.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from roboswe.execution.sandbox import CodeSandbox
from roboswe.library.attempts import AttemptRunner, Journal, replay_journal
from roboswe.library.simulator import SimulatorProcess
from roboswe.library.spec import GOALS, library_set

SOURCE = """import json, math
import numpy as np
from scipy.spatial.transform import Rotation
import robot

def state(o):
    s=o['sensors']
    return {'step':o['control_step'],'success':o.get('task_success'),
      'base':s['robot0_base_pos'],'eef':s['robot0_eef_pos'],
      'quat':s['robot0_eef_quat_site'],'gripper':s['robot0_gripper_qpos'],
      'joints':s['robot0_joint_pos']}

def tick(dp=(0,0,0),dr=(0,0,0),base=(0,0,0),grip=-1,mode=0):
    return robot.step([*map(float,dp),*map(float,dr),float(grip),*map(float,base),0.,float(mode)])

def run(command):
    o=robot.observe()
    kind=command['kind']
    for i in range(min(command.get('ticks',50),50)):
        s=o['sensors']
        if kind=='base':
            B=Rotation.from_quat(s['robot0_base_quat']).as_matrix()
            error=np.array(command['target'])-np.array(s['robot0_base_pos'])[:2]
            if np.linalg.norm(error)<.018: break
            local=B[:2,:2].T@error
            v=[float(np.sign(e)*(.26+min(.14,1.5*abs(e)))) if abs(e)>.01 else 0. for e in local]
            o=tick(base=[*v,0],grip=command.get('grip',-1),mode=1)
        elif kind=='arm':
            B=np.array(o['action_schema']['controllers'][0]['parameters']['right']['base_frame_rotation_world'])
            R=Rotation.from_quat(s['robot0_eef_quat_site']).as_matrix()
            error=np.array(command['target'])-np.array(s['robot0_eef_pos'])
            rotation=np.array(command['rotation'])
            er=Rotation.from_matrix(B.T@rotation@R.T@B).as_rotvec()
            base_command=[0.,0.,0.]
            base_distance=0.
            if 'base_target' in command:
                base_error=np.array(command['base_target'])-np.array(s['robot0_base_pos'])[:2]
                base_distance=float(np.linalg.norm(base_error))
                base_B=Rotation.from_quat(s['robot0_base_quat']).as_matrix()
                local=base_B[:2,:2].T@base_error
                base_command=[float(np.sign(e)*.28) if abs(e)>.008 else 0. for e in local]+[0.]
            if np.linalg.norm(error)<command.get('tolerance',.007) and np.linalg.norm(er)<.07 and base_distance<.018: break
            o=tick(np.clip(B.T@error/.05,-.18,.18),np.clip(er/.5,-.14,.14),base=base_command,grip=command.get('grip',-1))
        elif kind=='grip':
            o=tick(grip=command['grip'])
        else: raise ValueError('unknown motion command')
    result=state(o)
    s=o['sensors']
    result['grasp_width']=abs(s['robot0_gripper_qpos'][0]-s['robot0_gripper_qpos'][1])
    if kind in ('arm','base'):
        actual=np.array(s['robot0_eef_pos'] if kind=='arm' else s['robot0_base_pos'][:2])
        result['distance_to_target']=float(np.linalg.norm(np.array(command['target'])-actual))
        result['converged']=result['distance_to_target']<command.get('tolerance',.018 if kind=='base' else .007)
        if kind=='arm':
            current=Rotation.from_quat(s['robot0_eef_quat_site']).as_matrix()
            result['rotation_error']=float(np.linalg.norm(Rotation.from_matrix(np.array(command['rotation'])@current.T).as_rotvec()))
            result['converged']=result['converged'] and result['rotation_error']<.07
        if 'base_target' in command:
            result['base_distance']=float(np.linalg.norm(np.array(command['base_target'])-np.array(s['robot0_base_pos'])[:2]))
            result['converged']=result['converged'] and result['base_distance']<.018
    return result
"""


def decision(operation, path="", content=""):
    return {
        "operation": operation,
        "path": path,
        "content": content,
        "message": "Geometry-informed positive control, not a candidate attempt",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--simulator-image", required=True)
    parser.add_argument("--code-image", required=True)
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    spec = library_set()
    spec.update(
        id="geometry-informed-oven-control",
        initial_robot_base_ref="oven_main_group",
        task_timeout_seconds=1800,
    )
    spec["tasks"] = [
        {"index": 0, "round": 1, "pair_key": "OpenOven", **asdict(GOALS["OpenOven"])}
    ]
    spec.pop("sha256")
    spec["sha256"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    config = {
        "robot": "PandaOmron",
        "task": "robocasa/OpenOven",
        "split": None,
        "seed": 0,
        "env_kwargs": {
            "layout_ids": 2,
            "style_ids": 1,
            "camera_widths": 128,
            "camera_heights": 128,
        },
    }
    (directory / "control-provenance.json").write_text(
        json.dumps(
            {
                "kind": "geometry_informed_test_oracle_not_candidate",
                "config": config,
                "spec": spec,
            },
            indent=2,
        )
    )
    kwargs = {
        "image": args.simulator_image,
        "timeout": 1800,
        "max_steps": 4000,
        "lifetime_seconds": 2400,
    }
    journal = Journal(directory / "submissions.jsonl")
    complete = False
    try:
        with (
            SimulatorProcess(config, spec, directory / "original", **kwargs) as runtime,
            CodeSandbox(args.code_image) as sandbox,
        ):
            runtime.record_start("control", width=640, height=480)
            runner = AttemptRunner(runtime, sandbox, journal)
            info = runner.start_task(0)
            runner.apply(decision("write", "reference.py", SOURCE))
            print(
                "READY "
                + json.dumps(
                    {
                        "sensors": info["observation"]["sensors"],
                        "action_schema": info["observation"]["action_schema"],
                    }
                ),
                flush=True,
            )
            try:
                for line in sys.stdin:
                    command = json.loads(line)
                    if command["kind"] == "finish":
                        result = runner.finish_task(
                            reason="geometry_informed_control_complete"
                        )
                        complete = True
                        (directory / "observed.json").write_text(
                            json.dumps(result, indent=2)
                        )
                        print("OBSERVED " + json.dumps(result), flush=True)
                        break
                    source = (
                        "from reference import run\nimport json\nprint(json.dumps(run("
                        + repr(command)
                        + ")))\n"
                    )
                    runner.apply(decision("write", "stage.py", source))
                    result = runner.apply(
                        decision("execute", "stage.py"),
                        timeout=90,
                        max_control_steps=50,
                    )
                    print("RESULT " + json.dumps(result), flush=True)
                    feedback = (
                        json.loads(result["output"])
                        if result["returncode"] == 0 and not result["error"]
                        else {}
                    )
                    if feedback.get("success"):
                        result = runner.finish_task(reason="sustained_success")
                        (directory / "observed.json").write_text(
                            json.dumps(result, indent=2)
                        )
                        complete = True
                        print("OBSERVED " + json.dumps(result), flush=True)
                        break
                    if (
                        command.get("require_convergence")
                        and not feedback.get("converged")
                    ) or feedback.get("grasp_width", 0) < command.get(
                        "minimum_grip_width", 0
                    ):
                        print(
                            "CONTROL_ABORT: requested motion/grasp condition not reached",
                            flush=True,
                        )
                        break
                if runner.task_active:
                    result = runner.finish_task(reason="control_stopped")
                    (directory / "observed.json").write_text(
                        json.dumps(result, indent=2)
                    )
                    complete = True
            finally:
                runtime.record_stop("control")
                sandbox.archive(directory / "library.zip")
    finally:
        journal.close()
    if complete:
        with (
            SimulatorProcess(
                config, spec, directory / "verification", role="verifier", **kwargs
            ) as runtime,
            CodeSandbox(args.code_image) as sandbox,
        ):
            runtime.record_start("verification", width=640, height=480)
            report = replay_journal(
                directory / "submissions.jsonl",
                runtime,
                sandbox,
                directory / "verification.jsonl",
            )
            runtime.record_stop("verification")
        report["kind"] = "geometry_informed_positive_control_not_candidate"
        (directory / "report.json").write_text(json.dumps(report, indent=2))
        print("VERIFIED " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
