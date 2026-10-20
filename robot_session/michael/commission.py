"""Reproducible basic-motion and fingertip-contact commissioning; no tasks."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from .build import FINGERS
from .runtime import Camera, describe, make_environment, tactile


class Recorder:
    def __init__(self, env, destination):
        self.camera = Camera(env)
        self.container = av.open(str(destination), "w")
        self.stream = self.container.add_stream("libx264", rate=20)
        self.stream.width, self.stream.height = 960, 720
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "21", "preset": "fast"}
        self.env = env

    def frame(self, label):
        image = Image.fromarray(self.camera.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 46), fill="#10191f")
        draw.text((18, 16), "MICHAEL  /  " + label, fill="#e6eef0")
        values = tactile(self.env.robots[0])
        draw.rectangle((0, 656, 960, 720), fill="#10191f")
        for i, side in enumerate(("right", "left")):
            text = (
                side.upper()
                + "  "
                + "  ".join(
                    f"{f.replace('_finger', '')}: {values[side][f]['normal_N']:.2f} N"
                    for f in FINGERS
                )
            )
            draw.text((18, 670 + 22 * i), text, fill="#97deba")
        frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        return image

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.camera.close()


def run(output, video=True):
    output.mkdir(parents=True, exist_ok=True)
    env = make_environment()
    robot = env.robots[0]
    model, data = env.sim.model._model, env.sim.data._data
    recorder = Recorder(env, output / "basic-movements.mp4") if video else None
    report = {
        "robot": "MichaelOpenArmWuji",
        "task_attempted": False,
        "motion": [],
        "tactile": [],
    }
    trajectory = []
    action = robot.create_action_vector({})
    home = action.copy()
    slices = robot.composite_controller._action_split_indexes

    def step(label, command):
        env.step(command)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise AssertionError("Nonfinite physics")
        if any(w.number for w in data.warning):
            raise AssertionError("MuJoCo physics warning")
        trajectory.append(
            {
                "time": float(data.time),
                "phase": label,
                "action": command.tolist(),
                "qpos": data.qpos.tolist(),
                "tactile": tactile(robot),
            }
        )
        if recorder:
            return recorder.frame(label)

    def move(label, updates, seconds=1.0):
        nonlocal action
        before = data.qpos.copy()
        target = action.copy()
        target[slice(*slices["base"])] = 0
        for part, values in updates.items():
            target[slice(*slices[part])] = values
        for alpha in np.linspace(0, 1, max(2, int(seconds * 20))):
            smooth = alpha * alpha * (3 - 2 * alpha)
            step(label, action + smooth * (target - action))
        action = target
        report["motion"].append(
            {
                "name": label,
                "max_joint_displacement": float(abs(data.qpos - before).max()),
            }
        )

    try:
        for _ in range(30):
            still = step("Hold / no contact", action)
        if recorder:
            still.save(output / "overview.png")
        assert model.nq == model.nu == 58
        assert model.nsensor == 34
        assert max(abs(data.qvel)) < 0.02
        assert all(
            v["normal_N"] == 0
            for side in tactile(robot).values()
            for v in side.values()
        )
        (output / "control-schema.json").write_text(
            json.dumps(describe(env), indent=2) + "\n"
        )

        for side in ("right", "left"):
            arm = home[slice(*slices[side])].copy()
            arm[3] += 0.20
            arm[6] += 0.30 if side == "right" else -0.30
            move(f"{side.title()} elbow and wrist", {side: arm}, 1.5)
            for _ in range(10):
                step(f"{side.title()} arm hold", action)
            move(f"{side.title()} arm return", {side: home[slice(*slices[side])]}, 1.0)
        for side in ("right", "left"):
            hand = np.tile([0.3, 0.08, 0.25, 0.15], 5)
            hand[:4] = [0.15, -0.15, 0.20, 0.15]
            move(
                f"{side.title()} fingers flex / spread", {f"{side}_gripper": hand}, 1.5
            )
            move(f"{side.title()} fingers open", {f"{side}_gripper": np.zeros(20)}, 1.5)
        base_before = data.qpos[robot._ref_base_joint_pos_indexes].copy()
        move("Base forward and turn", {"base": [0.08, 0, 0.10]}, 1.0)
        for _ in range(10):
            step("Base forward and turn", action)
        move("Base stop", {}, 0.5)
        base_after = data.qpos[robot._ref_base_joint_pos_indexes].copy()
        report["base_displacement"] = (base_after - base_before).tolist()
        assert abs(base_after[0] - base_before[0]) > 0.025
        assert abs(base_after[2] - base_before[2]) > 0.03
        move("Lift 3 centimetres", {"torso": [0.13]}, 1.0)
        move("Lift return", {"torso": [0.1]}, 1.0)

        # Move a calibration fixture onto each real pad in turn, retaining robot
        # dynamics. Log actual collision geom names, off-channel zeros and release.
        mocap = model.body("tactile_probe").mocapid[0]
        for side in ("right", "left"):
            for index, finger in enumerate(FINGERS):
                site = model.site(
                    robot.gripper[side].correct_naming(f"{finger}_tactile_site")
                ).id
                if recorder:
                    recorder.camera.set_view(side)
                for depth in np.linspace(0.004, -0.006, 15):
                    outward = 1 if side == "right" else -1
                    data.mocap_pos[mocap] = data.site_xpos[site] + data.site_xmat[
                        site
                    ].reshape(3, 3) @ np.array(
                        [0, outward * (model.site_size[site, 1] + 0.008 + depth), 0]
                    )
                    still = step(
                        f"Touch test / {side} {finger.replace('_finger', '')}", action
                    )
                values = tactile(robot)
                force = values[side][finger]["normal_N"]
                others = [
                    v["normal_N"]
                    for s, fingers in values.items()
                    for f, v in fingers.items()
                    if (s, f) != (side, finger)
                ]
                pairs = [
                    (model.geom(c.geom1).name, model.geom(c.geom2).name)
                    for c in data.contact
                ]
                expected = robot.gripper[side].correct_naming(f"{finger}_pad_collision")
                assert any(
                    expected in pair and "tactile_probe_geom" in pair for pair in pairs
                ), (side, finger, pairs)
                assert force > 0.01, (side, finger, force)
                assert max(others) < 1e-7, (side, finger, others)
                if recorder and side == "right" and finger == "index_finger":
                    still.save(output / "fingertip-contact.png")
                data.mocap_pos[mocap] = [2, 0, 1]
                for _ in range(10):
                    step("Release / zero contact", action)
                released = tactile(robot)[side][finger]["normal_N"]
                assert released == 0.0
                report["tactile"].append(
                    {
                        "side": side,
                        "finger": finger,
                        "normal_N": force,
                        "other_fingertips_max_N": max(others),
                        "released_N": released,
                        "collision_geom": expected,
                    }
                )
                print(
                    f"PASS touch {side}/{finger}: {force:.3f} N; other tips 0; released 0",
                    flush=True,
                )

        # Active touch: the fixture stays still while the right index finger
        # closes. Stop increasing the target as soon as tactile feedback fires.
        site = model.site(
            robot.gripper["right"].correct_naming("index_finger_tactile_site")
        ).id
        data.mocap_pos[mocap] = data.site_xpos[site] + data.site_xmat[site].reshape(
            3, 3
        ) @ np.array([0, model.site_size[site, 1] + 0.008, 0])
        if recorder:
            recorder.camera.set_view("right")
        initial_probe = data.mocap_pos[mocap].copy()
        before_hand = action[7:27].copy()
        active_peak = 0.0
        for i in range(40):
            action[7 + 4] = 0.006 * (i + 1)
            action[7 + 6] = 0.004 * (i + 1)
            step("Index finger / close until touch", action)
            active_peak = max(
                active_peak, tactile(robot)["right"]["index_finger"]["normal_N"]
            )
            if active_peak > 0.10:
                break
        assert active_peak > 0.10, active_peak
        assert np.array_equal(data.mocap_pos[mocap], initial_probe)
        for _ in range(12):
            step("Contact detected / hold target", action)
        move("Index finger / withdraw", {"right_gripper": before_hand}, 1.0)
        data.mocap_pos[mocap] = [2, 0, 1]
        for _ in range(10):
            step("No contact", action)
        report["active_touch"] = {
            "peak_normal_N": active_peak,
            "fixture_stationary": True,
            "released_N": tactile(robot)["right"]["index_finger"]["normal_N"],
        }
        assert report["active_touch"]["released_N"] == 0
        report["max_settled_velocity"] = float(max(abs(data.qvel)))
        report["passed"] = True
        report["physics_warnings"] = [int(w.number) for w in data.warning]
        (output / "model.xml").write_text(env.sim.model.get_xml())
        np.savez_compressed(
            output / "final-state.npz", qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl
        )
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "trajectory.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in trajectory)
        )
        if recorder:
            recorder.close()
        env.close()
    print(f"PASS commissioning: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/michael-commissioning")
    )
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run(args.output, video=not args.no_video)
