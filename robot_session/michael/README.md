# Michael: OpenArm 2.0 + Wuji Hand 2

An opt-in robot integration for the existing robosuite / RoboCasa runtime. It is
registered as `MichaelOpenArmWuji`. This package supplies an empty commissioning
room, an empty RoboCasa kitchen, a paused local control panel, and repeatable
motion/contact checks. It does not start or solve any benchmark task.

For program-level control from Python (pandas movement tables, a real-time
owner-thread executor, tactile guards and measured receipts), see
[BODY.md](BODY.md).

## Open the controls

From the repository root, with the existing simulation environment:

```sh
.venv/bin/python -m robot_session.michael.serve --port 8873
```

Open <http://127.0.0.1:8873>. The process listens only on loopback. The same
process owns physics, rendering, and command handling. It advances physics only
for bounded movement commands; looking at the scene or sensor values does not
advance time. **Pause physics / Escape** freezes the simulation immediately.
The panel is a simulation tool and has no hardware connection.

The scene selector switches between the commissioning room and RoboCasa kitchen.
Switching creates a fresh scene. The panel offers each arm's seven joint targets,
each hand's twenty joint targets, gentle presets, base pulses, lift control,
camera orbit/zoom, and all ten fingertip readings. Targets are displayed in
degrees; measured joint positions appear alongside them. Native API commands use
radians. The manufacturer servos can have nonzero gravity-induced tracking error.

The kitchen panel is a human cutaway view: walls and ceilings are hidden only in
the spectator renderer. Collision geometry and the robot's own camera images are
unchanged.

**Feel probe** puts a stationary calibration sphere just ahead of the selected
index pad and slowly closes that finger. A contact reading above 0.10 N stops
further movement and pauses physics so the reading can be inspected. **Remove
probe** removes the fixture and withdraws the finger. This calibration fixture is
available in the commissioning room. It is not a task object or a robot sensor.

## Exact sources and local changes

`sources.json` pins the manufacturer revisions and lists SHA-256 checksums for
every downloaded source file. `vendor/` preserves their XML and mesh files and
their licenses. `build.py` produces separate adapted XML under `generated/`;
the downloaded files and the checkouts under `upstream/` are not modified.

| Component | Source |
| --- | --- |
| OpenArm v2 bimanual, 7 joints per side | [Enactic openarm_mujoco](https://github.com/enactic/openarm_mujoco/tree/ee1110561fa64efe041b90c7b274b93825c9bd82/v2) |
| Wuji Hand 2 Beta 2 with mount, 20 joints per side | [Wuji description](https://github.com/wuji-technology/wuji-description/tree/c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95/hand2/hand2_beta2) |
| Omron mobile base and lift | Installed robosuite model, composed locally |

The OpenArm stock fingers are removed. Its seven arm joints, wrist housing,
inertias, position-actuator gains and force limits are retained. The Wuji
with-mount hand models preserve their anatomy, limits, inertias and position
actuators. The arm controller sends joint positions to those position actuators;
it does not send torque commands into position-control inputs.

The 40 cm shoulder riser and wrist stand-offs are provisional simulation
fixtures, not validated physical mount designs. The retained OpenArm wrist
housing is not a CAD-validated replacement flange. Hardware clearance, mounting
bolts, load capacity, wiring, and the eventual real mobile base remain open.

The Omron model is an ideal planar X/Y/yaw base with a 0–0.34 m lift. It does not
model wheel slip, balance, or tipping. This integration removes its artificial
joint stiction for slow commissioning motion and gives the lift a limited
position servo. It is not evidence of real Omron holonomic motion or suitability
for the physical robot.

## Native control surface

58 scalar values, at 20 Hz. Slices are Python `[start:end]` indices:

| Slice | Part | Meaning |
| --- | --- | --- |
| `0:7` | `right` | Seven absolute OpenArm joint angles, radians |
| `7:27` | `right_gripper` | Twenty absolute Wuji joint angles, radians |
| `27:34` | `left` | Seven absolute OpenArm joint angles, radians |
| `34:54` | `left_gripper` | Twenty absolute Wuji joint angles, radians |
| `54:57` | `base` | Normalized X/Y/yaw velocity in the current base frame |
| `57:58` | `torso` | Absolute lift extension, metres |

For the inherited base controller, X/Y command 1 means a 1 m/s velocity target;
yaw command 1 means 1.5 rad/s. The panel uses much smaller pulses. Base commands
return to zero after movement. Each hand has four independent joints per finger,
ordered thumb, index, middle, ring, pinky. Full joint names and limits are
discovered from the live model and returned by `/api/state` or the session action
schema. An all-zero vector is **not a hold pose**.

```python
from robot_session.michael.runtime import make_environment

env = make_environment()  # kitchen=True for RoboCasa
robot = env.robots[0]
hold = robot.create_action_vector({})
right = hold[:7].copy()
right[6] += 0.10
action = robot.create_action_vector({"right": right})
observation, reward, done, info = env.step(action)
print(observation["robot0_right_fingertip_touch"])
env.close()
```

`create_action_vector` holds omitted arm/hand/lift targets and stops the base.
For a persistent `RobotSession`, use this directory's `config.json`; the existing
environment adapter registers Michael automatically. `MichaelCommissioningKitchen`
also corrects the kitchen reset's zero-action settling phase to retain the
commissioning pose. Other task-specific reset paths have not been commissioned.

## What the fingertip feedback means

The manufacturer Beta 2 model has one dedicated pad body and collision mesh per
finger. Its XML does not define sensor outputs. This integration adds an ideal
MuJoCo `touch`, `force`, and `torque` sensor to **each actual pad body**. The touch
volume comes from that pad mesh's bounding box with a 0.3 mm margin.

- `robot0_{left,right}_fingertip_touch`: five normal contact forces, N.
- `robot0_{left,right}_fingertip_force`: fifteen values, five local XYZ forces, N.
- `robot0_{left,right}_fingertip_torque`: fifteen values, five local XYZ torques, N·m.

The force/torque channels include pad inertia and gravity; an unloaded force
channel is not necessarily zero. Touch measures normal contact force and is
zero when the pad is not touching. The same values and their units/frame metadata
are exposed through the persistent session's additional robot sensors.

This is simulated contact at the real pad locations. It does not emulate Wuji
firmware, calibrated hardware noise, bandwidth, taxel layout, or soft-skin
deformation. See the [MuJoCo sensor reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-touch).

## Repeat the verified movements

```sh
.venv/bin/python -m robot_session.michael.commission \
  --output artifacts/michael-commissioning/new-run
```

This saves an MP4, overview/contact stills, per-step commands and measured state,
a control schema, assembled MJCF, final physics arrays, and a JSON report. It
asserts stable holding; 58 axes; all ten isolated pad contacts with zero readings
on other fingertips and after release; real contact geom identities; a stationary
probe contacted by an actuated finger; nonzero base travel; and no physics
warnings. Arms, fingers, base yaw and lift also move in the recorded sequence.
It does not make a RoboSWE task-success claim.

To restore missing source assets, run
`.venv/bin/python -m robot_session.michael.fetch_assets`. About 21 MB of pinned
manufacturer assets are included. The normal robosuite/RoboCasa dependencies and
kitchen assets are still required. No cloud compute is needed for these checks.
