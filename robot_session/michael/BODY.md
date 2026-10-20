# Michael Body / SimBody (v0)

`robot_session.michael.body` is a motor interface for Michael. You write
time-indexed pandas tables. One owner thread runs them on the real
`MichaelOpenArmWuji` MuJoCo/robosuite model. It holds leases, checks guards and
completion conditions at control rate, and reports what was commanded and what
was measured. It does not provide a skill library, task logic or a hardware
backend. A ROS 2 / hardware backend would implement the same `Body` protocol
later; it does not exist yet.

```python
import pandas as pd
from robot_session.michael.body import SimBody, touch, Reached, Status

with SimBody() as body:                    # clock="realtime", control_hz=100, room
    q = body.state().joints.commanded      # pandas Series, rad / m
    table = pd.DataFrame({
        "t": [0.8, 1.6],                   # seconds after start; t=0 is the current command
        "right_arm.joint4": [q["right_arm.joint4"] + 0.3] * 2,
        "right_arm.joint7": [q["right_arm.joint7"] + 0.4, q["right_arm.joint7"]],
    })
    handle = body.run(table)               # returns once the owner thread accepts it
    ...                                    # physics keeps running while you think
    result = handle.result(timeout=5)
    result.status, result.attained, result.target_error, result.log()
```

`python -m robot_session.michael.examples.basic_movements --output DIR` runs
arm, concurrent hand, lift, base and touch-stop movements in real time. It writes
a video, telemetry, events and a report.

## Channels and resources

`body.channels()` lists all 58 native axes by name, built from the live model.
At startup each actuator is checked to drive its named joint.

| Resource | Channels | Mode, unit |
| --- | --- | --- |
| `right_arm`, `left_arm` | `right_arm.joint1` … `joint7` | position, rad |
| `right_hand`, `left_hand` | `right_hand.<finger>.<joint>`, e.g. `right_hand.index_finger.mcp_flex`; fingers `thumb, index_finger, middle_finger, ring_finger, pinky` | position, rad |
| `lift` | `lift.height` | position, m (0–0.34) |
| `base` | `base.vx`, `base.vy` (m/s), `base.wz` (rad/s) | velocity, current base frame |

Native slices and the normalized base action are internal. Base values are
physical: SimBody converts them through the actuator ctrlrange (±1 m/s, ±1.5 rad/s).

`body.signals()` names the measured values that conditions can reference:
- every channel name, which means the measured joint position, or the measured
  base velocity for base channels;
- the ten `<hand>.<finger>.touch_N` pads.

## Movement tables

A table is a finite DataFrame with a `t` column (or an index named `t`), in
seconds after start. All other columns are channel names. Validation happens
before anything is sent, and errors raise `TableError`:

- Every value is real, numeric and finite. Missing or NaN targets are
  rejected, as are boolean and complex columns; omit a column to hold that
  channel.
- `t` is strictly increasing, the first row is `> 0`, and the table lasts at most 120 s.
- The current *commanded* value is the implicit `t=0` knot. Values are
  interpolated linearly at the control rate.
- Position targets must lie strictly within the model's joint/actuator bounds
  (the same bounds the native controller enforces; there is no tolerance) and
  within the rate envelope on every segment. Before each step the executor
  clips commands to those bounds, which only removes interpolation roundoff. The first segment is checked against the
  live command when the owner thread activates the motion.
- Base velocities are limited in magnitude and acceleration, starting from 0.
  **The last row must be zero**, so base motion always expires. An omitted base
  channel is commanded zero. With `base_kinematics="differential"` (the
  default), any nonzero `base.vy` is rejected.
- Omitted channels of a leased resource keep their current command. Resources
  not in the table are not leased and keep holding.

`Envelope` holds the software limits: arm 1.0 rad/s, hand 3.0 rad/s, lift
0.05 m/s, base 0.25 m/s, 0.5 rad/s, 0.5 m/s², 1.0 rad/s². They are simulation
defaults only and were not derived from hardware.

## Execution

- **One owner thread** creates, steps, reads, renders and closes the
  environment. Callers send requests through a queue and read immutable
  snapshots. No public attribute exposes MuJoCo or robosuite objects.
- **Rate.** `control_hz` defaults to 100 and must be an integer divisor of the
  500 Hz physics rate. Physics stays at robosuite's 0.002 s timestep. Each tick
  runs the commands, one `env.step` (5 physics substeps at 100 Hz), the finite
  and MuJoCo-warning checks, measurement, conditions and logging. The existing
  panel and commissioning runner still use 20 Hz.
- **`clock="realtime"`** steps continuously against the wall clock, including
  while the caller sleeps or waits on a model. Idle resources hold their last
  command, and the base is commanded zero whenever no motion owns it. If a
  tick is late, the executor catches up by at most `max_catchup_s` (0.05 s).
  Beyond that, the wall time is dropped: simulated time falls behind and there
  is no burst. `health.late_ticks`, `health.dropped_s` and `overrun` events
  report it.
- **`clock="manual"`** advances only inside `body.advance(seconds)`. It is for
  debugging and deterministic tests. Only one `advance()` may run at a time; a
  second one raises `BodyError`, and `close()` releases a waiting caller with
  `BodyClosed`.
- **Request scheduling.** Control requests (run, cancel, close, probe,
  advance) are served immediately between ticks. Rendering requests
  (`capture`, `record` start) run only while the executor is ahead of the next
  tick, so a flood of captures delays physics by at most one render. At most
  four render requests may be outstanding; further callers block. Periodic
  `record()` frames render inside the tick that is due.
- **Leases.** `run()` takes all of its resources at once or raises
  `ResourceBusy`. Motions on disjoint resources run together. v0 has **no
  preemption or append**; cancel first, then submit.

### How a motion ends

| `status` | Meaning |
| --- | --- |
| `completed` | Every row was commanded (no `until` given). This is **not** a success claim. |
| `condition_met` | `until` held on measurements, possibly mid-table. |
| `guarded` | A guard held; the motion stopped early. |
| `timed_out` | `until` did not hold by `timeout_s` (default: duration + 2 s; the final row is held meanwhile). |
| `cancelled` | `handle.cancel()`, `cancel_all()` or `close()`. |
| `faulted` | Executor exception, non-finite state or MuJoCo warning. Physics stops. |

The `MotionResult` separately reports:
- `rows_commanded` / `rows_total` and `table_time_commanded`: how far the
  command stream actually got;
- `trigger`: condition, signal, value and sim time;
- `final_target` (the last row), `final_command` and `final_measured`, plus
  `tracking_error` and `target_error`;
- `attained`: whether the measured positions ended within the `Reached`
  tolerance of the last row;
- `log()`: per-tick commanded, measured and condition signals as a
  `(kind, name)` MultiIndex;
- `events()`.

### Conditions and stopping

- `Signal(pattern, op, value, for_s=0, mode="any"|"all")` uses fnmatch over
  `body.signals()`. `touch("right_hand", 0.1)` means "any right pad above 0.1 N".
  `Reached(rad=0.05, m=0.01, for_s=0.05)` tests the table's final row against
  the measured positions of its position channels.
- `Reached` as `until` is only evaluated once the final row has been
  commanded, so an out-and-back path is never cut short by starting near its
  end point. `Signal`/`touch` conditions are evaluated from the first tick and
  can stop a table mid-way, which is their purpose.
- `guard=[...]` and `until=...` are evaluated by the owner thread after every
  control tick. No model call is involved. A guard wins over `until` on the
  same tick.
- **Stop behaviour** (guard, `until`, cancel, timeout): table progression stops
  at once. Position channels hold `stop_hold="commanded"` (the default: the
  last command) or `"measured"` (the clipped measured position at the stop).
  Base velocity ramps to zero at the envelope deceleration. The handle
  resolves and releases its leases only after the hold and zero base command
  have actually been sent in a physics step, so `final_command`, the log and
  `state()` commanded values all mean "sent", never "planned". In manual clock
  a measured hold or base ramp needs `advance()`, so use `cancel(wait=False)`.
- Holding a command near contact keeps the servo force present at that
  instant; there is no blind retract and no compliance. In this model, gravity
  pulls the Wuji fingers past their command, so `"measured"` holds a slightly
  *more* flexed target. In the probe test it left about 0.09 N on the pad,
  against about 0.03 N with `"commanded"`. Neither option guarantees lower force.
- No threshold here is claimed safe for hardware. Tactile values come from
  ideal MuJoCo touch sensors on the real pad meshes, with no noise, bandwidth,
  taxels or firmware (see README).

### Shutdown and faults

- `close()` cancels every motion (reason "body closed"; a stop that was
  already decided keeps its status and notes "stop not completed"),
  stops physics and closes the environment.
- A fault resolves every handle as `faulted`, stops physics, and leaves
  `state().health.faulted` and `error` set.
- **What a fault receipt and snapshot mean.** A tick is *confirmed* once
  `env.step` has returned and the finite-state and MuJoCo-warning checks have
  passed. Only then does its command count as sent, and only then is its state
  read as a new measurement sample. A step that raises, or fails those checks,
  may have advanced physics partly or fully, so its state and sensor arrays
  are not trusted and **measurements are not refreshed after a fault**.
  - Measured values in `state()` and in the receipt (`joints.measured`, base,
    tactile, `final_measured`) are the last confirmed sample.
    `BodyState.sim_time` and `MotionResult.measured_time` / `end_time` give
    that sample's sim time.
  - `BodyState.physics_time` is the raw simulator clock (NaN if non-finite).
    After a failed step it can be later than `sim_time`; the difference is how
    far physics moved without being measured.
  - Commanded values and `final_command` are the last confirmed sent command.
    `MotionResult.attempted_command` is the command passed to the failed step,
    whether or not it was applied; it is `None` on receipts that did not end
    in such a step.
- After close or fault, `run()`, `capture()` and the other requests raise
  `BodyClosed` / `BodyFaulted`. Requests already queued are failed, never left
  hanging.
- Stopping physics freezes the simulation. A real robot would need its own
  safe-stop behaviour.

## State, logs and memory

- `body.state()` returns an immutable `BodyState` published after every tick:
  - `joints`: commanded (as last sent to physics), measured, error, velocity,
    bounds, owner;
  - `base`: commanded and measured velocity at the yaw axis, in the base frame;
  - `base_pose`: world-frame yaw-axis position and heading. This is simulator
    ground truth, not odometry;
  - `tactile`: touch, plus pad force and torque including pad gravity and inertia;
  - `health`: timing, warnings, faults.

  Each property builds a fresh DataFrame from read-only copies. Mutating it
  cannot affect the body or later reads.
- `body.telemetry(last_s)` reads a fixed ring (default 10 s at control rate) of
  commanded, measured and touch values. `body.events()` reads the last 2000
  events. Motion logs are preallocated for the motion's maximum duration. Idle
  real-time operation does not grow memory.

## Images

- `body.capture(view, width, height)` renders on the owner thread with
  MuJoCo's `Renderer` and returns `Frame(rgb, sim_time, view)`; the array is
  read-only. Views are `overview`, `right_hand`, `left_hand`, or any MJCF
  camera in `body.views()`.
- `body.record(view, fps, ...)` captures at fixed simulated-time spacing
  (first frame at start). The frame/byte budget is bounded, and
  `truncated` reports when it ran out.
- Rendering costs real time: about 24 ms for 640×480 here. At 10 fps the
  executor kept up. The first render creates the GL context, which takes about
  150 ms and shows up as one dropped interval.
- For the kitchen, SimBody builds the scene with
  `make_environment(..., camera_observations=False)`. robosuite's own
  offscreen context uses GLFW, which macOS only permits on the main thread
  (otherwise the process traps). The kitchen's robot cameras are still
  available to `capture()`.

## Base: what is and is not modelled

- The Omron model is an ideal planar X/Y/yaw velocity abstraction with no
  wheels, slip, balance or tipping. `base_kinematics="differential"` is a
  software rule that rejects lateral commands. It does **not** make the
  physics a differential-drive base.
- **Base frame fix.** robosuite's inherited
  `MobileBaseJointVelocityController` rotated commands from the current to the
  initial frame with R(−θ) instead of R(+θ). Base commands therefore came out
  rotated by −2θ once the base had turned: at 90° "forward" drove backward.
  Michael now uses `models.MichaelBaseVelocityController`, which applies
  R(+θ); gains, geometry and `upstream/` are unchanged. Raw `env.step` native
  commands, the panel, the commissioning runner and SimBody all use it, and
  SimBody adds no compensation of its own. Two tests cover this: a direct
  native-command test and the SimBody heading test (0.400 m along the heading,
  < 1 mm lateral, after a 90° turn).
- Base velocity and `base_pose` are measured at the yaw axis, which is 0.21 m
  behind the `mobilebase0_base` body origin.

## Measured on this machine (Apple silicon, MuJoCo 3.3.1)

- 100 Hz control: room ~2.1 ms per `env.step`; kitchen ~2.5 ms with camera
  observations off (20 ms with them on). The real-time ratio was 1.00 in the
  room and the kitchen, both idle and moving, apart from the reported stalls
  below.
- Steady-state servo error under gravity in the commissioning pose: arms up to
  ~0.045 rad, hands up to ~0.062 rad, lift ~0.025 m. Choose `Reached`
  tolerances with this in mind. `completed` with `attained=False` is a normal
  outcome.
- The owner thread shares the Python GIL with the caller. Long GIL-holding
  work in the caller can delay ticks. One 75 ms stall in the demo coincided
  with the caller exporting telemetry, but the cause was not confirmed. The delay is reported and never bursted. A
  separate process would be the next step if this matters.

## Simulation-only fixture

- `SimBody.sim_place_probe(hand, finger, gap_m)` and `sim_remove_probe()` move
  the commissioning room's stationary 8 mm calibration sphere next to a pad.
- They are for tests and demos only and are not part of the `Body` protocol. No
  other MuJoCo mutation is exposed.

## Not in v0

- Preempt/append blending.
- Cartesian or IK targets.
- Base path segments with odometry.
- Arm/base interlocks.
- Compliance or retract reflexes.
- Hardware/ROS 2 backend.
- Parquet run archive.
- Multi-process isolation.

## Validation

```sh
.venv/bin/python -m pytest tests/test_michael_body.py -q
```
