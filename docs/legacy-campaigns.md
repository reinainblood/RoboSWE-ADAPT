# Historical RoboSWE campaign reference

This document preserves the earlier README for the legacy session, serial, and
continuous-library protocols. Its implementation-status statements describe that
earlier development snapshot, not the current RoboSWE-ADAPT runner. For new
five-trial work, use the [adaptive CLI guide](adaptive-cli.md). Historical results
and context-compaction policies remain separate from the new release.

---

# RoboSWE-365

Kirsten's robot software benchmark project. The current repository contains
persistent RoboCasa session infrastructure and host-side job inspection tools.
A long-horizon library protocol, numeric task verifiers, isolated code execution,
and independent code replay are now implemented as components. The reference campaign command is experimental. Task-set calibration is still
in progress; this is not yet a validated benchmark release.

## Adaptive release development

The current user-selected scope is recorded in
[datasets/adaptive-release/selection.json](../datasets/adaptive-release/selection.json):
83 selections / 85 native task names, including two proposed merges. This is a
selection manifest, not a scene-admitted runnable dataset. Historical pilots are
excluded from the new leaderboard. The selection rationale and unresolved
metadata checks are preserved in the manifest.

See [the experimental public adaptive CLI](../docs/adaptive-cli.md),
[SQLite context-policy readiness](../docs/sqlite-context-policy.md), and
[the evidence dashboard](../docs/review-dashboard.md). Docker/Modal dispatch and
independent replay have offline tests; live backend qualification is still
required. OpenRouter adaptive execution currently refuses to launch without a
verified pre-request token counter. No public leaderboard or complete new-model
campaign is claimed by this source release.

## Explore the current work

- [Persistent robot sessions](../robot_session/README.md): lifecycle and simulator interfaces.
- [Modal trial infrastructure](../docs/modal-trials.md): remote simulators, isolation, and verified artifact backups. These research tools live under `pairing/`, separately from the reference campaign CLI below.
- [Michael robot integration](../robot_session/michael/README.md): OpenArm/Wuji simulation, local controls, and source provenance.
- [Programmatic body control](../robot_session/michael/BODY.md): movement tables, measured receipts, and tactile guards.
- [Experimental Glimpse visual audit](../docs/glimpse_judge.md): an optional judge that does not change mechanical scores; live provider validation and calibration remain separate work.

Run the local software checks with `python -m pytest tests robot_session/tests -q`
in the configured development environment. Simulator-dependent checks require the
upstream dependencies described in the session documentation. Local run outputs,
videos, credentials, and machine-specific agent configuration are excluded from
Git; clone the source and supply your own saved jobs to use the results viewer.

## Install the host tools

From this repository, with Python 3.11 or newer:

```sh
uv tool install --editable '.[modal]'
```

Or install into your own activated environment with `pip install -e '.[modal]'`.
Omit `[modal]` if you only need Docker or to browse saved results.
The viewer uses Python's standard library and bundled Three.js; it does not need
marimo, a model API, a live simulator, or internet access to view saved jobs.

## CLI status

| Command | Current behavior |
| --- | --- |
| `roboswe jobs` | Working local HTML jobs viewer |
| `roboswe datasets` | Lists supported dataset references |
| `roboswe tasks` | Reads task names/subsets from an official local RoboCasa checkout |
| `roboswe run --dry-run ...` | Validates and emits a reproducible campaign plan |
| `roboswe run ...` | Experimental Docker campaigns with local Codex login, isolated code, videos and independent replay |
| `roboswe execute RECEIPT SOURCE.py` | Executes one script in an existing session, collecting before/after scenes and logs |
| `roboswe capture RECEIPT` | Captures current visual state without executing robot actions |
| `roboswe import-scene FILE` | Adds an existing snapshot to the viewer |
| `roboswe session ...` | Delegates to the existing session lifecycle CLI |
| `roboswe evaluate JOB` | Re-executes completed campaign code in fresh simulator/code containers; creates separate verification records |
| `roboswe resume JOB` | **Stub:** exit 2; no job state is modified |

`run` is reserved for dataset campaigns. The earlier local single-script command
is now named `execute`. `run` now supports sequential Docker campaigns. Modal campaign execution, parallel
scheduling and campaign resume remain unimplemented. The components and their
validation status are described below. No successful dry run
should be interpreted as a tested model/robot combination or a scored campaign.

## Reference a dataset or subset

The current dataset ID is `robocasa365` (`robocasa-365` and `robocasa` are aliases).
`--dataset-root` points directly to the official RoboCasa **source checkout**,
where `robocasa/utils/dataset_registry.py` defines the task sets. It does not point
to a downloaded demonstrations directory. The CLI reads that file without
importing or executing RoboCasa, so listing/planning does not load MuJoCo or a
robot. It does not download assets or demonstrations automatically.

```sh
roboswe datasets
roboswe tasks --dataset robocasa365 --dataset-root upstream/robocasa --list-subsets
roboswe tasks --dataset-root upstream/robocasa --subset target50
roboswe tasks --subset atomic --json
```

Official subset names are read from `TASK_SET_REGISTRY`; aliases `all`, `atomic`,
and `composite` map to the corresponding official sets. Counts follow the checked
out registry rather than the “365” product name. `--split` separately selects
scene/object partitions; it does not implicitly choose a task subset.

```sh
roboswe run --mode serial \
  --dataset robocasa365 \
  --dataset-root upstream/robocasa \
  --subset target50 \
  --robot PandaOmron \
  --agent roboswe \
  --model YOUR_MODEL_ID \
  --backend modal \
  --image im-YOUR_IMAGE_ID \
  --split target \
  --seeds 0 100 \
  --episodes 3 \
  --timeout 600 \
  --max-steps 1000 \
  --concurrency 4 \
  --cpu 4 --memory-mb 16384 \
  --camera-width 640 --camera-height 480 \
  --jobs-dir jobs \
  --dry-run --plan-out plans/target50.json
```

That example resolves 50 tasks × 6 episode seeds = 300 planned trials. Episode
seeds are expanded explicitly: `0, 1, 2, 100, 101, 102`. Overlapping ranges are
rejected. The plan records exact task IDs, registry SHA-256, checkout commit,
robot, agent/model settings, budgets, output directory, and unsupported features. A plan never contains environment-variable credential values; use
`--api-key-env MODEL_API_KEY` and optionally `--api-base-url https://example.com/v1`
to reserve provider configuration in a plan. The reference harness is `roboswe`;
arbitrary participant harnesses are not accepted. The implemented model client uses
the existing local Codex ChatGPT login; it does not require or copy an API key.
Model ID defaults to the locally configured model and the actual resolved ID is
recorded. `--reasoning-effort medium` is the explicit default; global settings are
not modified. API options and Modal remain plan-only.

To select tasks directly instead of a named subset:

```sh
roboswe run --mode serial --dataset-root upstream/robocasa \
  --tasks OpenOven,CloseOven --robot PandaOmron --agent roboswe --dry-run

roboswe run --mode serial --dataset-root upstream/robocasa \
  --task-file my-tasks.json --robot PandaOmron --agent roboswe --dry-run
```

A task file is a JSON array of task names or a text file with one name per line;
`robocasa/` prefixes are accepted. Explicit tasks must exist in the catalog and,
when a subset is also supplied, belong to that subset. Invalid task names fail
instead of silently shrinking the run. `--plan-out` requires `--dry-run` and will
not overwrite an existing plan.

## Long-horizon library mode (default)

```sh
roboswe run --robot PandaOmron --dry-run --plan-out plans/kitchen-library.json
```

The versioned `kitchen-library-v1` sequence repeats eight mixed fixture goals twice:
open microwave, open drawer, faucet on, open oven, close microwave, close drawer,
faucet off, close oven. It uses layout 2, style 1, the same PandaOmron throughout,
a 12-minute budget per task, and a four-hour lifetime for the retained-context sequence.
Revision 2 starts at the microwave using the supported reference-fixture placement;
all later tasks continue from the actual state left by the preceding task. Workspace, model conversation,
and physical state carry over; the agent receives no movement library. This is a
new continuous RoboSWE sequence using RoboCasa task goals, not sixteen official
independent episodes. Use `--mode serial` for arbitrary dataset/subset selections.

To execute, build the simulator and code images, then pass their immutable IDs:

```sh
roboswe session build docker
python -c 'from roboswe.execution.sandbox import build_image; print(build_image())'

roboswe run --robot PandaOmron \
  --image sha256:SIMULATOR_IMAGE_ID --code-image sha256:CODE_IMAGE_ID \
  --reasoning-effort medium --video-judge

roboswe run --mode serial --subset target50 --robot PandaOmron \
  --image sha256:SIMULATOR_IMAGE_ID --code-image sha256:CODE_IMAGE_ID

roboswe jobs
```

Library mode additionally runs matched-state cold controls by default. For each
second-round goal it reruns the original prefix, discards its code workspace, and
starts a fresh participant at the exact verified physical state. The cold code is
then independently replayed too. This adds substantial verification time beyond
the retained-context sequence. `--no-learning-controls` skips these comparisons
and produces descriptive changes with no graded library score.

The simulator image includes the downloaded assets and can be large. Execution
currently requires Docker and `--concurrency 1`. A host lease and live-container
check prevent overlapping reference campaigns. The simulator defaults to 6 GiB;
the separate code/workspace containers add 1.5 GiB to the capacity check. Shared
Docker VM settings are never changed automatically. `--video-judge` additionally needs
`pip install -e '.[judge]'`. Each task gets a saved job, videos, numeric report, and
a `library.zip` of the Python sources actually present at its boundary. The original
recordings and submission history remain in the campaign directory; the task's
default video shows independent replay. New benchmark task records also offer an
**Original attempt / Independent replay** selector, preserving each recording's
own clock and downloadable files. Older records retain their existing single
recording unless their archived footage is explicitly attached. Timed-out model turns are interrupted while
retaining the library conversation for the next task. Simulator lifetimes and
attempt watchdogs also run inside their containers. Each submission is limited to
50 control ticks and 60 seconds by default (`--max-attempt-steps`,
`--attempt-timeout`), forcing another feedback turn before an extended motion
program can continue. Execution-error rates use recorded attempt counts.

The reference client uses a fixed 272,000-token context setting and requests
automatic compaction at 120,000 tokens, independently of global Codex settings.
It uses Responses over HTTP/SSE with the existing ChatGPT login. The process-local
`roboswe_http` provider disables WebSockets after repeated failures in long
multimodal histories; it sets no API key or custom endpoint and does not edit
global Codex configuration. The transport is recorded in the comparison policy.
It keeps one conversation across the library sequence, subject to Codex's history
compaction; it does not promise verbatim retention of every earlier frame.
Each task starts with a user instruction. Subsequent sensor observations and
execution results are submitted as tool output, rather than additional user
instructions. `model-events.jsonl` records token usage, compaction lifecycles,
turn status and errors without copying model text. Cold participants use the same
context policy with fresh conversations, and policy mismatches invalidate their
comparison. A provider turn rejected with the specific `Bad Request` envelope
gets one retry within the original task deadline. Recovery first compacts the
same thread. If the provider rejects that compaction too, the client forks at
the last successful model turn, compacts that retained history, and retries the
latest feedback. Every applied operation came from a successful turn, so this
excludes only later model turns whose proposals were never applied; it does not
reset the robot or discard the code library. The parent thread and checkpoint
turn are recorded in `model-events.jsonl`. At most one checkpoint fork and one
inference retry are allowed, and all recovery uses the original task budget.
Unsuccessful recovery stops the campaign and withholds its score. Other
model/backend errors stop immediately.
Participant threads are saved in the existing local Codex account so model
history survives an app-server process exit. This is model-history durability;
it does not implement resumption of a stopped robot campaign.

Long recordings require host disk space as well as container memory. Launch checks
require 10 GiB free on both the evidence volume and the host temporary volume by
default (`--min-free-disk-gb`, minimum 1 GiB). During execution, replay, and visual
audits, a 1 GiB reserve is checked between operations so the run can stop while
its evidence is still writable. These checks do not reserve disk blocks.
An unavailable Docker engine or failed attempt-container removal is an
infrastructure error: it is journaled, stops the campaign, and is not counted as
an agent coding error. Secondary cleanup failures preserve the original exception.
Private simulator and sandbox containers use `--log-driver none`: Docker must not
duplicate multi-megabyte sensor RPC responses or unbounded submitted-program
output into daemon logs. Canonical journals, bounded code output, simulator stderr,
and recordings are still saved by RoboSWE on the host.

Campaigns preserve a ZIP and hashes of their coordinator Python source at launch.
`roboswe evaluate JOB_ID` produces a new verification report and video without
modifying the original run; it refuses active or unfinished campaigns.

Implemented components and controls:

- Eight numeric verifiers with explicit upstream thresholds, precondition checks,
  and ten consecutive successful control steps. Forty real-model predicate controls
  agree with the upstream conditions.
- A network-disabled code container with a bounded persistent workspace, no host
  mounts or credentials, and only `robot.observe()` / `robot.step(native_vector)`.
  Each attempt gets a fresh PID namespace; the kitchen remains in its own process.
- A host-owned, hash-chained submission journal. Independent replay re-executes
  code revisions chronologically, including code that failed after moving the robot.
  Exact initial-state and action-sequence mismatches prevent certification.
- A structured-response model client through `codex app-server`, using the existing
  ChatGPT login. Built-in host tools and inherited integrations are disabled.
- Optional blind visual auditing of timestamped MP4 frames through that login.
  With `--video-judge`, both retained-library tasks and matched cold comparisons
  receive this audit. Their jobs expose original and independent replay clips.
  Reports disclose all-frame versus sampled coverage. The audit does not overwrite
  a mechanical score or infer smoothness from sparse frames. It samples multiple
  recorded cameras and includes a consecutive ending window for the hold condition.
- Experimental `--video-judge glimpse` selects `elorian/glimpse` through OpenRouter
  for a native-video audit. It requires `OPENROUTER_API_KEY` in the host environment
  and records timestamped evidence, uncertainty, usage, and video hashes without
  changing scores. Existing recordings and custom public rubrics can also be
  audited separately. See [the Glimpse judge guide](../docs/glimpse_judge.md).

The public robot interface includes controller reference frames/scales, native
action slices, robot joint limits, Panda gripper frame conventions, and camera
intrinsics/extrinsics. Proprioception and configured wrist force/torque sensors are
exposed. `--camera-depths` adds compressed float32 depth arrays in optical-axis
metres; it does not add LiDAR. Fixture poses, object-state and segmentation labels
remain outside the participant interface. These are ideal simulated sensors; this
is not a validation of hardware localization accuracy.

Failed attempts are not replayed merely with the same wall timeout: faster
verification would otherwise advance the robot farther. For recorded wall-time
interruptions, verification executes the saved code up to the recorded accepted
control-step boundary and checks the action history. It does not play an action
log in place of executing code. Original timings remain the scored timings.

The real replay control executes three zero-action ticks, raises deliberately,
revises the file, then executes another tick; a fresh simulator and fresh code
workspace reproduce all four steps. This proves the plumbing, **not task-solving
ability**. Both runs record overview and robot-camera MP4s.

```sh
# Build the small code sandbox (no simulator assets in it).
python -c 'from roboswe.execution.sandbox import build_image; print(build_image())'
# Run actual Docker boundary controls.
ROBOSWE_TEST_DOCKER=1 python -m pytest tests/test_execution_sandbox.py -q
# Requires the simulator dependencies/assets in this Python environment.
python -m robot_session.check_library --output artifacts/predicate-controls.json
python -m robot_session.check_isolated_replay --output artifacts/replay-control
python -m robot_session.check_sensor_interface --output artifacts/sensor-control.json
```

Learning diagnostics compare matched first/second-round goals, preserve failure
and missing-task denominators, and report completion time, success changes and
execution-error rates. Failed early trials have censored completion times; they
are not treated as successful slow baselines. These diagnostics are not graded.

The library score instead compares retained-context second-round performance
with independently verified cold controls at identical task-start states:

`100 × warm task success rate × max(0, mean((cold censored time − warm censored time) / task budget))`

Failed attempts consume the full time budget for this comparison. Changing a
first-round time or padding execution counts cannot increase this score. Error
rates are reported separately, without a bonus that could reward denominator
manipulation. Missing/infrastructure-invalid controls withhold the score;
precondition-invalid goals get zero comparison credit and remain in the fixed
denominator. A single comparison is a pilot estimate of the combined effect of
retained context and code, not a confidence interval or proof that library files
alone caused the gain. Separate context-only/file-only ablations remain future
work. Do not describe the verifier as universally unrewardhackable. See
the internal implementation plan and evidence ledger (not included in this source package).

## Session lifecycle through the same command

```sh
roboswe session --help
roboswe session build docker
roboswe session start docker --image sha256:IMAGE \
  --config robot_session/config.example.json --receipt artifacts/session-agent.json
roboswe session status artifacts/session-agent.json
```

All existing lifecycle operations remain available here, including snapshot,
checkpoint, resume, replay, and explicit stop. Building images requires the source
checkout with official assets and an editable installation; assets are not shipped
in the Python wheel. Session `resume` restores a container checkpoint; the reserved
top-level `roboswe resume` command will eventually resume a benchmark campaign.

## Browse jobs

```sh
roboswe jobs
roboswe jobs --jobs-dir /path/to/job-output
```

This starts a read-only HTTP server bound to `127.0.0.1`, opens the HTML viewer in
your browser, and stays running until Ctrl+C. Use `--no-browser --port 3033` for a
fixed local URL. The default output directory is `jobs/` under the current working
directory. Viewing results does not start, stop, reset, or step a robot.

Select a job to watch **Video**, inspect **Before / After 3D**, the exact submitted **Code**,
**stdout**, and **stderr**. Drag to orbit, scroll to zoom, right-drag to pan, and
use Room, Around robot, or Overhead to change the view. Walls can be toggled.
The captured simulation step is shown in every scene.

## Run code and save job output

First create a persistent session using the [session instructions](../robot_session/README.md).
Then submit a Python file through the host-side job wrapper:

```sh
roboswe execute artifacts/session-agent.json my_action.py --timeout 120 --name "Approach oven"
roboswe jobs
```

The wrapper captures a before scene, executes the exact saved Python source once
inside that existing container, saves stdout/stderr and the process exit code,
then captures an after scene. It neither finishes nor tears down the session.
The timeout applies to submitted code; scene captures have separate bounded
timeouts. A snapshot failure is recorded as an artifact warning and does not
change a successful code exit into a benchmark failure. `completed` means that
the submitted process exited with code zero, **not that the robot task succeeded**.

The low-level `python -m robot_session.manage run ...` remains available and does
not collect these job records. Use `roboswe execute` or `roboswe.jobs.run_job` for
automatic job output. Run one action program at a time per session; this initial
job layer does not schedule or lock competing robot commands.

Capture an existing state without running action code:

```sh
roboswe capture artifacts/session-agent.json --name "Inspect current state"
```

Import a previously exported scene without accessing any container:

```sh
roboswe import-scene artifacts/after-scene.json.gz --name "Oven approach"
```

Capture/import records are labeled **snapshot**, not code executions.

## Continuous video for humans

`roboswe execute` records by default, starting immediately before submitted code
runs and stopping afterward, including when the code raises an exception or times
out. The recorder lives on the simulator's owner thread and captures every
completed control step. Observation reads and waiting do not advance physics.
It does not sample every internal MuJoCo physics substep.

The **Video** tab in `roboswe jobs` provides native playback/scrubbing, speed
selection, previous/next recorded frame buttons, camera selection, and two clocks:

- **Wall clock:** variable-timestamp MP4 preserving actual elapsed time and waits
  between control steps, including rendering, encoding, transport, and execution
  overhead. Idle intervals hold the last actual frame; they are not interpolated.
- **Simulation:** every initial/control-step frame in order at the environment's
  control frequency. Waiting time is omitted in this explicitly labeled view.

The human overview is a 640×480 cutaway view by default. Change it with
`--video-width` and `--video-height` on `execute`. Robot-camera videos retain the
actual observation resolution; a low-resolution robot observation is not silently
replaced with a sharper privileged render. All streams include an initial frame;
`video-timeline.jsonl` maps frames to step count, simulation time, and wall time.
The MP4 encoder adds a short final frame duration when finalizing the stream.

```sh
roboswe execute artifacts/session-agent.json my_action.py --video-width 640 --video-height 480
roboswe jobs
```

**Rebuild session images for recording support.** Existing running containers keep
their old code and do not acquire the recorder automatically. If recording cannot
start, `execute` reports `recording_failed` and does not silently run unrecorded.
`--no-video` is the explicit opt-out for legacy images or deliberate non-recorded
work. Mid-run recording errors are retained separately from code exit status;
partial clips remain available where possible. Complete recordings cannot be
guaranteed after a forced process/container kill, so collect them before provider
expiry. Fragmented MP4 reduces damage from interrupted finalization.

The session also exposes `/recording/start` and `/recording/stop`, and matching
`SessionClient.start_recording` / `stop_recording` methods. Notebook or future agent
loops must bracket the **whole attempt** with those calls to preserve the gaps
between multiple code submissions. Calling `/step` alone does not start a recording.
Only one recording can be active per session. Stop recording before checkpointing.

`--jobs` remains a compatibility alias for the preferred `jobs` subcommand.

These recordings and timeline metadata can be inputs to an optional video judge,
but no LLM judge or smoothness scoring has been implemented. They are agent-side
artifacts, not tamper-proof independent verifier evidence. A slow video alone is
not evidence of safe or skillful manipulation.

The bounded real-simulator check below creates its own local episode and tests
successful execution, waiting intervals, actual frame decoding, and retained video
after a deliberate code failure. It always closes that test process:

```sh
.venv/bin/python -m robot_session.check_recording --output /tmp/roboswe-video-check-UNIQUE
```

## Output contract

```text
jobs/<timestamp>-<id>/
  job.json                 # schema_version 1, timestamps, outcome, session provenance
  source.py                # exact submitted bytes, for execution jobs
  stdout.txt
  stderr.txt
  video.json               # view metadata, timing, recording status/errors
  video-timeline.jsonl      # real step / simulation / wall-clock events
  video-overview-wall.mp4
  video-overview-simulation.mp4
  video-camera-N-wall.mp4   # one pair per actual robot camera
  video-camera-N-simulation.mp4
  before.scene.json.gz      # if capture succeeded
  after.scene.json.gz       # if capture succeeded
```

`job.json` records artifact filenames, the submitted source SHA-256, task/session/step metadata for each scene,
and capture errors. Source and logs remain available when execution fails. An
interrupted host records uncertainty about the remote process; the viewer cannot
infer that a robot process has stopped from a stale `running` record.

The scene exporter reads the session's inspection ZIP and restores its physics
state into an independent model for geometry extraction. It never mutates the
session server. Captures contain visual geometry, poses, and textures, not an
executable physics checkpoint. Surface shading and primitive UV mapping are
approximate; this is a human inspection view, not a replacement for the robot's
camera observations or trusted scoring. Mesh geometry is preserved; textures
are downsampled to a maximum of 512 pixels per side. Only the first supported 2D
texture per material is represented. Geometry types outside the supported mesh,
plane, box, sphere, ellipsoid, capsule, and cylinder types are omitted.

## Notebook integration

Install the `notebook` extra to use the same viewer in marimo:

```python
from roboswe.scene import capture_scene
from roboswe.notebook import SceneViewer
import base64

# session is a robot_session.backends.SessionHandle
capture_scene(session, "after.scene.json.gz")
SceneViewer(scene_data=base64.b64encode(open("after.scene.json.gz", "rb").read()).decode())
```

In an async notebook cell, run blocking host calls with `await asyncio.to_thread(...)`.
The main viewer/export code lives in `roboswe/`; it is independent of the older
notebook setup under `pairing/`.

## Development checks

```sh
python -m pytest -q tests robot_session/tests
```

Third-party viewer code: Three.js 0.180.0, MIT license, retained under
`roboswe/viewer/vendor/THREE-LICENSE.txt`. The official RoboCasa/robosuite checkouts
and simulation assets have their own licensing and setup requirements.

## Physical reference control

A geometry-informed oven control reaches the unchanged 90%-of-travel criterion
through the restricted action interface and reproduces it by executing its code
in a fresh simulator. It is a test oracle, never supplied as a candidate motion
library and never counted as a model result. It uses bounded control chunks and
backs the mobile base away with the opening door to avoid arm joint limits.

```sh
python -m robot_session.check_oven_motion \
  --simulator-image sha256:SIMULATOR_IMAGE_ID --code-image sha256:CODE_IMAGE_ID \
  --output artifacts/oven-physical-control < robot_session/controls/open_oven.jsonl
```

For re-verifying a historical campaign whose requested allocation exceeds the
current Docker VM, an explicit allocation can be recorded separately:

```sh
roboswe evaluate JOB_ID --memory-mb 6144
```

This preserves the original plan and timings. Replays and model runs share the
reference-host lease. Campaign resume after a simulator failure is not implemented.
