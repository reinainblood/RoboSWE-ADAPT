# Persistent RoboCasa sessions

Session infrastructure only. No agent harness, task implementation, scoring, or
verifier policy. Official source under `upstream/` is read-only.

One container owns one RoboCasa environment in one long-lived Python process.
Submitted Python runs as a separate process inside that container and uses
`robot_session.client.SessionClient` to interact with it. Returning, raising an
exception, or timing out the submitted program does not close the environment.
The simulation advances only on explicit `step` calls.

The lifecycle supports three different operations:

- **Inspection snapshot:** export scene XML, physics state, observations,
  episode metadata, configuration, versions, and identity as a ZIP. The process
  stays alive. This is not a Python/controller-memory restore format.
- **Container checkpoint/resume:** preserve the filesystem and process memory,
  including the live environment, and later resume that state. Backend-specific
  requirements and retention limits apply below.
- **Fresh verifier replay:** launch the original immutable image with the original
  configuration in a fresh container; require an exact initial XML/physics
  fingerprint match; then run the submitted code. Agent filesystem changes and
  its final simulation state are not used to initialize the verifier.

## Build and start

Run commands from the **RoboSWE-ADAPT project root**, one directory above `upstream`.
The host launcher needs Python 3.11+; only the Modal backend needs `modal==1.5.5`
(the tested SDK version). The host launcher does not need the robot dependencies.
The container installs its own simulation runtime.

```sh
python -m robot_session.manage build docker
# Copy the resulting sha256:... image ID:
python -m robot_session.manage start docker \
  --image sha256:YOUR_IMAGE_ID \
  --config robot_session/config.example.json \
  --receipt artifacts/session-agent.json
```

Docker requires a running engine. No ports, host directories, credentials, or
Docker socket are exposed to the container. The image includes both official
checkouts and their downloaded assets (about 10 GB in this workspace), so the
first build/upload is substantial. Later runs reuse it. Source changes require
rebuilding; replay reuses the recorded image ID without rebuilding.

For Modal, use the same operations with `modal`:

```sh
python -m robot_session.manage build modal
python -m robot_session.manage start modal \
  --image im-YOUR_IMAGE_ID \
  --config robot_session/config.example.json \
  --receipt artifacts/session-agent.json \
  --cpu 4 --memory 16384 --lifetime 86400
```

The SDK explicitly uses app `roboswe-365-sessions` in **FDR**. There is no public
web endpoint; clients communicate through authenticated sandbox
exec and container-local HTTP. Launch receipts retain image, sandbox/container,
app/environment, resource settings, and verified initial fingerprint.

CPU OSMesa rendering is used on both backends, including Apple Silicon Docker
VMs. CPU/memory are configurable. Modal's maximum lifetime is 24 hours; include
agent execution, inspection, and checkpointing in that budget. There is no idle
shutdown configured. Docker keeps running until explicitly stopped/checkpointed
or externally interrupted; its `--lifetime` value records the budget but is not a
Docker kill timer. No infrastructure can preserve memory through a forced kill;
export/checkpoint before the provider deadline.

`config.example.json` is a small infrastructure configuration: fixed task,
robot `PandaOmron`, layout/style 1, seed 0, and 64-pixel cameras. It is not a benchmark definition.
Change the task/configuration explicitly for actual work. `split` can be null,
`pretrain`, `target`, or `all`; explicit layout/style IDs apply with null split.
The same configuration is used for constructor and seeded reset. There is no
HTTP reset operation: a replay always starts a new process/container.

## Use the live session

The robot is a **required** top-level configuration field. For example:

```json
{
  "robot": "GR1FloatingBody",
  "task": "robocasa/PickPlaceCounterToCabinet",
  "split": null,
  "seed": 0,
  "env_kwargs": {"layout_ids": 1, "style_ids": 1}
}
```

The adapter passes the exact robot name to RoboCasa and checks the instantiated
robot model. It uses a local Gymnasium adapter over the native RoboCasa
environment, **not** upstream's PandaOmron-only Gym converter. It derives the
native action space from the selected robot/controller and discovers cameras
from the assembled scene. Explicit `env_kwargs.camera_names` are validated
against that scene. Observations retain native robot state keys and image keys,
plus `annotation.human.task_description`.

A submitted Python program can inspect the actual controls:

```python
from robot_session.client import SessionClient

session = SessionClient()
observation = session.observe()
schema = session.action_schema()
print(schema["shape"], schema["controllers"])

# One zero-vector infrastructure probe; this is NOT a neutral/hold command
# for every controller, nor a task solution.
action = [0.0] * schema["shape"][0]
observation, reward, terminated, truncated, info = session.step(action)
```

`step` takes the native numeric action vector without remapping its values.
The schema reports its size, bounds, dtype and controller input slices; slices
are `[start, end)` relative to each controller's `offset`. Whole-body IK exposes
its target inputs rather than the dimensions of its internal joint controllers.
JSON `null` in a lower/upper bound means negative/positive infinity. Action
values must still be finite. Missing, extra, out-of-range and non-finite action
values are rejected before stepping physics.

**API v2 migration:** the old six-field PandaOmron `action.*` dictionary is no
longer accepted. Native image keys such as `robot0_eye_in_right_hand_image`
replace the previous `video.*` aliases. Old images/receipts do not acquire this
interface by changing the notebook import: launch the new image with an explicit
robot. Existing live sessions are left alone. Robot identity participates in the
initial fingerprint and remains in checkpoints and fresh-replay receipts.

Verified local configurations are `PandaOmron` (12 action values) and
`GR1FloatingBody` (34). `GR1FloatingBody` is the upstream floating-body humanoid,
not a claim of freely walking GR1 support. Arbitrary installed robots are passed
through without name substitution, but RoboCasa must support the robot/task
combination. This checkout's kitchen placement fails for plain `GR1` because it
assumes mobile-base joints; `Tiago` encountered a MuJoCo sparse-Hessian error.
The tested `SpotWithArmFloating` also hits the upstream mobile-base joint-name
assumption. These are initialization failures, not silently substituted Panda sessions.

Programs creating their own environment will not operate this persistent session.

```sh
python -m robot_session.manage run artifacts/session-agent.json submitted.py --timeout 600
python -m robot_session.manage status artifacts/session-agent.json
python -m robot_session.manage finish artifacts/session-agent.json
python -m robot_session.manage snapshot artifacts/session-agent.json artifacts/agent-final.zip
```

`run` uploads the exact source bytes under a SHA-256-derived filename, executes
that Python file in `/workspace`, and returns its stdout/stderr and exit status.
It does not automatically finish, snapshot, score, or tear down the session.
Additional inspection code can be run the same way before teardown. Dependency
files or multi-file submissions can be placed using backend exec/file APIs; this
small runner accepts a single Python entrypoint and does not define submission
packaging for the benchmark.

`finish` is idempotent and freezes further actions while retaining observations
and the live environment. A failed physics step also freezes further mutation
because partial state changes may have occurred. Other code-process failures do
not affect the server. There is no reset/unfreeze API; resume preserves whichever
phase was checkpointed. Checkpoint an active session if you need to continue
stepping it after resume.

A ZIP contains `manifest.json`, `model.xml`, `observations.npz`, and `physics.npz`
(flattened simulator state plus MuJoCo integration state). NumPy files load with
`allow_pickle=False`. A sibling `.session.json` records the host-side container
identity. Observations and snapshots do not step or reset physics.

These artifacts are inspection evidence. Agent-side observations/reward fields
are not trusted verifier verdicts; replay and scoring belong to the independent
verifier. Arbitrary submitted code has access to its own container, so this
wrapper does not claim to prevent tampering within the agent container.

## Fresh verifier session

```sh
python -m robot_session.manage replay artifacts/session-agent.json \
  --out artifacts/session-verifier.json
python -m robot_session.manage run artifacts/session-verifier.json submitted.py --timeout 600
python -m robot_session.manage finish artifacts/session-verifier.json
python -m robot_session.manage snapshot artifacts/session-verifier.json artifacts/verifier-final.zip
# Run your own verifier inspection here, then explicitly tear down:
python -m robot_session.manage stop artifacts/session-verifier.json \
  --snapshot artifacts/verifier-at-close.zip
```

The original agent receipt remains usable after that container has stopped.
Its base image must still exist. Same-image replay is within the original backend
and architecture; cross-architecture bitwise replay is not promised. Initialization
mismatches fail startup and clean up the new container rather than accepting a
different scene. Exact initial equality is a precondition, not a guarantee that
all arbitrary submitted code is deterministic.

## Resumable whole-container checkpoint

After all exec commands have returned, first export the inspection ZIP, then:

```sh
python -m robot_session.manage checkpoint artifacts/session-agent.json artifacts/agent-checkpoint.json
# This operation suspends/stops the original container.
python -m robot_session.manage resume artifacts/agent-checkpoint.json \
  --out artifacts/session-resumed.json
python -m robot_session.manage status artifacts/session-resumed.json
```

Resume verifies session UUID, initial fingerprint, step count, and phase. It
preserves the running state; it does not rerun the entrypoint or reset RoboCasa.
The original base image remains in the receipt so fresh verifier replay still
starts clean even when invoked from a resumed session receipt.

**Modal:** uses experimental memory snapshots, not filesystem-only snapshots.
They include memory and filesystem, currently terminate the original sandbox,
and cannot capture an outstanding exec command. CPU only; no GPU or pinned
region. Snapshots expire seven days after creation, and snapshot chains inherit
the first snapshot's expiry. Restoring may wait for matching instance capacity.
The checkpoint receipt stores the snapshot ID. See
[Modal snapshot documentation](https://modal.com/docs/guide/sandbox-snapshots).

**Docker:** uses native CRIU checkpoint/restore on the same Linux daemon. Requires
experimental daemon features, CRIU installed on the **daemon host**, and compatible
kernel/security settings. Installing CRIU inside this image is not sufficient.
The command checks basic daemon support and reports native errors without falling
back to a filesystem-only copy. The stopped container and its daemon storage
must be retained; do not remove/prune it. Resume starts that retained container
with its checkpoint. Moving the checkpoint between hosts or restoring it as a
fresh fork is not implemented. The ordinary `stop` method stops a
checkpointed container but retains its filesystem and checkpoint data. A Docker
checkpoint receipt is single-use: after resume, take a new checkpoint for a new
suspend point. Rewinding old process memory over a since-modified filesystem is
not supported. See
[Docker checkpoint documentation](https://docs.docker.com/reference/cli/docker/checkpoint/).

Checkpoints are backend-specific; neither is a portable Docker-to-Modal memory
migration format. The inspection ZIP remains the portable analysis artifact.

## Validation

```sh
.venv/bin/python -m pytest -q robot_session/tests
# Optional explicit real-simulator check, separate from your notebook:
.venv/bin/python -m robot_session.check_local --robot PandaOmron --output /tmp/roboswe-session-check-UNIQUE
```

The local check launches two independent simulator processes, compares their
entrypoint and post-action fingerprints, exports snapshots, checks hold behavior,
and always tears them down. It does not execute notebook cells.

The Dockerfile installs pinned direct **simulation** dependencies and editable
upstream source without optional upstream training/dataset dependencies. It is
not a complete RoboCasa training environment. Full installed versions are saved
in `/app/runtime-packages.txt`; reuse immutable image IDs for replay. Transitive
versions are resolved at build time; separate builds are not a lockfile guarantee.

For an explicit live container check after building an image:

```sh
python -m robot_session.check_container modal --image im-YOUR_IMAGE_ID \
  --robot GR1FloatingBody --checkpoint --output /tmp/roboswe-modal-check-UNIQUE
```

This checks real code execution, survival after a code exception, snapshots,
whole-container memory/filesystem resume, further stepping, clean verifier
replay, and teardown. It creates metered Modal resources and terminates its probe
containers in `finally`. Docker uses the same command with `docker`; add
`--checkpoint` only on a CRIU-capable daemon. Docker checkpoint containers are
stopped but retained because deleting them would destroy their checkpoints.

### Verified state

- 22 lifecycle/configuration/schema tests pass; Ruff checks pass.
- Local real-simulator lifecycle checks pass for PandaOmron and GR1FloatingBody:
  fresh initial-state equality, native action execution, matching post-action
  physical state on replay, inspection snapshots, hold, and cleanup.
- Both robots passed full container checks in Modal FDR using image
  `im-JkBCuOs89zM0zHyeU7CVLz`: submitted-code execution, survival after a code
  exception, inspection export, memory/filesystem checkpoint and resume,
  continued stepping, and fresh verifier replay with matching physical state.
- All six robot-aware probe containers were confirmed stopped afterward.
- Local Docker execution and CRIU restore remain unverified: this host has no
  Docker CLI/engine available.

The image was built by adding the updated session package to the previously
validated immutable simulation image; robot assets and dependency versions were
unchanged. Tests use the example task/layout and a zero action, not a benchmark
solution or evidence that a locomotion/manipulation policy succeeds.

## Use from the notebook

The root `notebook.py` imports this exact checkout in its visible setup cell:

```python
from robot_session import backends as sessions
```

The notebook declares `modal==1.5.5` in its sandbox dependencies. Its `sessions`
variable exposes `start`, `replay`, `resume`, `build_image`, and `SessionHandle`.
The import was verified in the live kernel and does not create a container.
Use `sessions.start(...)` when you choose to launch, retain the returned handle,
and use its `run_code`, `snapshot`, `checkpoint`, `finish`, and `stop` methods.
`session` code in the container itself uses `SessionClient` as described above.

Open the notebook with `marimo edit notebook.py --sandbox` so its declared
packages are installed. Keep `notebook.py` beside the `robot_session/` package;
both are in this repository. The robot libraries run inside the container, so
the notebook kernel does not need its own RoboCasa installation.

## Continuous recording

Rebuild images to include `av==18.1.0` and the session-owned recorder.
`roboswe execute` starts and collects recordings automatically; `roboswe jobs`
plays them. The low-level `manage run` command and individual `SessionClient.step`
calls do not automatically start recordings. For a multi-submission notebook
attempt, start one recording before the first submission and stop it after the
last so wall-clock gaps are preserved.

The recorder captures each completed control step plus initial/final timing,
with separate simulation-time and wall-clock MP4 streams. A human cutaway and
the selected robot's actual RGB observations are saved. Rendering happens on the
simulator owner thread without resetting or advancing it. Inference and coding
pauses have no semantic labels: the timeline simply records waiting between steps.

`POST /recording/start` accepts `recording_id`, `width`, and `height`.
`POST /recording/stop` accepts `recording_id` and an optional `reason`. It returns
a manifest whose files can be fetched at `GET /recording/<id>/<filename>`.
Recordings are stored under `ROBOT_RECORDINGS_DIR` (default `recordings/` in the
container working directory). Existing recording IDs cannot be overwritten.
The host must collect files before the sandbox expires. Active recordings must
be stopped before container checkpointing. There is no claim of recording
integrity against arbitrary code inside the same agent container.
