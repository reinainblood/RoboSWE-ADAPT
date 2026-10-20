# From a model decision to a verified robot trial

RoboSWE-ADAPT gives a coding agent a robot interface and asks it to develop action
programs through five trials. The harness connects the model, its code workspace,
the simulator, and the evidence recorder. It also enforces the budgets and runs
verification. The agent supplies programs; the trusted runtime determines what
physically happened.

## Entrypoints and control

[`roboswe.adaptive.build_plan`](../roboswe/adaptive.py) resolves the dataset,
participant, images, and budgets. Its `run` function dispatches each task to
[`pairing.adaptive_run.run`](../pairing/adaptive_run.py), which owns the retained
model history and code workspace. [`pairing.cohort50.drive`](../pairing/cohort50.py)
is the decision loop used by this release: construct the current observation,
ask the participant, execute its operation, record the response, and repeat.

The model receives a structured operation interface for inspecting observations,
reading and writing files, executing programs, and finishing an attempt. Python
programs interact with the robot through the sandbox bridge. They can observe and
submit action vectors; they cannot call arbitrary simulator methods. The bridge
does not expose reset, object teleportation, physics-state assignment, or grading
mutation.

The action schema comes from the configured robot and controller. It documents
the vector shape, bounds, and controller conventions. This preserves the native
controller interface instead of supplying task-specific skills such as “pick up
the apple.” Different robot configurations can have different action dimensions;
programs must inspect the schema rather than assume one.

## What the agent sees

[`public_observation`](../roboswe/library/runtime.py) constructs the observation
boundary. It selects camera images, calibrated depth where configured, robot
joint and gripper state, end-effector poses, and documented localization signals.
It omits privileged object-state tables, object-to-gripper poses, semantic
segmentation labels, and scene XML. Simulated world localization and calibration
are explicit advantages of this interface; the benchmark does not model the
full uncertainty of a deployed robot's perception stack.

This is a harness-defined observation packet assembled from simulator sensors,
not an unfiltered controller log. Execution feedback adds the program's output,
errors, and accepted-action receipts. The model can request observations
explicitly. After a motion operation, the decision loop obtains fresh feedback
and images; file-only operations reuse the unchanged sensor state. Model-facing
images are JPEG encodings, while the simulator recording retains its own visual
evidence.

Physics advances when the program submits control steps or invokes the bounded
wait operation. Writing code, inspecting a file, or waiting for the model's next
response does not move the robot. Cameras attached to the robot change their
view as their physical mounting moves. There is no general scene-editing camera
or object manipulation interface.

## Budgets and learning trials

The configuration specifies active wall time, total control steps, and per-script
time and step limits. The decision loop tells the model those limits and the
remaining episode budget. These are distinct from the catalogue's median
demonstration duration and the native registry's episode horizon.

All five trials in a series use the same settings and initial-state fingerprint.
The physical scene resets between trials, while authored code and model history
continue. SQLite preserves the conversation; a sliding window removes complete
old exchanges from active context when necessary and enables bounded history
retrieval. The [context policy](sqlite-context-policy.md) describes native token
counting and the separately identified OpenRouter estimation policy.

The initial local pilot uses 1,800 active seconds, 6,000 control steps, and
50-step/60-second script limits. These are pilot settings, not universal RoboCasa
defaults. A future task configuration may need a larger step budget; that choice
must be fixed before comparing participants and retained across their trials.

## Mechanical grading and independent replay

[`SerialRuntime`](../roboswe/library/serial.py) evaluates the upstream task's
success predicate outside the authored-code sandbox. An initially solved scene
fails the precondition check. The current specification requires ten consecutive
successful control steps, rather than a single transient positive predicate.
Exceptions retain motion already performed; an error does not rewind the robot.

After each attempt, `verify_attempt` in
[`pairing.adaptive_run`](../pairing/adaptive_run.py) creates a fresh simulator and
code workspace, restores the trial's starting library, and replays the recorded
operations without another model call.
[`replay_journal`](../roboswe/library/attempts.py) checks recorded operations,
action digests, physical verdicts, and reproduction evidence. The
[results exporter](../roboswe/adaptive_results.py) only publishes an official
outcome when replay and precondition evidence are valid.

Isolation protects the grading state from direct writes, and replay checks that
the recorded program reproduces its outcome. Neither changes what the upstream
predicate means. A predicate that accepts an unintended physical shortcut still
needs task-specific review. Admission therefore includes inspecting the task's
natural-language objective and success code, alongside reset and execution
checks.

## Visual review and interpretation

The optional multi-angle video audit compares visible behavior with the task's
predicate claims. It reports contradictions and unobservable conditions
separately. An internal appliance flag may not be visually provable, so a video
judge supplements the mechanical evidence rather than replacing it.

The five-trial record supports success curves, reliable-success streaks, and
paired step or timing comparisons among successful attempts. Raw error counts
are descriptive: exploration and recovery should not automatically be penalized.
An efficiency claim needs a declared rule for sustained improvement across
several trials, not just a favorable pair. See
[reading adaptive results](adaptive-results.md) for the currently exported
measures and incomplete-outcome accounting.
