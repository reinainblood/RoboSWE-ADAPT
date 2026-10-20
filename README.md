# RoboSWE-ADAPT

**Adaptive Development of Action Programs Through Trials**

RoboSWE-ADAPT evaluates how coding agents turn robot experience into better action
programs. An agent gets five attempts at the same household task in RoboCasa365.
The kitchen resets between attempts; the agent keeps its code and interaction
history. It can inspect camera images and robot sensors, write Python, execute
bounded controller actions, and revise its programs after seeing what happened.

[Website and demonstrations](https://roboswe.com) ·
[Run the benchmark](docs/adaptive-cli.md) ·
[Harness and verification](docs/harness-and-verification.md) ·
[Task catalogue](datasets/adaptive-release/selection.json) ·
[Draft preprint](https://roboswe.com/paper/) ·
[Methods source](paper/main.tex) ·
[Evidence dashboard](docs/review-dashboard.md)

## What the benchmark measures

A successful attempt must satisfy the task's mechanical predicate for a sustained
interval. The adaptive pipeline then independently replays the recorded code in a
fresh simulator and workspace. Reports distinguish observed success, reproduced
success, and invalid verification. Available camera views, code, controller
receipts, and provider-exposed reasoning make the behavior inspectable.

Five attempts let us study more than a single pass or fail: whether failure turns
into reliable success, whether successful programs use fewer control steps, and
how the agent recovers from mistakes. The methods draft proposes these adaptation
measures separately; the first release will freeze its scoring specification
before reporting comparative rankings. Optional video auditing checks visible
predicate evidence without replacing the mechanical verifier.

## Tasks and release development

The development catalogue contains **83 selections representing 85 native task
names**, including two proposed task compositions. It emphasizes longer household
activities with six or more subtasks and several skills, supplemented with about
twenty shorter calibration candidates. Selection uses the official
[RoboCasa task catalogue](https://robocasa.ai/docs/build/html/tasks/composite_tasks.html).
Its “Horizon” column is median demonstration duration in seconds, not the maximum
control-step budget or model wall-time limit.

Each runnable task needs an admitted scene: task, robot, layout, style, seed,
controller, initial-state reference, and explicit budgets. The candidate selection
file is the menu; a runnable dataset has a `tasks` array containing these scene
configurations. Environment defects are documented during admission. A model's
failure on a valid task is an outcome, not a reason to discard the task.

The results view reports trial coverage and verifier status. Demonstrations
from earlier pilots carry their original protocol labels.

## Run a dataset

Use Python 3.11 or newer and a running Docker engine. For a fresh source checkout,
install the harness, prepare the pinned upstream sources and assets, then build
the images:

```sh
git clone https://github.com/reinainblood/RoboSWE-ADAPT.git
cd RoboSWE-ADAPT
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[modal,judge]'
roboswe prepare --source-root .
roboswe prepare --source-root . --run --out provisioning.json
roboswe build-images -e docker --source-root . --out docker-images.json
```

The first preparation command previews the work; `--run` performs it. The
upstream asset downloader asks for confirmation. Existing upstream installations
are inspected read-only. See [provisioning](docs/provisioning.md) for source pins,
asset requirements, and existing-checkout behavior.

Pass the image-build receipt directly with `--images`. Use the
[prospective eleven-task pilot](datasets/adaptive-pilot11/README.md), or define
your own scenes from [the manifest example](examples/adaptive/dataset.json).
Select the participant and budgets in [the run configuration](examples/adaptive/run.json).
The image receipt supplies immutable image IDs and records their provenance. Supply provider
credentials through the trusted coordinator's environment or secret manager.

Copy `examples/adaptive/run.json` to `my-run.json`, set the exact provider/model
and budgets, and supply the corresponding API key in your shell environment.
The default example uses `OPENAI_API_KEY`; see the CLI guide for other providers.

```sh
cp examples/adaptive/run.json my-run.json
# Edit my-run.json before admission or running.
roboswe admit datasets/adaptive-pilot11 --config my-run.json --images docker-images.json -e docker --out admitted.json
roboswe run admitted.json --config my-run.json --images docker-images.json -e docker --dry-run
roboswe run admitted.json --config my-run.json --images docker-images.json -e docker
python -m roboswe.review --port 8876
```

Admission performs two independent simulator resets without calling a model.
It saves matching initial fingerprints and records rejected tasks separately.
Use the same admitted scene manifest and image receipt when comparing models.

`roboswee` is an alias for `roboswe`. A directory containing `dataset.json` can also
be supplied. The [CLI guide](docs/adaptive-cli.md) covers providers, spending caps,
Modal provisioning, verification, and output records.

## Execution and context

Authored code runs in a networkless Docker sandbox. A separate trusted simulator
owns physical state and grading. The model gets the documented action vector and
robot observations, not a writable simulator or grading API. Exceptions retain
any motion already performed, so the next decision sees the resulting scene.

`-e docker` runs the simulator locally. `-e modal` runs it remotely using standard
Modal authentication and an explicitly selected profile and environment. The
coordinator and authored-code Docker sandbox stay local in both modes. Provider
keys belong only to the coordinator.

History is archived in SQLite, including provider-returned reasoning when exposed.
Active context slides by complete exchanges without generated summaries. Native
API adapters use count endpoints; OpenRouter offers an explicitly selected
conservative budgeting policy with estimates and actual usage recorded together.
See the [context policy](docs/sqlite-context-policy.md) for the exact rules and
comparison implications.

## Inspect and share

The [light evidence dashboard](docs/review-dashboard.md) presents multiple camera
angles alongside decisions, code, and available reasoning. It exports a full
native episode record and an ATIF-style projection. A static export can be shared
without credentials or a simulator. The optional local video-audit button runs
only for finalized jobs when enabled by the server operator.

The [public website](https://roboswe.com) provides curated playthroughs and a
results snapshot. The public site is static; it does not expose a paid model API.
Remote artifacts are certified durable only after hash-verified readback.

## Development and related components

```sh
python -m pytest tests robot_session/tests -q
```

Simulator integration checks require the upstream dependencies and assets.
The adaptive runner keeps trials sequential and does not automatically resume a
crashed series. Certified between-trial continuation is an explicit operation;
its evidence requirements and commands are described in the CLI guide.
SQLite exports are retained locally and can be backed up to Modal FDR with
hash-verified readback and a tested restore path. See the
[context archive workflow](docs/sqlite-context-policy.md#remote-evidence-backups).
Backups preserve evidence; they do not enable automatic resumption of a crashed
series.

- [Persistent robot sessions](robot_session/README.md)
- [Modal simulator and artifact infrastructure](docs/modal-trials.md)
- [Michael OpenArm/Wuji integration](robot_session/michael/README.md)
- [Programmatic body control](robot_session/michael/BODY.md)
- [Historical campaign commands and protocols](docs/legacy-campaigns.md)

RoboSWE-ADAPT builds on RoboCasa365, robosuite, and MuJoCo. Upstream code and assets
retain their own licenses. Third-party viewer code includes Three.js under its
retained MIT license. Credentials, machine-specific settings, and raw run/video collections are
excluded from the source repository; selected public demonstration assets are
included with their provenance.

## License

Original RoboSWE-ADAPT code is licensed under the [Apache License 2.0](LICENSE).
Copyright 2026 Kirsten Ruge.

RoboCasa, robosuite, MuJoCo, third-party robot assets, and vendored libraries
retain their respective licenses and attribution. The project license does not
relicense those materials. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Contact

Maintained by [Kirsten Ruge (@reinainblood)](https://github.com/reinainblood).
Open a [GitHub issue](https://github.com/reinainblood/RoboSWE-ADAPT/issues) for
questions, bug reports, or contributions. Visit [roboswe.com](https://roboswe.com)
for demonstrations and the [draft preprint](https://roboswe.com/paper/).
