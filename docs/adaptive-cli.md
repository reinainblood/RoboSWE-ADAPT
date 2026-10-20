# Running RoboSWE-ADAPT

RoboSWE-ADAPT runs five retained development trials per task. Use one scene
manifest and one run configuration to select the tasks, model, budgets, and
execution backend. The same command collects evidence and independently replays
each completed attempt.

```sh
roboswe run DATASET --config CONFIG -e docker|modal --images IMAGE_BUILD_RECEIPT
```

`roboswee` is an alias. `DATASET` can be a JSON file or a directory containing
`dataset.json`. Older flag-only `roboswe run --mode ...` invocations use the
[legacy protocols](legacy-campaigns.md); their flags and context policies are not
part of an adaptive invocation.

## Install and build

Use Python 3.11 or newer. Install the pinned upstream sources and simulator assets
as described in the [session guide](../robot_session/README.md). Docker must run on
the coordinator host, including when the simulator is on Modal.

```sh
pip install -e '.[modal,judge]'
roboswe build-images -e docker --source-root . --out docker-images.json
```

Pillow for camera feedback is a core dependency; the `judge` extra adds host video decoding.
Pass the build receipt directly; image IDs need not be copied into the run configuration:

```sh
roboswe run DATASET --config CONFIG -e docker --images docker-images.json --dry-run
```

Remove `--dry-run` to execute the plan. Builds may allocate resources; planning
only reads local JSON and never builds, downloads, resolves credentials, or starts
inference. Config-only invocations still work when image IDs are provided in
`execution`.

## Admit a scene set

The [prospective pilot](../datasets/adaptive-pilot11/README.md) supplies eleven
task configurations. Before spending model credits on a new image and scene set,
run simulator-only admission:

```sh
roboswe admit datasets/adaptive-pilot11 --config run.json --images docker-images.json \
  -e docker --out admitted.json
roboswe run admitted.json --config run.json --images docker-images.json -e docker
```

Admission performs two independent resets per task, checks that the initial
predicate is false and the fingerprints match, and executes five diagnostic
control steps. It calls no model. Modal admission also requires full remote
artifact readback. Use `--dry-run` to inspect the bounded allocation plan first.

The output pins accepted fingerprints. `admitted.json.admission/` preserves
reset receipts, and `admitted.json.rejected.json` records rejected or unattempted
tasks. Source and admitted task/trial counts remain separate. Existing outputs
are never overwritten. Infrastructure or cleanup failures stop admission;
environment rejection is not a model failure.

Use the same admitted manifest, image receipt and trial settings for models
being compared. A new image or asset inventory defines a new scene revision.

## Modal images

For Modal, authenticate through its standard CLI and select your configured
profile and environment:

```sh
modal setup
roboswe build-images -e modal --modal-profile YOUR_PROFILE \
  --modal-environment YOUR_ENVIRONMENT --out modal-images.json
```

Consult `modal setup --help` for profile options. Use the resulting receipt:

```sh
roboswe run DATASET --config CONFIG -e modal --images modal-images.json --dry-run
```

The receipt supplies the explicit profile/environment and both immutable image
IDs. `--images` replaces those build fields in `execution`, while preserving other
settings such as `modal_app` and artifact transfer. The receipt backend must match
`-e` (or the configured backend when `-e` is absent). Unknown receipt fields are
rejected; receipts cannot inject credentials or executable settings. The plan and
each task record the receipt path and SHA256 of its exact bytes. Without
`--images`, set these image/profile/environment fields in the config as before. This workspace uses `FDR`; another installation uses its own environment.
Modal authentication stays in the normal profile store. It is distinct from the
model provider credential. The generated plan resolves
`execution.modal_volume` to `roboswe-job-artifacts-v1` in your selected profile and
environment, or uses your explicit custom volume name. Simulator artifact storage,
final SQLite archival, receipts and generated restore commands all use those same
resolved settings. A missing named volume is not permission to fall back to another
environment or workspace.

## Prepare a dataset and run configuration

[dataset.json](../examples/adaptive/dataset.json) shows the required task row:

```json
{
  "tasks": [{
    "id": "blend-ingredients-example",
    "task": "BlendIngredients",
    "config": {
      "robot": "PandaOmron",
      "task": "robocasa/BlendIngredients",
      "seed": 0,
      "split": null,
      "env_kwargs": {"layout_ids": 1, "style_ids": 1,
                     "camera_widths": 640, "camera_heights": 480}
    }
  }]
}
```

That row is a schema example, not a claim that the scene has passed admission.
Use a verified scene configuration for evaluation. An optional
`initial_fingerprint` pins its expected initial state. Otherwise the first reset
establishes the reference for the subsequent attempts. Admission should establish
scene validity and reset reproducibility before comparing participants.

The [run example](../examples/adaptive/run.json) contains `participant`, `trials`,
`limits`, `execution`, and `video_judge`. Set the exact model ID and verified
context/output limits. Image placeholders must be replaced before planning.
The default five-trial limits are:

| Budget | Default |
| --- | ---: |
| Active work per attempt | 1,800 seconds |
| Total robot control steps | 6,000 |
| Control steps per script | 50 |
| Wall time per script | 60 seconds |

These budgets are explicit and can be configured in `limits`; keep them identical
across the five attempts and matched model comparisons. The catalogue's Horizon
column represents median demonstration seconds. It is distinct from a registry
step horizon and from these execution limits.

## Model providers and credentials

Supply the key through the coordinator process environment or your secret
manager. Do not put its value in JSON, authored robot code, or a simulator image.
`participant.api_key_env` can select a different environment-variable name.

| `participant.provider` | Default key variable | Context admission |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | Native Responses token counting |
| `anthropic` | `ANTHROPIC_API_KEY` | Native Messages token counting |
| `google` | `GEMINI_API_KEY` | Native Gemini `countTokens` |
| `openrouter` | `OPENROUTER_API_KEY` | Explicit versioned estimate below; v2 also needs `GEMINI_API_KEY` for counting |

The current Anthropic adapter requires its pinned model ID; inspect the configured
adapter rather than treating it as an arbitrary model gateway. An optional
`workspace_id` selects the Anthropic workspace. Google counting and generation
use the same request builder; access to counting does not imply generation quota.

For a new native Anthropic series, set `participant.thinking_display` to
`"summarized"` to request **provider-generated thinking summaries**. The client
sends the same adaptive/display configuration to counting and generation and
records the selected representation in model metadata and thinking traces.
Returned blocks and signatures remain unchanged in history and evidence. This
is not raw internal reasoning, and readable text is not guaranteed on every turn.

Omitting the field (or using `null`) preserves the legacy request payloads and
headers; Opus5.5's documented default is omitted readable thinking. Other values
and use with another provider are rejected. This option changes only requested
output visibility: it does not enable a thinking-drop policy or change SQLite
sliding, effort, model or output limits. Keep the display setting fixed across a
series and disclose it when comparing latency. See the
[official display documentation](https://platform.claude.com/docs/en/build-with-claude/thinking).

The Anthropic and mapped OpenRouter Gemini counters allow at most two attempts
for a transient counting-service failure. Each request is capped at 90 seconds
and by the time remaining in the original episode. Counting generates no model
completion; its attempts and latency are recorded. Inference requests are not
automatically retried. An uncertain Anthropic inference charge keeps its budget
reservation until it can be reconciled with provider evidence.

For prospective OpenRouter Argon runs, select `sqlite-sliding-or-google-count-v2`
explicitly and supply the Google count-only credential alongside the OpenRouter
inference credential. This calibrated estimate preserves more context than the
legacy byte/image proxy while retaining a safety margin. It is not exact native
OpenRouter counting. The public runner requires a shared ledger, spend cap, and
pinned prices:

```json
{
  "provider": "openrouter",
  "model": "google/gemini-4-argon",
  "context_window": 1048576,
  "max_output_tokens": 32768,
  "effort": "high",
  "context_policy": "sqlite-sliding-or-google-count-v2",
  "ledger_path": "artifacts/my-run-spend.json",
  "spend_limit_usd": 10.0,
  "input_price_per_million": 2.0,
  "output_price_per_million": 10.0
}
```

This is a participant-object example, not a complete run configuration. The v2
mapping supports this exact model; verify its capabilities and prices before use.
The legacy `sqlite-sliding-or-conservative-v1` remains identifiable for historical
development evidence but is not recommended for new Argon comparisons. The
ledger reserves the full configured input-window cost plus output allowance
before a request and settles against its usage receipt. Uncertain request outcomes
retain reservations. A cap may stop a run with some credit remaining rather than
risk exceeding the cap. API errors do not trigger silent retries or model swaps.
Native Google and Anthropic also support an explicit ledger and spending cap.
The [context guide](sqlite-context-policy.md) explains the policy differences.

## Plan, run, and inspect

```sh
roboswe run my-dataset.json --config my-run.json -e docker --dry-run
roboswe run my-dataset.json --config my-run.json -e docker
roboswe run my-dataset.json --config my-modal-run.json -e modal
python -m roboswe.review --port 8876
```

Planning validates configuration without resolving secrets or allocating a
simulator. A real run starts a fresh simulator for each attempt, retains one
model history and authored code workspace per task, and checks each reset before
model input. Initial-state mismatches stop the series with their evidence.

The Docker backend runs the simulator locally. Modal runs the simulator remotely;
the trusted coordinator and networkless authored-code Docker container remain
local. Optional `modal_app`, `modal_volume`, and `modal_sdk_python` specify remote
infrastructure and SDK interpreter. This is not a cloud-only coordinator, so
provider keys are not injected through Modal secrets into the simulator or
participant sandbox.

## Verification and output

Trusted simulator predicates are evaluated during execution and must remain true
for the configured success interval. After a completed attempt, the pipeline
re-executes its recorded code operations from the complete starting workspace in a fresh
simulator and workspace. `verification.json` distinguishes reproduced success,
reproduced failure, and invalid replay. A replay error does not become a pass.
Replay uses simulator compute but makes no new participant inference calls.

Workspace checkpoints include generated data files such as NumPy arrays and JSON,
as well as Python source. Each archive records file hashes, permissions, and
modification timestamps. Restoration validates the complete archive before
writing into an empty workspace. Checkpoints support regular files and directories,
with a 256 MiB content limit and 16,384-entry limit; symbolic links, hard links,
special files, and unsafe paths are rejected explicitly. The checkpoint helper
runs separately in the same pinned code image, with no network access.

The job inventory connects reports, authored code, operation journals, controller
receipts, camera recordings, provider responses, available reasoning, SQLite
history exports, and slide events. Preserve model errors and incomplete trials;
do not silently replace a completed failure. The current runner executes
sequentially. In-trial crash resume remains unsupported.

A narrow certified **between-trial continuation** path supports the pinned
OpenRouter Gemini mapped-count v2 client after one to four verified trials, before
any model input or action in the next numbered trial. It requires an exact private model/history
checkpoint, a complete workspace checkpoint, verified evidence durability, and
recorded termination of the original owned processes and simulators. An ordinary
saved conversation or source archive is insufficient.

```sh
roboswe resume-adaptive ORIGINAL_PARENT --certificate CERTIFICATE
# Only after inspecting the validated boundary:
roboswe resume-adaptive ORIGINAL_PARENT --certificate CERTIFICATE --run
```

The default command validates local evidence without inference or allocation.
`--run` additionally checks provider exclusivity, owned simulator cleanup and
unsettled spending, then continues the original parent and numbered trials.
The zero-input next-trial prefix is preserved under `resume-history`; its original
child ID is reused. Completed trials are never repeated. This is a maintenance
recovery procedure with a reviewed certificate, not a general restart command.
Every completed trial needs ordered verification and durable evidence. Current
support does not extend to other clients or interruptions after next-trial input.

## Optional video audit

Video auditing defaults off. To request a paid predicate-specific audit after
replay, add an explicit OpenRouter audit budget:

```json
"video_judge": {
  "backend": "gemini",
  "ledger_path": "artifacts/visual-audit-spend.json",
  "spend_limit_usd": 10
}
```

`gemini` selects exact Argon on Google Vertex/global; `glimpse` selects the
experimental Together route. Supply `OPENROUTER_API_KEY` to the trusted
coordinator. A shared participant/audit ledger must use the same cap in both
configurations. Unknown request costs retain their reservations.

The local dashboard also offers an on-demand button for finalized jobs, enabled
with an explicit backend, ledger and cap. The key never reaches browser code.
Visual findings remain separate from mechanical scores. A visual failure agreeing
with a mechanical failure is reported as consistent failure, not a verifier
disagreement. See the [review guide](review-dashboard.md).

## Development coverage

The Docker path has completed a two-reset preflight and live OpenRouter trial
execution. A finished five-trial release result is still pending. Native counting and
generation access are checked independently during provider qualification.
Backend qualification is recorded per environment;
a Docker run does not establish a live Modal result.

Modal attempts now require verified remote video transfer and archive their
finalized SQLite/evidence bundle with a separate readback receipt. Local MP4
copies remain available for the viewer. A storage failure stops the series while
preserving the physical verification result. Docker jobs can use the explicit
[archive command](sqlite-context-policy.md#remote-evidence-backups). A video
receipt alone does not certify the separate history database. In-trial crash
resume remains unsupported; certified between-trial continuation requires the
additional evidence described above.
