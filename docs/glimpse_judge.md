# Optional video audits

This is an optional visual audit beside RoboSWE's mechanical verifier. It is off
by default. `changes_score` is always false: a model opinion never replaces the
trusted state-transition, sustained-success, or independent-replay requirements.

## Current integration

The dashboard now defaults to `gemini` when paid auditing is explicitly enabled:
exact `google/gemini-4-argon` through `google-vertex/global`, with low reasoning
effort and 8,192 output tokens. It uses the same blind predicate rubric and
multi-angle recordings. In a two-case integration check it produced structured
judgments consistent with one historical observed success and one replay-verified
failure. This checks the pipeline; it is not an accuracy estimate across tasks.

Glimpse remains an explicit experimental option. Its positive-case audits
exhausted their output allowances, including a low-effort 32,768-token check,
without a final structured verdict. Those audit failures remain preserved.

Both backends expose timestamps and criterion-level evidence, including unknown
conditions. Reports distinguish a visible failure from disagreement with a
mechanical result. See [dashboard configuration](review-dashboard.md) and
[automatic audit configuration](adaptive-cli.md#optional-video-audit) for the
required spending ledger and cap. Historical Glimpse-specific details follow.

## Model research, checked 2026-10-18

[OpenRouter's listing](https://openrouter.ai/elorian/glimpse) and
[endpoint metadata](https://openrouter.ai/api/v1/models/elorian/glimpse/endpoints)
identify `elorian/glimpse`, currently served by Together, with text/image/video
input, text output, structured outputs, and 262,144 context tokens. Listed prices
are $0.60 per million input tokens and $3 per million output tokens. Video token
usage is provider-dependent; file size is not a token or dollar estimate.

[Together's model card](https://www.together.ai/models/elorian-glimpse) describes
spatial reasoning, object tracking, manipulation-event understanding, and
timestamped episode segmentation. Its benchmark figures are Elorian-reported.
[Elorian's launch write-up](https://elorian.ai/blog/introducing-glimpse) uses its
native agent harness at maximum thinking effort. This bounded, single-request
integration does not reproduce that harness or its tools. Model capability claims
are a reason to test the judge, not evidence of RoboSWE judging accuracy.

The request follows [OpenRouter's video-input API](https://openrouter.ai/docs/guides/overview/multimodal/videos).
Full MP4 bytes are submitted as base64 `video_url` parts. The report explicitly
records that provider sampling is unknown; sending a full clip does not prove
every frame was inspected. Exact joint thresholds, hidden forces, and ten
consecutive simulator ticks cannot be certified from an ambiguous video.

## Enable it

Install the existing `judge` extra (`av` and Pillow) and supply
`OPENROUTER_API_KEY` in the coordinator's environment through your normal secret
handling. The client reads that one variable; it does not load credential files.
The key is never written into the plan or report and is never passed to the robot
participant. Selected video bytes and public requirements go to OpenRouter/Together.

Add `--video-judge glimpse` to an otherwise configured `roboswe run` command. This
applies after independent replay, including matched cold comparisons. Plans save
the selected backend. `--video-judge` alone retains the existing Codex frame audit;
older plans without a backend field also retain the Codex route.

To audit an existing verifier-owned recording without running a campaign:

```sh
python -m roboswe.library.glimpse_judge /absolute/path/replay.mp4 \
  --instruction "Place the mug beside the plate and release it." \
  --output /absolute/path/new-glimpse-audit
```

The output directory must be new. `--mechanical-success true` or `false` is an
optional trusted local comparison, never part of the model prompt. `--criteria`
accepts a JSON file containing public success requirements. `--rubric` accepts
a JSON array such as:

```json
[
  {"id": "placement", "description": "At the end, the mug is visibly beside the plate."},
  {"id": "release", "description": "The gripper visibly releases the mug, which remains supported."}
]
```

A supplied rubric replaces the default single task-goal criterion; include every
requirement you want assessed. Its conjunction defines the visual result. Public
criteria must not contain measured values, reference answers, or mechanical
outcomes. Use recordings from one episode, with camera names and timestamps
referring to each individual clip; do not concatenate unrelated attempts.

The Python entry point also accepts `additional_videos={"wrist": path}` and
explicit `max_video_bytes`, `max_video_seconds`, `max_tokens`, and `timeout`
overrides. Default limits are 24 MiB and 300 seconds **summed across all views**,
8,192 completion tokens, a 180-second network timeout, and no retry. A four-camera
episode therefore has a default maximum of 75 seconds per camera. Oversized
evidence fails without cropping or uploading; select a deliberate larger budget
or a separately documented clip when appropriate. These are local experiment
limits, not claimed provider limits or a fixed per-call price.

## Read the result

`audit.json` contains public prompt/instructions, request settings, SHA-256 hashes
and durations for the videos, response text, provider/model identity, usage/cost
when returned, and validated per-criterion checks. Each check has true/false/null,
a reason, and evidence with a camera name and start/end seconds. Conclusive checks
require evidence intervals inside the supplied clip. Validation checks the form
and bounds of that evidence, not whether the model described the pixels correctly.

- `agrees` / `disagrees`: a conclusive visual result compared with a trusted boolean.
- `inconclusive`: at least one criterion is unknown and none visibly failed.
- `uncompared`: a conclusive visual result with no mechanical comparison supplied.
- `judge_failed`: missing key, invalid media, budget limit, provider error, truncated
  output, wrong model, invalid schema, or invalid evidence. This is not a robot fail.

One visibly false criterion makes the conjunction false even if another is unknown.
Confidence is the model's qualitative self-report, not a calibrated probability.
The model gets no tools and no solver transcript or mechanical outcome. Failed
requests do not fall back to another model. Reports stay outside participant
feedback; giving this feedback to a solver would define a separate assisted track.

## Two experiments this makes possible

1. **Goals expressed through object relationships.** Pilot tasks such as “leave the
   mug beside the plate, with its handle accessible,” or “place all utensils in
   the tray.” A public rubric can cover arrangement, identity, counting, and
   release without first writing a dedicated geometric predicate for each
   wording. These remain experimental labels until checked against human review
   and simulator evidence where available.
2. **Process and order, beyond the final pose.** Inspect “place the bowl inside,
   release it, then close the door,” or a successful retry after an initial failed
   grasp. Timestamped checks distinguish endpoint success from following the
   requested procedure and can supply candidate event labels for replay search
   and learning analysis. Observed recovery is not itself proof of causal learning.

A first calibration should include independently labeled successes, failures,
wrong-object examples, wrong-order examples, and occluded/truncated footage. Keep
the mechanical verdict hidden, inspect disagreements, and report false positives,
false negatives, abstentions, errors, latency, and actual cost separately. Do this
before considering any judge-based scored task track. No campaign or calibration
is launched by installing this module or importing it.
