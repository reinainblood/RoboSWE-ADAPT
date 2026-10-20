# Retained history and sliding context

RoboSWE-ADAPT preserves experience as exact interaction records and authored code.
A task series keeps one conversation and code workspace across five physical
resets. No generated summary replaces the earlier history. The active request
window may slide when it approaches the configured context limit; archived turns
remain available through bounded retrieval.

The public [adaptive runner](adaptive-cli.md) selects a provider adapter through
`pairing.direct_models.create_participant`. Existing historical campaigns retain
their legacy policy and are not retroactively relabeled as adaptive results.

## Archive and retrieval

The coordinator-owned `history.sqlite` stores exact JSON provider message blocks,
SHA-256 digests, timestamps, and attempt identifiers. Provider-returned thinking,
signatures, opaque replay blocks, and exposed reasoning are preserved where the
transport supplies them. Opaque fields remain opaque; the dashboard does not
reconstruct hidden reasoning.

The live store belongs to the series. Each successful client turn and exceptional
exit exports a consistent SQLite backup into the current attempt directory.
Authored robot code cannot write the archive. These exports are local durable
records. Modal finalization additionally archives them with hash-readback
verification; Docker jobs can use the explicit backup command below. Opening an
existing store does not yet restore a crashed series's active conversation.

A stable instruction advertises `history_search`. Before the first slide, the
operation returns disabled. Afterward it performs literal substring matching,
not agent-supplied SQL, and returns at most five excerpts of 2,400 characters each
(12,000 total). Up to four inference calls per robot decision share the original
deadline. Retrieval does not advance physics.

## Window rule

For configured context capacity `C`, output reserve `O`, and framing allowance
`S = 4096`, the input budget is `C - O - S`. Instructions, response schema,
images, tool envelopes, and conversational content participate in admission.

When the request exceeds that budget, the oldest complete user-led exchanges
are removed until at least 200,000 input accounting units have been removed, or
more if needed to fit. Exchanges can be larger than the nominal chunk. System
instructions remain pinned, and native tool calls are never separated from their
results. A single indivisible exchange that cannot fit stops the request rather
than being summarized or silently truncated.

The SQLite `slides` table records the attempt, timestamp, removed-message count,
and before/after accounting values. Interpret those values with the run's context
policy: they are native counts under native-count adapters and proxy units under
the conservative OpenRouter adapter. Some historical column names contain
`estimated_`; the name alone does not identify the counting method.

## Native-count policy: `sqlite-sliding-v1`

| Provider | Count request | History transport |
| --- | --- | --- |
| OpenAI | `/v1/responses/input_tokens` | Stateless Responses replay; storage and automatic truncation disabled |
| Anthropic | `/v1/messages/count_tokens` | Matching Messages payload, including images, system instructions, cache controls, and output schema |
| Google | `countTokens` | Shared request builder for counting and native Gemini generation |

Adapters count the request format used for generation. Counts are request-budget
measurements, not a promise of identical billing values. Count latency is recorded
and charged against the active decision deadline. The sliding search bounds count
requests and checks the retained suffix again before committing a slide.

The OpenAI client preserves returned output items, including exposed summaries
and encrypted reasoning for replay, and does not use `previous_response_id` or
provider compaction. Google preserves returned content and thought signatures.
Model IDs and configured input/output limits are pinned in each run rather than
assumed from a model-family name.

Counting availability is checked independently of generation quota and is not
an actual robot-run result. The supported native API adapters
are separate from Codex managed threads: the app-server integration does not
support this exact-window policy and rejects it rather than silently compacting.

## Anthropic signed thinking and readable summaries

Opus 5.5 can bind returned thinking blocks to the earlier request prefix.
Anthropic documents that accounts enforcing this binding can reject a replay
when that prefix changes. A local window slide changes the prefix, so token-count
availability alone does not establish generation compatibility after a slide.
See [Anthropic's preserved-thinking contract](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking).

`sqlite-sliding-v1` preserves returned blocks and signatures unchanged. It does
not request `drop_block` or silently remove retained thinking to avoid a provider
error. A rejected request remains an infrastructure interruption with its history
preserved. The development campaign has recorded a successful post-slide Opus
response on its configured account; that observation is not a guarantee for every
account or provider version.

The optional `thinking_display: "summarized"` setting controls readable output,
not prefix-binding behavior. Anthropic supplies a provider-generated thinking
summary, which the viewer labels accordingly. It is not a generated replacement
for the archived conversation. The chosen display mode is recorded and held
constant across a five-trial series.

## OpenRouter policy: `sqlite-sliding-or-conservative-v1`

OpenRouter's post-response token usage does not forecast the next multimodal
request. The experimental OpenRouter adapter therefore offers an explicit
conservative policy instead of claiming an exact pre-request count:

```text
base units = serialized UTF-8 text bytes
           + 16,384 per image
           + 8,192 fixed units
admission units = ceil(base units × calibration multiplier)
```

Image data is replaced by an image marker for the text-byte calculation. The
response schema is included. The multiplier starts at one and only increases;
after a response it is raised if necessary to cover 1.25 times the reported
prompt-token count relative to that request's base units.

Each `context-estimates.jsonl` receipt records the estimated input units, actual
prompt tokens, multiplier before and after, whether the estimate was low, and
`exact_native_count: false`. The proxy is not a proven native tokenizer bound.
Overflow stops execution with evidence retained; it does not trigger a silent
retry, smaller output limit, or provider substitution. Provider context compression
and model fallbacks are disabled.

Select this policy explicitly in the participant configuration. Public CLI runs
also require pinned prices, a shared spend ledger, and a spend cap. Spending
reservations use the full configured model window and output allowance, not the
unproven proxy. Actual usage/cost receipts settle known requests; uncertain
outcomes retain their reservation. See the [CLI guide](adaptive-cli.md).

## Calibrated Google counting for OpenRouter Argon

`sqlite-sliding-or-google-count-v2` replaces the byte/image proxy for prospective
Argon series. It uses a count-only Google request reconstructed from the
OpenRouter text, images, response schema, and returned reasoning metadata.
Generation still sends the original OpenRouter messages unchanged.

The estimate starts at `ceil(native_count × 1.30) + 8192`. After a response, the
multiplier can increase to `1.1 × actual_prompt_tokens / native_count`; it never
decreases within the series. This is a calibrated estimate, not an exact count of
OpenRouter's undisclosed upstream request. In bounded historical-request checks,
the mapped native counter undercounted longer requests by roughly 9–14%; the
margin covered the held-out examples. That evidence is not a proof for every
future request.

Counting requires `GEMINI_API_KEY` in the trusted coordinator as well as
`OPENROUTER_API_KEY` for inference. Unknown message metadata, count failures, and
expired counting deadlines stop the run rather than switching to another
estimator. Receipts include the transformed-request hash, native count, estimate,
actual usage, and calibration update.

The v1 development series exposed premature slides at much smaller actual prompt
sizes and was stopped after its first independently verified trial. Its evidence
remains separate; a v2 series starts with a fresh history and code workspace.

## Comparing policies

All policies archive exact history and slide whole exchanges without summaries.
They do not necessarily retain the same amount of active context: a conservative
proxy can cause an earlier slide. Reports must identify the model, provider route,
context policy, configured limits, and slide events. Sliding changes the request
prefix, so caching discounts depend on the provider and are not guaranteed.

This protocol measures adaptation of the retained agent system: code and history
are both available. A matched fresh-context control would be needed to isolate a
causal contribution of conversational retention. No such control arm is silently
added by the adaptive runner.

## Evidence and checks

Offline checks cover whole-exchange eviction, output reserve and overhead,
200,000-unit removal, oversized requests, retrieval gating and bounds, SQLite
integrity/export, and provider message preservation. The Docker development path
has additionally completed reset preflight and live OpenRouter execution. Those
checks do not substitute for completed five-trial evaluation results.

Primary API references:

- [OpenAI token counting](https://developers.openai.com/api/docs/guides/token-counting)
- [OpenAI reasoning replay](https://developers.openai.com/api/docs/guides/reasoning)
- [Anthropic Messages counting](https://platform.claude.com/docs/en/api/messages/count_tokens)
- [Google token counting](https://ai.google.dev/api/tokens)
- [OpenRouter OpenAPI](https://openrouter.ai/openapi.json)

## Remote evidence backups

The optional `pairing.adapt_artifact_archive` command archives an explicit set of
job evidence files and a consistent SQLite backup to the existing
`roboswe-job-artifacts-v1` Modal volume in **FDR**. Select the authenticated profile
explicitly:

```sh
MODAL_ENVIRONMENT=FDR python -m pairing.adapt_artifact_archive \
  jobs/JOB_ID --profile YOUR_PROFILE
```

For a running job, add `--checkpoint`. This produces a labeled checkpoint; it is
not finalized trial evidence or a crash-resume certificate. SQLite uses its backup
API, while the surrounding files are not one atomic database transaction.

The command addresses archives by SHA-256, reads the full remote archive back,
checks its bytes and member hashes, and validates a restored SQLite database.
Only then does it write a local receipt with a profile-pinned restore command.
Restore into a new directory. Local files remain in place, and the separate
model-image archive watcher is unaffected. Images and videos have their own
artifact workflow and are not included in this small evidence archive.
