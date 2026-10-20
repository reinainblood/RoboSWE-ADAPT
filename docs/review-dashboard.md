# Evidence review dashboard

Start from the repository root:

```sh
.venv/bin/python -m roboswe.review --port 8876
```

To enable the local paid audit button, provide `OPENROUTER_API_KEY` in the trusted
server environment and select an explicit shared audit ledger and total USD cap:

```sh
.venv/bin/python -m roboswe.review --port 8876 --enable-judge \
  --judge-backend gemini --judge-ledger artifacts/visual-audit-spend.json \
  --judge-spend-cap 10
```

Paid audits remain disabled by default. Once enabled, Gemini is the default backend:
exact `google/gemini-4-argon` through `google-vertex/global`, with low reasoning
and an 8192-token output allowance. `--judge-backend glimpse` explicitly selects
`elorian/glimpse` through Together with the same output allowance. Neither route
allows provider fallback. The server owns this configuration; the browser cannot
choose a model, provider, cap, or credential.

Before each request the ledger reserves the full model context at the checked
input price plus the output allowance: $2.179072 for Gemini, or $0.1818624 for
Glimpse. Provider-reported cost settles the reservation even when the answer is
refused or fails the judgment schema. A transport failure or missing cost retains
the reservation for reconciliation, with no automatic retry. Existing ledger
caps cannot be changed by restarting the server. Sanitized provider responses
are saved beside the ledger under its `-receipts` directory before judgment
parsing. Prices are recorded in `roboswe/library/judge_budget.py` from OpenRouter's
public model metadata checked on 2026-10-20; recheck them before a later campaign.

The key never enters browser JavaScript or
the robot code container. The server binds only to loopback. The POST endpoint
requires a same-origin custom header, validates Host/Origin and job identifiers,
accepts finalized jobs only, and permits one audit at a time. It cannot execute
robot submissions. Unknown/inherited predicates without an explicit source
contract fail closed. Audit exceptions are displayed; no automatic retry.

The page shows recorded job status (not proof of process liveness), local camera
videos, observed mechanical and replay-verified results separately, predicate
sources, timestamped visual checks, provider-reported usage/cost, and decisions
beside provider-exposed reasoning. Old Anthropic series receipts are matched to
turns only when the exact final decision has a unique match. Summaries without a
reliable turn mapping stay separate. Hidden thinking is never reconstructed.

Two exports are available: `roboswe-episode-v1` retains original journal events,
model inputs/decisions, available reasoning, timings and control actions; the ATIF
projection follows v1.6 field structure. Full Harbor-schema validation remains
pending; the projection is not a replacement for the native evidence export.

A shareable static bundle can contain selected episodes:

```sh
.venv/bin/python -m roboswe.review --export artifacts/review-share \
  --job 20261009T221300Z-0daa0ac68c \
  --job 20261009T215453Z-7260a07536
python3 -m http.server 8877 --bind 127.0.0.1 --directory artifacts/review-share
```

The exported snapshot includes snapshot.js, so index.html also works when opened directly without a server. Host the exported folder on an approved static host for an external URL. The
static snapshot disables paid actions and has no credential. It contains selected
agent transcripts and videos, so review its contents before publication. The
local live dashboard has NOT been exposed publicly. Source style reference:
[Harbor Viewer](https://github.com/harbor-framework/harbor/tree/main/apps/viewer)
(light monochrome, square borders, dense evidence layout); no Harbor source assets
or branding were redistributed.

## Initial Glimpse experiment, 2026-10-20

Three calls, no simulator reruns: successful Opus blender attempt with 4096 then
16384 output-token caps both returned `finish_reason=length`; failed first attempt
with 16384 returned valid timestamped checks agreeing with the mechanical failure.
Provider-reported costs: $0.0700752, $0.0550992, $0.0817944 ($0.2069688 total).
These are availability/format observations, not calibration accuracy. At that stage the success
case remained unjudged. Preserve all receipts under each job's `predicate-audits/`.

The predicate-specific auditor compares visible conditions and reports supported,
contradicted, not observable, or judge failed. A contradiction of a condition is
not automatically disagreement with the mechanical verdict (both may say failure).
A numerical visual support fraction is only emitted when all conditions have a
conclusive visual judgment; it never overwrites the benchmark score. Exact
geometric tolerance, hidden on/off state, and sustained physics ticks cannot be
certified merely from video. Predicate source is read from the current pinned
checkout and hashed; historical source equivalence still needs independent audit.

## Gemini fallback validation

Gemini Argon subsequently returned valid visual judgments on the historical
positive and negative blender episodes, agreeing with both mechanical outcomes.
Their provider-reported costs were $0.060054 and $0.03909 ($0.099144 total).
This is a two-example format and agreement check, not an accuracy estimate.
The dashboard distinguishes **consistent failure** from **visual mechanical
disagreement**: a visible unmet condition agrees with a mechanical failure and
is not evidence of a lying verifier. Historical records receive this comparison
as a read-only display projection; their original audit files remain unchanged.
