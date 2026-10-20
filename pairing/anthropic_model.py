"""Direct Anthropic Messages client for retained-learning RoboSWE campaigns.

The API key is resolved from macOS Keychain and is never written to disk or
included in diagnostics.  The client keeps the exact provider message blocks
needed for Opus adaptive-thinking continuity and enables automatic prompt
caching on every request.
"""

from __future__ import annotations

import base64
import fcntl
import gzip
import json
import mimetypes
import os
import re
import subprocess
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from pairing.context_history import POLICY, initialize, sliding_ask


API_URL = "https://api.anthropic.com/v1/messages"
WORKSPACES_URL = "https://api.anthropic.com/v1/organizations/workspaces?include_default=true&limit=100"
MODEL = "claude-opus-5-5"
PROVIDER = "anthropic-direct"
EFFORT = "high"
CACHE_TTL = "5m"
CONTEXT_WINDOW = 1_000_000
ROLLOVER_AT = 760_000
MAX_TOKENS = 32_768
PROVIDER_THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})

# Claude Opus 5.5 standard/global rates in USD per million tokens.
RATES = {
    "input_tokens": 4.0,
    "cache_creation_input_tokens": 5.0,
    "cache_read_input_tokens": 0.20,
    "output_tokens": 20.0,
}


# Official Opus5.5 standard/global cache write prices. Keep RATES' historical
# aggregate field for compatibility; actual writes are priced by receipt TTL.
# https://platform.claude.com/docs/en/models/opus-5-5/overview
CACHE_WRITE_RATES = {"5m": 5.0, "1h": 8.0}


class ModelRefusal(RuntimeError):
    """Provider explicitly refused; not a simulator infrastructure defect."""
    status = "model_refusal"


class BudgetLimitReached(RuntimeError):
    """The shared paid-inference guard cannot reserve another request."""


def credential() -> str:
    """Resolve the user-selected Anthropic key without exposing its value."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    result = subprocess.run(
        [
            "/usr/bin/security",
            "find-generic-password",
            "-s",
            "zsh_secrets",
            "-a",
            "ANTHROPIC_API_KEY",
            "-w",
        ],
        capture_output=True,
        timeout=20,
    )
    value = result.stdout.decode().strip() if result.returncode == 0 else ""
    if not value:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is unavailable in macOS Keychain; value suppressed"
        )
    return value


def resolve_workspace_id(key: str, *, name: str = "benchmarking") -> str:
    """Select the explicit active Anthropic workspace for benchmark spend."""
    request = urllib.request.Request(
        WORKSPACES_URL,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Anthropic workspace discovery HTTP {exc.code}; credential suppressed"
        ) from None
    matches = [
        row
        for row in payload.get("data") or []
        if row.get("name") == name and row.get("archived_at") is None
    ]
    if len(matches) != 1 or not str(matches[0].get("id", "")).startswith("wrkspc_"):
        raise RuntimeError(f"Expected one active Anthropic workspace named {name!r}")
    return matches[0]["id"]


def _usage_sources(usage: dict) -> list[dict]:
    iterations = usage.get("iterations")
    return iterations if isinstance(iterations, list) and iterations else [usage]


def _cache_write_counts(source: dict, *, fallback_ttl: str) -> dict[str, int]:
    """Use actual duration buckets; assign only unspecified writes to request TTL."""
    if fallback_ttl not in CACHE_WRITE_RATES:
        raise ValueError("Unsupported cache TTL for accounting")
    details = source.get("cache_creation") or {}
    if not isinstance(details, dict):
        raise ValueError("Invalid cache_creation usage breakdown")
    counts = {}
    for ttl in CACHE_WRITE_RATES:
        value = details.get(f"ephemeral_{ttl}_input_tokens") or 0
        if type(value) is not int or value < 0:
            raise ValueError("Cache creation bucket must be a nonnegative integer")
        counts[ttl] = value
    total = source.get("cache_creation_input_tokens")
    if total is None:
        total = sum(counts.values())
    if type(total) is not int or total < sum(counts.values()):
        raise ValueError("Cache creation aggregate is inconsistent with duration buckets")
    counts[fallback_ttl] += total - sum(counts.values())
    return counts


def billed_usage(usage: dict) -> dict[str, int]:
    """Sum provider iterations once; duration details can supply missing aggregate."""
    sources = _usage_sources(usage)
    return {
        field: sum(
            sum(_cache_write_counts(source, fallback_ttl="5m").values())
            if field == "cache_creation_input_tokens" else int(source.get(field) or 0)
            for source in sources
        )
        for field in RATES
    }


def request_cost(usage: dict, *, cache_ttl: str = CACHE_TTL) -> float:
    """Price actual5m/1h buckets; absent duration detail uses configured request TTL.

    Iterations already include the final iteration, so never also bill the
    top-level effective usage. Missing/partial duration fields do not silently
    price configured1h writes at the cheaper5m rate.
    """
    if cache_ttl not in CACHE_WRITE_RATES:
        raise ValueError("Unsupported cache TTL for accounting")
    normalized = billed_usage(usage)
    cost = sum(normalized[field] * rate for field, rate in RATES.items()
               if field != "cache_creation_input_tokens")
    for source in _usage_sources(usage):
        counts = _cache_write_counts(source, fallback_ttl=cache_ttl)
        cost += sum(counts[ttl] * rate for ttl, rate in CACHE_WRITE_RATES.items())
    return cost / 1_000_000


class SpendLedger:
    """Process-safe aggregate spend ledger with conservative reservations."""

    def __init__(self, path: Path | str, *, limit_usd: float = 195.0):
        self.path = Path(path)
        self.limit_usd = float(limit_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._initialize()

    def _initialize(self):
        with self.path.open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            if not stream.read().strip():
                stream.seek(0)
                json.dump(
                    {
                        "version": 1,
                        "limit_usd": self.limit_usd,
                        "spent_usd": 0.0,
                        "reserved": {},
                        "requests": 0,
                        "updated_epoch": time.time(),
                    },
                    stream,
                    indent=2,
                )
                stream.truncate()
                stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)

    def _locked(self, update):
        with self.path.open("r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.seek(0)
            data = json.load(stream)
            if abs(float(data["limit_usd"]) - self.limit_usd) > 1e-9:
                raise RuntimeError("Shared Anthropic spend limit changed")
            result = update(data)
            data["updated_epoch"] = time.time()
            stream.seek(0)
            json.dump(data, stream, indent=2)
            stream.truncate()
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)
            return result

    def reserve(self, amount_usd: float, *, label: str) -> str:
        reservation = str(uuid.uuid4())

        def update(data):
            committed = float(data["spent_usd"])
            held = sum(float(value["amount_usd"]) for value in data["reserved"].values())
            if committed + held + amount_usd > self.limit_usd + 1e-9:
                data["limit_reached"] = True
                raise BudgetLimitReached(
                    f"Anthropic campaign spend guard reached ${self.limit_usd:.2f}"
                )
            data["reserved"][reservation] = {
                "amount_usd": amount_usd,
                "label": label,
                "created_epoch": time.time(),
            }
            return reservation

        return self._locked(update)

    def commit(self, reservation: str, actual_usd: float, *, usage: dict):
        def update(data):
            held = data["reserved"].pop(reservation)
            if actual_usd > float(held["amount_usd"]) + 1e-9:
                raise RuntimeError("Provider request exceeded its conservative reservation")
            data["spent_usd"] = float(data["spent_usd"]) + actual_usd
            data["requests"] = int(data.get("requests") or 0) + 1
            data["last_usage"] = billed_usage(usage)
            data["last_effective_usage"] = {
                key: int(usage.get(key) or 0) for key in RATES
            }
            data["last_cost_usd"] = actual_usd

        self._locked(update)

    def release(self, reservation: str):
        def update(data):
            data["reserved"].pop(reservation, None)

        self._locked(update)

    def snapshot(self) -> dict:
        with self.path.open() as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            data = json.load(stream)
            fcntl.flock(stream, fcntl.LOCK_UN)
        data["reserved_usd"] = sum(
            float(value["amount_usd"]) for value in data["reserved"].values()
        )
        data["remaining_unreserved_usd"] = max(
            0.0,
            float(data["limit_usd"]) - float(data["spent_usd"]) - data["reserved_usd"],
        )
        return data


class AnthropicLearningModel:
    """Persistent Opus 5.5 conversation with strict structured decisions."""

    def __init__(
        self,
        instructions: str,
        directory: Path | str,
        *,
        effort: str = EFFORT,
        context_window: int = CONTEXT_WINDOW,
        compact_at: int = ROLLOVER_AT,
        cache_ttl: str = CACHE_TTL,
        max_tokens: int = MAX_TOKENS,
        ledger_path: Path | str | None = None,
        spend_limit_usd: float = 195.0,
        workspace_id: str | None = None,
        workspace_name: str | None = "benchmarking",
        context_policy: str = POLICY,
        token_counter=None,
    ):
        if not 1_000 <= compact_at < context_window:
            raise ValueError("Invalid Anthropic context policy")
        if cache_ttl not in {"5m", "1h"}:
            raise ValueError("Invalid prompt-cache TTL")
        self.instructions = instructions
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key = credential()
        self.workspace_name = workspace_name
        self.workspace_id = (
            workspace_id
            or os.environ.get("ROBOSWE_ANTHROPIC_WORKSPACE_ID")
            or (resolve_workspace_id(self.key, name=workspace_name) if workspace_name else None)
        )
        if self.workspace_id is not None and not self.workspace_id.startswith("wrkspc_"):
            raise ValueError("Invalid Anthropic workspace ID")
        self.effort = effort
        self.context_window = context_window
        self.compact_at = compact_at
        self.cache_ttl = cache_ttl
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.calls = 0
        self.compactions = 0
        self.rollovers = 0
        self.latest_context_tokens = 0
        self.last_effective_usage = {key: 0 for key in RATES}
        self.last_request_started_epoch: float | None = None
        self.last_provider_thinking_trace: dict | None = None
        self.cumulative_usage = {key: 0 for key in RATES}
        self.cumulative_cost_usd = 0.0
        self.thread_id = f"anthropic-opus55-{uuid.uuid4()}"
        self.ledger = SpendLedger(ledger_path, limit_usd=spend_limit_usd) if ledger_path else None
        self.metadata = {
            "model": MODEL,
            "model_provider": PROVIDER,
            "api_url": API_URL,
            "reasoning_effort": effort,
            "adaptive_thinking": "always_on",
            "context_window": context_window,
            "compact_at": compact_at,
            "max_output_tokens": max_tokens,
            "structured_output": "output_config.format json_schema",
            "prompt_caching": {
                "mode": "automatic",
                "ttl": cache_ttl,
                "usage_fields_required": [
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ],
            },
            "credential_source": (
                "macOS Keychain service=zsh_secrets account=ANTHROPIC_API_KEY; "
                "host memory only"
            ),
            "workspace": {"id": self.workspace_id, "name": workspace_name},
            "thread_id": self.thread_id,
            "persistent_thread": True,
            "conversation_transport": "local exact Messages API block replay",
            "provider_exposed_thinking": {
                "artifact": "provider-thinking.jsonl",
                "block_types": sorted(PROVIDER_THINKING_BLOCK_TYPES),
                "policy": (
                    "Preserve only thinking blocks returned by the provider, verbatim; "
                    "do not infer or reconstruct unavailable internal reasoning."
                ),
            },
            "local_context_rollover": {
                "trigger_effective_tokens": compact_at,
                "policy": (
                    "Before the provider context reaches its limit, archive the exact message "
                    "history locally and begin a small new Messages context. The continuity "
                    "envelope contains only deterministic excerpts of Opus's own recent public "
                    "decisions; the persistent code workspace and NOTES.md remain authoritative."
                ),
                "provider_compaction": (
                    "disabled after live evidence showed Anthropic compact-2026-01-12 billed "
                    "a cold full-context cache creation"
                ),
            },
            "fallback_models": [],
            "pricing_usd_per_million_tokens": {**RATES, "cache_creation_input_tokens": CACHE_WRITE_RATES[cache_ttl]},
            "cache_write_prices_by_ttl_usd_per_million": dict(CACHE_WRITE_RATES),
            "shared_spend_limit_usd": spend_limit_usd if ledger_path else None,
        }

        initialize(self, context_policy=context_policy, context_window=context_window,
                   output_reserve=max_tokens, history_attribute="messages", token_counter=token_counter or self._count_history_tokens)

    def _count_history_tokens(self, *, messages, pending, overhead):
        """Provider token count for the same multimodal/system/schema input.

        Count-only endpoint: no model completion, no compaction. It still uses
        the active task deadline and records latency separately from inference.
        """
        body = {
            "model": MODEL,
            "system": [{"type": "text", "text": overhead["instructions"],
                        "cache_control": {"type": "ephemeral", "ttl": self.cache_ttl}}],
            "messages": [*messages, {"role": "user", "content": self._content(pending["text"], pending["images"])}],
            "output_config": {"effort": self.effort,
                              "format": {"type": "json_schema", "schema": overhead["schema"]}},
            "cache_control": {"type": "ephemeral", "ttl": self.cache_ttl},
        }
        started = time.monotonic()
        deadline = getattr(self, "_history_deadline", started + 180)
        if deadline <= started:
            raise TimeoutError("Token counting exhausted the active task deadline")
        request = urllib.request.Request(
            API_URL + "/count_tokens", data=json.dumps(body, separators=(",", ":")).encode(),
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                     **({"anthropic-workspace-id": self.workspace_id} if self.workspace_id else {}), "content-type": "application/json"})
        for attempt in range(1, 3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Token counting exhausted the original active task deadline")
            timeout = min(90, remaining)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.load(response)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                transient = not isinstance(exc, urllib.error.HTTPError) or exc.code in {429, 500, 502, 503, 504, 529}
                delay = 1.0
                if isinstance(exc, urllib.error.HTTPError) and exc.headers:
                    try:delay = max(0.0, float(exc.headers.get("retry-after", "1")))
                    except ValueError:delay = 1.0
                retry = transient and attempt < 2 and delay <= 30 and deadline - time.monotonic() > delay
                diagnostic = self._transport_diagnostic(exc)
                self._write_request_event({"operation":"count_tokens", "attempt":attempt,
                    "seconds":time.monotonic()-started, "attempt_timeout_seconds":timeout,
                    "original_deadline_seconds_remaining":max(0, deadline-time.monotonic()),
                    "will_retry":retry, "retry_delay_seconds":delay if retry else None,
                    "completion_generated":False, "error":diagnostic})
                if retry:
                    time.sleep(delay)
                    continue
                if deadline <= time.monotonic():
                    raise TimeoutError("Token counting exhausted the original active task deadline") from RuntimeError(json.dumps(diagnostic))
                raise RuntimeError("Anthropic count-only request failed; no inference attempted; see provider-request-events.jsonl") from RuntimeError(json.dumps(diagnostic))
            count = payload.get("input_tokens")
            if type(count) is not int or count < 0:
                self._write_request_event({"operation":"count_tokens", "attempt":attempt,"completion_generated":False,
                    "will_retry":False,"error":{"type":"InvalidTokenCount"}})
                raise RuntimeError("Anthropic token counter returned invalid input_tokens")
            with (self.directory / "token-counts.jsonl").open("a") as stream:
                stream.write(json.dumps({"input_tokens":count, "seconds":time.monotonic()-started,
                    "messages":len(body["messages"]), "attempts":attempt, "completion_generated":False,
                    "retry_policy":"count-only; at most2 attempts of90s within original episode deadline"}) + "\n")
            return count

    def _transport_diagnostic(self, exc):
        """Preserve real exception categories/reasons without headers or credentials."""
        def clean(value):
            text = str(value)
            if self.key:text = text.replace(self.key, "[REDACTED]")
            return re.sub(r"(?:sk-[A-Za-z0-9_-]{10,}|AIza[A-Za-z0-9_-]{20,})", "[REDACTED]", text)[:1000]
        result = {"type":type(exc).__name__, "message":clean(exc)}
        reason = getattr(exc, "reason", None)
        if reason is not None:
            result.update(reason_type=type(reason).__name__, reason=clean(reason))
            if isinstance(getattr(reason,"errno",None),int):result["errno"]=reason.errno
        if isinstance(exc, urllib.error.HTTPError):
            result["http_status"]=exc.code
            if exc.headers:result["request_id"]=clean(exc.headers.get("request-id", ""))
        return result

    def _write_request_event(self, event):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "provider-request-events.jsonl").open("a") as stream:
            stream.write(json.dumps({"epoch":time.time(), **event}) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.key = ""

    @staticmethod
    def _content(text: str, images) -> list[dict]:
        content = [{"type": "text", "text": text}]
        for image in images:
            path = Path(image)
            media_type = mimetypes.guess_type(path.name)[0]
            if media_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
                raise ValueError(f"Unsupported trusted image format: {path.name}")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    },
                }
            )
        return content

    def _reserve_amount(self, max_tokens: int) -> float:
        # The full context at the configured TTL's worst-case cache-write rate plus
        # the requested output ceiling.  This prevents concurrent lanes from
        # collectively crossing the user-funded campaign cap.
        input_rate = max(RATES["input_tokens"], CACHE_WRITE_RATES[self.cache_ttl])
        return self.context_window * input_rate / 1_000_000 + max_tokens * RATES["output_tokens"] / 1_000_000

    def _request(self, body: dict, *, timeout: float, purpose: str) -> dict:
        # Inference is never retried. Count-only recovery is separate above.
        if timeout <= 0:
            raise TimeoutError("Anthropic request deadline expired before transmission")
        started = time.monotonic()
        deadline = min(started+timeout, getattr(self, "_history_deadline", started+timeout))
        if deadline <= started:
            raise TimeoutError("Anthropic original request deadline expired before transmission")
        reservation = self.ledger.reserve(self._reserve_amount(int(body["max_tokens"])), label=purpose) if self.ledger else None
        request = urllib.request.Request(API_URL, data=json.dumps(body,separators=(",",":")).encode(),
            headers={"x-api-key":self.key,"anthropic-version":"2023-06-01",
                **({"anthropic-workspace-id":self.workspace_id} if self.workspace_id else {}),"content-type":"application/json"})
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            if reservation:self.ledger.release(reservation)
            self._write_request_event({"operation":"inference","phase":"cancelled_before_transmission", "reservation":reservation,
                "reservation_released":bool(reservation),"retry":False})
            raise TimeoutError("Anthropic deadline expired before network transmission; no inference sent")
        self._write_request_event({"operation":"inference","phase":"transmission_started","reservation":reservation,
            "purpose":purpose,"attempt":1,"retry":False})
        try:
            with urllib.request.urlopen(request,timeout=remaining) as response:
                payload=json.load(response)
        except BaseException as exc:
            diagnostic=self._transport_diagnostic(exc)
            self._write_request_event({"operation":"inference","phase":"unsettled_failure","reservation":reservation,
                "reservation_retained":bool(reservation),"seconds":time.monotonic()-started,"retry":False,"error":diagnostic})
            if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
            message = f"Anthropic HTTP {exc.code}" if isinstance(exc,urllib.error.HTTPError) else "Anthropic inference transport/response failure"
            raise RuntimeError(message + "; no retry; unknown spend reservation retained") from RuntimeError(json.dumps(diagnostic))
        try:
            usage=payload.get("usage")
            if not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]<0 for k in ("input_tokens","output_tokens")):
                raise ValueError("Missing valid Anthropic usage receipt")
            cost=request_cost(usage,cache_ttl=self.cache_ttl)
            if reservation:self.ledger.commit(reservation,cost,usage=usage)
        except Exception as exc:
            # Save the received message even when its billing metadata is unusable.
            encoded=json.dumps({"reservation":reservation,"response":payload})
            if self.key:encoded=encoded.replace(self.key,"[REDACTED]")
            with (self.directory/"provider-unsettled-responses.jsonl").open("a") as stream:stream.write(encoded+"\n")
            self._write_request_event({"operation":"inference","phase":"unsettled_usage","reservation":reservation,
                "reservation_retained":bool(reservation),"retry":False,"error":self._transport_diagnostic(exc)})
            raise RuntimeError("Anthropic usage settlement failed; reservation retained; no retry") from RuntimeError(json.dumps(self._transport_diagnostic(exc)))
        self._write_request_event({"operation":"inference","phase":"settled","reservation":reservation,"cost_usd":cost,
            "response_id":payload.get("id"),"retry":False})
        self._record(payload,purpose=purpose,elapsed_seconds=time.monotonic()-started,cost=cost)
        return payload

    def _record(self, payload: dict, *, purpose: str, elapsed_seconds: float, cost: float):
        self.calls += 1
        usage = payload.get("usage") or {}
        normalized = billed_usage(usage)
        for key, value in normalized.items():
            self.cumulative_usage[key] += value
        self.cumulative_cost_usd += cost
        # Top-level usage is the final, effective message iteration.  Keep it
        # separate from billed_usage(), which may include provider iterations.
        effective = {key: int(usage.get(key) or 0) for key in RATES}
        self.last_effective_usage = effective
        self.last_request_started_epoch = time.time() - elapsed_seconds
        self.latest_context_tokens = sum(
            effective[key]
            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "output_tokens",
            )
        )
        receipt = {
            "call": self.calls,
            "purpose": purpose,
            "elapsed_seconds": elapsed_seconds,
            "id": payload.get("id"),
            "model": payload.get("model"),
            "stop_reason": payload.get("stop_reason"),
            "usage": normalized,
            "cost_usd": cost,
            "cumulative_usage": dict(self.cumulative_usage),
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "cache_hit": normalized["cache_read_input_tokens"] > 0,
        }
        thinking_blocks = [
            {"content_block_index": index, "block": block}
            for index, block in enumerate(payload.get("content") or [])
            if isinstance(block, dict)
            and block.get("type") in PROVIDER_THINKING_BLOCK_TYPES
        ]
        thinking_trace = {
            "schema_version": 1,
            "call": self.calls,
            "purpose": purpose,
            "provider_response_id": payload.get("id"),
            "model": payload.get("model"),
            "stop_reason": payload.get("stop_reason"),
            "elapsed_seconds": elapsed_seconds,
            "provider_exposed_block_count": len(thinking_blocks),
            "plaintext_thinking_available": any(
                isinstance(item["block"].get("thinking"), str)
                and bool(item["block"]["thinking"])
                for item in thinking_blocks
            ),
            "blocks": thinking_blocks,
        }
        self.last_provider_thinking_trace = thinking_trace
        with (self.directory / "provider-responses.jsonl").open("a") as stream:
            stream.write(json.dumps({**receipt, "response": payload}) + "\n")
        with (self.directory / "provider-thinking.jsonl").open("a") as stream:
            stream.write(json.dumps(thinking_trace, ensure_ascii=False) + "\n")
        with (self.directory / "model-usage.jsonl").open("a") as stream:
            stream.write(json.dumps(receipt) + "\n")

    def usage_summary(self) -> dict:
        return {
            "cumulative_token_usage": dict(self.cumulative_usage),
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "reported_context_windows": [self.context_window],
            "latest_context_tokens": self.latest_context_tokens,
            "compactions": self.compactions,
            "local_context_rollovers": self.rollovers,
            "cache_ttl": self.cache_ttl,
        }

    def save_context(self, path: Path | str) -> dict:
        raw = json.dumps(self.messages, ensure_ascii=False, separators=(",", ":")).encode()
        path = Path(path)
        with gzip.open(path, "wb") as stream:
            stream.write(raw)
        import hashlib

        return {
            "file": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "messages": len(self.messages),
        }

    def _rollover_context(self) -> str:
        """Archive exact history and return a small, model-authored continuity envelope.

        This is deliberately not an LLM summary.  It extracts only structured
        decisions that Opus already returned, so no other model contributes task
        knowledge and Anthropic is not asked to reread the full context.
        """
        number = self.rollovers + 1
        archive = self.directory / f"context-before-rollover-{number}.json.gz"
        checkpoint = self.save_context(archive)
        decisions = []
        for message in self.messages:
            if message.get("role") != "assistant":
                continue
            text_blocks = [
                block.get("text", "")
                for block in message.get("content") or []
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if len(text_blocks) != 1:
                continue
            try:
                decision = json.loads(text_blocks[0])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(decision, dict):
                continue
            decisions.append(
                {
                    "operation": str(decision.get("operation") or "")[:40],
                    "path": str(decision.get("path") or "")[:240],
                    "message": str(decision.get("message") or "")[:1200],
                }
            )
        recent = decisions[-12:]
        envelope = {
            "context_rollover": number,
            "notice": (
                "The complete prior provider transcript was archived by the trusted runner to "
                f"{archive.name}; it is not available through the robot workspace. No simulator "
                "reset or action occurred during rollover. Continue the same task and current "
                "attempt. Your Python workspace and NOTES.md persist unchanged. Treat NOTES.md "
                "and the fresh observation below as authoritative; read NOTES.md before further "
                "physical action if the immediate continuation is not already unambiguous."
            ),
            "recent_opus_public_decisions": recent,
        }
        envelope_text = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        continuity_path = self.directory / f"context-rollover-{number}.json"
        continuity_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n")
        self.messages = []
        self.rollovers = number
        self.latest_context_tokens = 0
        with (self.directory / "context-rollovers.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "number": number,
                        "checkpoint": checkpoint,
                        "continuity": continuity_path.name,
                        "recent_decisions": len(recent),
                        "provider_request": False,
                        "provider_cost_usd": 0.0,
                    }
                )
                + "\n"
            )
        return envelope_text

    @staticmethod
    def _parse_decision(payload: dict, schema: dict) -> dict:
        if payload.get("model") != MODEL:
            raise RuntimeError("Anthropic returned a substituted model")
        if payload.get("stop_reason") == "refusal":
            raise ModelRefusal("Anthropic explicitly refused the request; no robot action executed")
        text_blocks = [
            block.get("text", "")
            for block in payload.get("content") or []
            if block.get("type") == "text"
        ]
        if len(text_blocks) != 1:
            raise RuntimeError("Opus returned an unexpected structured-output block count")
        try:
            decision = json.loads(text_blocks[0])
        except json.JSONDecodeError:
            raise RuntimeError("Opus structured output was not valid JSON") from None
        required = set(schema["required"])
        if not isinstance(decision, dict) or set(decision) != required:
            raise RuntimeError("Opus returned an invalid robot-operation object")
        if not all(isinstance(value, str) for value in decision.values()):
            raise RuntimeError("Opus returned non-string robot-operation fields")
        if decision["operation"] not in schema["properties"]["operation"]["enum"]:
            raise RuntimeError("Opus returned an unsupported robot operation")
        return decision

    @sliding_ask
    def ask(
        self,
        text: str,
        *,
        images=(),
        schema: dict | None = None,
        timeout: float = 180,
        as_tool_output: bool = False,
    ) -> dict:
        del as_tool_output  # Structured output, not tool transport.
        if schema is None:
            # Imported lazily to avoid a module cycle; the cohort extension adds
            # write_execute to the base wait schema.
            from pairing.cohort50 import SCHEMA

            schema = SCHEMA
        if self.history_store is None and self.latest_context_tokens >= self.compact_at:
            rollover = self._rollover_context()
            text = (
                "TRUSTED LOCAL CONTEXT ROLLOVER (deterministic, no model summary):\n"
                + rollover
                + "\n\nCURRENT AUTHORITATIVE FEEDBACK:\n"
                + text
            )
        pending = {"role": "user", "content": self._content(text, images)}
        body = {
            "model": MODEL,
            "max_tokens": self.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": self.instructions,
                    "cache_control": {"type": "ephemeral", "ttl": self.cache_ttl},
                }
            ],
            "messages": [*self.messages, pending],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            "cache_control": {"type": "ephemeral", "ttl": self.cache_ttl},
        }
        payload = self._request(body, timeout=timeout, purpose="decision")
        decision = self._parse_decision(payload, schema)
        self.messages.extend(
            [pending, {"role": "assistant", "content": payload.get("content") or []}]
        )
        return decision


def diagnostic_trace() -> str:
    """Return a compact traceback for caller-owned artifact logs."""
    return traceback.format_exc()
