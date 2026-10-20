"""Zero-knowledge SPARSIE model client for RoboSWE campaigns.

The client never receives the SPARSIE credential. AgentSecrets injects it into
the one allowed HTTPS destination and returns only the HTTP response body.
"""

from __future__ import annotations

import base64
import gzip
import json
import select
import subprocess
import time
import uuid
from pathlib import Path
from pairing.context_history import POLICY, initialize, sliding_ask, with_search

from pairing.simulation_wait import WAIT_SCHEMA


SPARSIE_URL = "https://sparsie-api-production.up.railway.app"
LANE = "fable51-claude-oauth"
MODEL = "claude-fable-5-1"
PROVIDER = "claude-subscription"
TOOL_NAME = "submit_robot_operation"


class AgentSecretsMCP:
    """Minimal stdio MCP client for AgentSecrets' zero-knowledge api_call."""

    def __init__(self, executable: str = "agentsecrets"):
        self.executable = executable
        self.process = None
        self.sequence = 0

    def __enter__(self):
        self.process = subprocess.Popen(
            [self.executable, "mcp", "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "roboswe-sparsie", "version": "0.1"},
            },
            timeout=15,
        )
        if result.get("serverInfo", {}).get("name") != "AgentSecrets":
            raise RuntimeError("Unexpected zero-knowledge credential proxy")
        self._send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None

    def _send(self, value):
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("AgentSecrets MCP is not running")
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def _receive(self, request_id, timeout):
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("AgentSecrets MCP is not running")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("SPARSIE request exceeded the model deadline")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError("SPARSIE request exceeded the model deadline")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("AgentSecrets MCP closed unexpectedly")
            event = json.loads(line)
            if event.get("id") != request_id:
                continue
            if event.get("error"):
                raise RuntimeError(f"AgentSecrets MCP error: {event['error']}")
            return event.get("result", {})

    def _rpc(self, method, params, *, timeout):
        self.sequence += 1
        request_id = self.sequence
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        return self._receive(request_id, timeout)

    def call(self, name, arguments, *, timeout):
        result = self._rpc(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=timeout,
        )
        text = "\n".join(
            item.get("text", "")
            for item in result.get("content", [])
            if item.get("type") == "text"
        )
        if not text:
            raise RuntimeError("AgentSecrets returned no API response")
        return text


class SparsieLearningModel:
    """Persistent local conversation over SPARSIE's provider-only Fable lane."""

    def __init__(
        self,
        *,
        instructions: str,
        directory: Path,
        context_window: int = 1_000_000,
        compact_at: int = 800_000,
        lane: str = LANE,
        model: str = MODEL,
        context_policy: str = "legacy",
        token_counter=None,
    ):
        if not 1000 <= compact_at < context_window:
            raise ValueError("Invalid SPARSIE context policy")
        self.instructions = instructions
        transport_instruction = (
            "\n\nTransport contract: every response MUST invoke the provided "
            f"{TOOL_NAME} function exactly once. Do not answer with prose or raw JSON outside "
            "that function call. The function schema is the complete and only allowed output."
        )
        self.directory = Path(directory)
        self.context_window = context_window
        self.compact_at = compact_at
        self.lane = lane
        self.model = model
        self.mcp = AgentSecretsMCP()
        self.history = [
            {"role": "system", "content": instructions + transport_instruction}
        ]
        self.thread_id = f"sparsie-fable-{uuid.uuid4()}"
        self.last_tool_call_id = None
        self.calls = 0
        self.compactions = 0
        self.cumulative_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.latest_prompt_tokens = 0
        self.coaching_image = None
        self.coaching_receipt = None
        self.metadata = {
            "connection": "sparsie-agentsecrets-zero-knowledge",
            "auth_type": "SPARSIE_API_KEY via AgentSecrets proxy; value never exposed",
            "lane": lane,
            "model": model,
            "model_provider": PROVIDER,
            "reasoning_effort": "high",
            "context_policy": {
                "window_tokens": context_window,
                "auto_compact_tokens": compact_at,
                "scope": "provider-input",
                "method": "model_generated_summary_with_archived_original_history",
            },
            "thread_id": self.thread_id,
            "persistent_thread": True,
            "conversation_transport": "local raw-turn replay to SPARSIE provider-only endpoint",
            "credential_transport": "AgentSecrets MCP api_call",
            "fallback_models": [],
        }

        initialize(self,context_policy=context_policy,context_window=context_window,output_reserve=32_000,history_attribute="history", token_counter=token_counter)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mcp.__enter__()
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.mcp.close()

    def usage_summary(self):
        return {
            "cumulative_token_usage": dict(self.cumulative_usage),
            "reported_context_windows": [self.context_window],
            "latest_prompt_tokens": self.latest_prompt_tokens,
            "compactions": self.compactions,
            "recovery_forks": 0,
        }

    def _content(self, text, images):
        blocks = [{"type": "text", "text": text}]
        for path in images:
            path = Path(path)
            data = path.read_bytes()
            if data.startswith(b"\x89PNG\r\n\x1a\n"):
                media_type = "image/png"
            elif data.startswith(b"\xff\xd8\xff"):
                media_type = "image/jpeg"
            else:
                raise ValueError(f"Unsupported trusted image format: {path.name}")
            encoded = base64.b64encode(data).decode("ascii")
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                }
            )
        return blocks

    def _api(self, body, *, timeout):
        text = self.mcp.call(
            "api_call",
            {
                "url": f"{SPARSIE_URL}/v1/provider/chat/completions",
                "method": "POST",
                "headers": {"content-type": "application/json"},
                "body": json.dumps(body, separators=(",", ":")),
                "injections": {"bearer": "SPARSIE_API_KEY"},
            },
            timeout=timeout,
        )
        if not text.startswith("HTTP ") or "\n\n" not in text:
            raise RuntimeError("Malformed AgentSecrets API response")
        status_line, payload_text = text.split("\n\n", 1)
        try:
            status = int(status_line.split()[1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError("Malformed AgentSecrets HTTP status") from exc
        payload = json.loads(payload_text)
        if status != 200:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            raise RuntimeError(f"SPARSIE returned HTTP {status}: {str(detail or payload)[:1200]}")
        return payload

    def _record(self, payload):
        self.calls += 1
        usage = payload.get("usage") or {}
        normalized = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
        self.latest_prompt_tokens = normalized["prompt_tokens"]
        for key, value in normalized.items():
            self.cumulative_usage[key] += value
        with (self.directory / "provider-responses.jsonl").open("a") as stream:
            stream.write(json.dumps({"call": self.calls, "response": payload}) + "\n")
        with (self.directory / "model-usage.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "call": self.calls,
                        "usage": normalized,
                        "cumulative": dict(self.cumulative_usage),
                        "prompt_context_tokens": self.latest_prompt_tokens,
                        "compactions": self.compactions,
                    }
                )
                + "\n"
            )

    def _tool(self):
        return {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Submit exactly one bounded RoboSWE operation.",
                "parameters": with_search(WAIT_SCHEMA) if self.history_store else WAIT_SCHEMA,
            },
        }

    def _parse_decision(self, payload):
        if payload.get("model") != self.model:
            raise RuntimeError("SPARSIE returned a substituted model")
        receipt = payload.get("sparsie") or {}
        if (
            receipt.get("lane") != self.lane
            or receipt.get("provider") != PROVIDER
            or receipt.get("provider_only") is not True
        ):
            raise RuntimeError("SPARSIE returned an unexpected provider receipt")
        choices = payload.get("choices") or []
        if len(choices) != 1:
            raise RuntimeError("Fable returned an unexpected choice count")
        message = choices[0].get("message") or {}
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            raise RuntimeError("Fable did not submit exactly one robot operation")
        call = calls[0]
        function = call.get("function") or {}
        if function.get("name") != TOOL_NAME:
            raise RuntimeError("Fable called an unexpected tool")
        raw = function.get("arguments")
        decision = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(decision, dict) or set(decision) != {
            "operation",
            "path",
            "content",
            "message",
        }:
            raise RuntimeError("Fable returned an invalid robot-operation object")
        if not all(isinstance(value, str) for value in decision.values()):
            raise RuntimeError("Fable returned non-string robot-operation fields")
        if decision["operation"] not in (with_search(WAIT_SCHEMA) if self.history_store else WAIT_SCHEMA)["properties"]["operation"]["enum"]:
            raise RuntimeError("Fable returned an unsupported robot operation")
        self.last_tool_call_id = str(call.get("id") or f"call-{self.calls}")
        self.history.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": calls,
                **{k: message[k] for k in ("reasoning", "reasoning_details") if k in message},
            }
        )
        return decision

    def _compact(self, *, timeout):
        archive = self.directory / f"context-before-compaction-{self.compactions + 1}.json.gz"
        with gzip.open(archive, "wt", encoding="utf-8") as stream:
            json.dump(self.history, stream, separators=(",", ":"))
        body = {
            "model": self.lane,
            "messages": [
                *self.history,
                {
                    "role": "user",
                    "content": (
                        "Create a precise continuity checkpoint for this ongoing RoboSWE task. "
                        "Preserve verified observations, failed hypotheses, task state, useful code "
                        "and NOTES knowledge, action-interface facts, and the next concrete tests. "
                        "Do not invent success or hidden simulator state."
                    ),
                },
            ],
            "stream": False,
            "temperature": 0,
            "max_tokens": 16_000,
            "required_provider": PROVIDER,
        }
        payload = self._api(body, timeout=timeout)
        self._record(payload)
        content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Fable compaction returned no continuity checkpoint")
        self.compactions += 1
        summary_path = self.directory / f"context-summary-{self.compactions}.md"
        summary_path.write_text(content)
        self.history = [
            {
                "role": "system",
                "content": self.instructions
                + "\n\nTransport contract: every response MUST invoke the provided "
                + f"{TOOL_NAME} function exactly once.",
            },
            {
                "role": "user",
                "content": (
                    "Authoritative model-generated continuity checkpoint from the archived earlier "
                    "turns. Treat it as fallible task memory, not simulator ground truth:\n\n" + content
                ),
            },
            {"role": "assistant", "content": "I will continue from this checkpoint."},
        ]
        self.last_tool_call_id = None

    @sliding_ask
    def ask(self, text, *, images=(), timeout=180, as_tool_output=False, **_kwargs):
        images = tuple(images)
        if self.coaching_image is not None:
            images = (*images, self.coaching_image)
            text += (
                "\nThe final attached image is Kirsten's real-hand bagel demonstration, not a "
                "simulator observation. Follow the human coaching in your instructions; infer "
                "simulator geometry from the robot cameras."
            )
            if self.coaching_receipt:
                self.coaching_receipt.write_text(
                    json.dumps(
                        {
                            "prompt": text,
                            "images": [str(path) for path in images],
                            "demonstration_attached": True,
                        },
                        indent=2,
                    )
                )
            self.coaching_image = None
        if self.history_store is None and self.latest_prompt_tokens >= self.compact_at:
            self._compact(timeout=max(30, min(timeout, 300)))
        content = self._content(text, images)
        if as_tool_output and self.last_tool_call_id:
            text_block = next(block for block in content if block["type"] == "text")
            image_blocks = [block for block in content if block["type"] == "image_url"]
            self.history.append(
                {
                    "role": "tool",
                    "tool_call_id": self.last_tool_call_id,
                    "name": TOOL_NAME,
                    "content": text_block["text"],
                }
            )
            if image_blocks:
                self.history.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Trusted camera images associated with the preceding operation result.",
                            },
                            *image_blocks,
                        ],
                    }
                )
        else:
            self.history.append({"role": "user", "content": content})
        body = {
            "model": self.lane,
            "messages": self.history,
            "stream": False,
            "temperature": 0,
            "max_tokens": 32_000,
            "required_provider": PROVIDER,
            "tools": [self._tool()],
            "parallel_tool_calls": False,
        }
        payload = self._api(body, timeout=timeout)
        self._record(payload)
        return self._parse_decision(payload)
