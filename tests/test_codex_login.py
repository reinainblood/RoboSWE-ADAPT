import time
import tomllib
from types import SimpleNamespace

import pytest

from roboswe.codex_login import (
    MODEL_PROVIDER,
    CodexLogin,
    ModelCompactionRejected,
    ModelRequestRejected,
    _config_value,
    rejected_bad_request,
)


def test_http_provider_override_is_a_complete_toml_table_without_credentials():
    provider = tomllib.loads("provider=" + _config_value(MODEL_PROVIDER))["provider"]
    assert provider == MODEL_PROVIDER
    assert provider["requires_openai_auth"] is True
    assert provider["supports_websockets"] is False
    assert set(provider) == {
        "name",
        "wire_api",
        "requires_openai_auth",
        "supports_websockets",
    }


def test_rejected_compaction_forks_only_the_last_successful_checkpoint(monkeypatch):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    client = CodexLogin(persistent=True)
    client.thread_id = "original"
    client.last_completed_turn = "last-success"
    client.directory = SimpleNamespace(name="/isolated")
    clock = [100.0]
    monkeypatch.setattr("roboswe.codex_login.time.monotonic", lambda: clock[0])
    calls, compactions, forks = [], [], []

    def once(text, **kwargs):
        calls.append((client.thread_id, kwargs["timeout"]))
        if len(calls) == 1:
            clock[0] += 2
            raise ModelRequestRejected("Bad Request")
        return {"operation": "observe"}

    def compact(deadline):
        assert deadline == 110
        compactions.append(client.thread_id)
        if len(compactions) == 1:
            clock[0] += 1
            raise ModelCompactionRejected("Bad Request")
        clock[0] += 3

    def rpc(method, params, deadline):
        assert method == "thread/fork" and deadline == 110
        assert params["threadId"] == "original"
        assert params["lastTurnId"] == "last-success"
        assert params["sandbox"] == "read-only"
        assert params["approvalPolicy"] == "never"
        assert params["modelProvider"] == "roboswe_http"
        assert params["ephemeral"] is False
        forks.append(params)
        clock[0] += 1
        return {"thread": {"id": "recovered"}}

    client._ask_once, client._compact, client._rpc = once, compact, rpc
    client.close = lambda: pytest.fail("Successful recovery closed the client")
    assert client.ask("feedback", timeout=10)["operation"] == "observe"
    assert calls == [("original", 10), ("recovered", 3)]
    assert compactions == ["original", "recovered"] and len(forks) == 1
    assert client.metadata["recovery_lineage"] == [
        {
            "parent_thread_id": "original",
            "thread_id": "recovered",
            "last_completed_turn": "last-success",
        }
    ]


def test_ephemeral_thread_cannot_claim_checkpoint_recovery(monkeypatch):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    client = CodexLogin()
    client.last_completed_turn = "last-success"
    client._rpc = lambda *_: pytest.fail("Ephemeral history cannot be forked")
    with pytest.raises(RuntimeError, match="persistent model thread"):
        client._fork_checkpoint(time.monotonic() + 10)


@pytest.mark.parametrize("specific", [False, True])
def test_checkpoint_fallback_cannot_loop_or_bypass_other_errors(monkeypatch, specific):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    client = CodexLogin()
    forks, compactions, calls = [], [], []

    def reject(*args, **kwargs):
        calls.append(1)
        raise ModelRequestRejected("Bad Request")

    def compact(deadline):
        compactions.append(1)
        error = ModelCompactionRejected if specific else RuntimeError
        raise error("compaction failed")

    client._ask_once = reject
    client._compact = compact
    client._fork_checkpoint = lambda _: forks.append(1)
    with pytest.raises(RuntimeError, match="recovery failed"):
        client.ask("feedback")
    assert len(forks) == int(specific)
    assert len(compactions) == 1 + int(specific)
    assert len(calls) == 1


def test_bad_request_recovery_retries_once_in_same_thread_with_remaining_budget(
    monkeypatch,
):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    client = CodexLogin()
    client.thread_id = "same-thread"
    clock = [100.0]
    monkeypatch.setattr("roboswe.codex_login.time.monotonic", lambda: clock[0])
    calls = []

    def once(text, **kwargs):
        calls.append((text, kwargs["timeout"], client.thread_id))
        if len(calls) == 1:
            clock[0] += 2
            raise ModelRequestRejected("Bad Request")
        return {"operation": "observe"}

    def compact(deadline):
        assert deadline == 110
        clock[0] += 3

    client._ask_once = once
    client._compact = compact
    client.close = lambda: pytest.fail("Successful recovery closed the thread")
    assert client.ask("latest feedback", timeout=10)["operation"] == "observe"
    assert [c[1] for c in calls] == [10, 5]
    assert all(c[2] == "same-thread" for c in calls)
    assert "before any proposed operation was applied" in calls[1][0]


def test_recovery_cannot_loop_or_extend_the_deadline(monkeypatch):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    client = CodexLogin()
    calls = []
    closed = []

    def reject(*args, **kwargs):
        calls.append(1)
        raise ModelRequestRejected("Bad Request")

    client._ask_once = reject
    client._compact = lambda _: None
    client.close = lambda: closed.append(True)
    with pytest.raises(RuntimeError, match="recovery failed"):
        client.ask("feedback")
    assert len(calls) == 2 and closed == [True]


def test_bad_request_classifier_is_specific_to_provider_error_envelope():
    assert rejected_bad_request({"message": '{"detail": "Bad Request"}'})
    assert not rejected_bad_request({"message": "An agent printed Bad Request"})
    assert not rejected_bad_request({"message": '{"detail": "Unauthorized"}'})
    assert rejected_bad_request(
        {"message": 'Error running remote compact task: {"detail":"Bad Request"}'}
    )


def test_provider_rejection_preserves_thread_for_recovery(monkeypatch):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    client = CodexLogin()
    client.thread_id = "same"
    client.last_completed_turn = "last-success"
    client._rpc = lambda *_: {"turn": {"id": "failed"}}
    client.close = lambda: pytest.fail("Rejected turn discarded recoverable history")
    client.events.put(
        {
            "method": "turn/completed",
            "params": {
                "turn": {
                    "id": "failed",
                    "status": "failed",
                    "error": {"message": '{"detail": "Bad Request"}'},
                }
            },
        }
    )
    with pytest.raises(ModelRequestRejected):
        client._ask_once("feedback")
    assert client.last_completed_turn == "last-success"


def test_recovery_exhausting_budget_does_not_start_an_extra_turn(monkeypatch):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    client = CodexLogin()
    clock = [100.0]
    calls = []
    monkeypatch.setattr("roboswe.codex_login.time.monotonic", lambda: clock[0])

    def reject(*args, **kwargs):
        calls.append(1)
        raise ModelRequestRejected("Bad Request")

    def compact(deadline):
        clock[0] = deadline + 1

    client._ask_once = reject
    client._compact = compact
    with pytest.raises(RuntimeError, match="deadline expired"):
        client.ask("feedback", timeout=10)
    assert calls == [1]


def test_usage_and_compaction_diagnostics_exclude_model_text(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    path = tmp_path / "events.jsonl"
    client = CodexLogin(diagnostics=path)
    for event in [
        {
            "method": "item/completed",
            "params": {"item": {"type": "agentMessage", "text": "PRIVATE_MODEL_TEXT"}},
        },
        {"method": "item/completed", "params": {"item": {"type": "contextCompaction"}}},
        {
            "method": "thread/tokenUsage/updated",
            "params": {"tokenUsage": {"last": {"totalTokens": 1000}}},
        },
    ]:
        client._record_event(event)
    data = path.read_text()
    assert "PRIVATE_MODEL_TEXT" not in data
    rows = [json.loads(line) for line in data.splitlines()]
    assert len(rows) == 2
    assert rows[0]["item_type"] == "contextCompaction"
    assert rows[1]["token_usage"]["last"]["totalTokens"] == 1000


def test_opt_in_summary_log_contains_only_provider_summary(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    path = tmp_path / "summaries.jsonl"
    client = CodexLogin(
        reasoning_summary="detailed", verbosity="high", summary_log=path
    )
    client._record_event(
        {
            "method": "item/completed",
            "params": {
                "threadId": "thread",
                "turnId": "turn",
                "item": {
                    "type": "reasoning",
                    "summary": ["Visible provider summary"],
                    "content": ["DO_NOT_COPY_CONTENT"],
                    "encrypted_content": "DO_NOT_COPY_ENCRYPTED",
                },
            },
        }
    )
    data = path.read_text()
    assert json.loads(data)["summary"] == ["Visible provider summary"]
    assert "DO_NOT_COPY" not in data


@pytest.mark.parametrize("window,threshold", [(1000, 1000), (2000, 999), (True, 1000)])
def test_invalid_context_budget_rejected(monkeypatch, window, threshold):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    with pytest.raises(ValueError, match="context policy"):
        CodexLogin(context_window=window, compact_at=threshold)


def test_sensor_feedback_is_tool_output_not_a_new_user_instruction(
    tmp_path, monkeypatch
):
    import base64
    import json

    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    client = CodexLogin()
    client.thread_id = "test"
    requests = []

    def rpc(method, params, *_):
        requests.append((method, params))
        return {"turn": {"id": "response"}}

    client._rpc = rpc
    client.events.put(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "agentMessage",
                    "text": json.dumps({"operation": "observe"}),
                }
            },
        }
    )
    client.events.put(
        {
            "method": "turn/completed",
            "params": {"turn": {"id": "response", "status": "completed"}},
        }
    )
    png = tmp_path / "trusted.png"
    png.write_bytes(b"trusted camera bytes")
    assert (
        client.ask("sensor data", images=[png], as_tool_output=True)["operation"]
        == "observe"
    )
    params = requests[0][1]
    assert params["input"] == []
    assert params["toolOutput"]["name"] == "roboswe_feedback"
    output = params["toolOutput"]["output"]
    assert output[0] == {"type": "input_text", "text": "sensor data"}
    assert base64.b64decode(output[1]["image_url"].split(",", 1)[1]) == png.read_bytes()
    assert params["sandboxPolicy"] == {"type": "readOnly"}
    assert params["approvalPolicy"] == "never"


def test_host_tool_request_is_rejected(monkeypatch):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    client = CodexLogin()
    sent = []
    client._send = sent.append
    client.events.put({"id": 123, "method": "item/tool/call", "params": {}})
    with pytest.raises(RuntimeError, match="host tool"):
        client._next(time.monotonic() + 1)
    assert sent[0]["id"] == 123 and "error" in sent[0]


def test_unexpected_host_execution_fails_closed(monkeypatch):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    client = CodexLogin()
    client.events.put(
        {"method": "item/started", "params": {"item": {"type": "commandExecution"}}}
    )
    with pytest.raises(RuntimeError, match="prohibited"):
        client._next(time.monotonic() + 1)


def test_deadline_is_total_not_reset_by_events(monkeypatch):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    with pytest.raises(TimeoutError):
        CodexLogin()._next(time.monotonic() - 1)


def test_turn_timeout_interrupts_without_losing_conversation(monkeypatch):
    monkeypatch.setattr("roboswe.codex_login.shutil.which", lambda _: "/test/codex")
    client = CodexLogin()
    client.thread_id = "retained"
    client._rpc = lambda *_: {"turn": {"id": "expired"}}
    sent = []
    client._send = sent.append
    events = iter(
        [
            TimeoutError(),
            {"id": 1, "result": {}},
            {
                "method": "turn/completed",
                "params": {"turn": {"id": "expired", "status": "interrupted"}},
            },
        ]
    )

    def next_event(_):
        event = next(events)
        if isinstance(event, Exception):
            raise event
        return event

    client._next = next_event
    client.close = lambda: pytest.fail("Conversation was discarded")
    with pytest.raises(TimeoutError, match="conversation retained"):
        client.ask("bounded turn")
    assert sent[0]["method"] == "turn/interrupt"
    assert client.thread_id == "retained"
