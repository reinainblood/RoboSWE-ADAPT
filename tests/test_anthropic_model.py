import json
import time

import pytest

from pairing.anthropic_model import (
    MODEL,
    ROLLOVER_AT,
    AnthropicLearningModel,
    BudgetLimitReached,
    SpendLedger,
    billed_usage,
    request_cost,
)
from pairing.cohort50 import SCHEMA


def response(decision=None, *, usage=None):
    decision = decision or {
        "operation": "finish",
        "path": "",
        "content": "",
        "message": "Preflight complete.",
    }
    return {
        "id": "msg_test",
        "model": MODEL,
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "signed"},
            {"type": "text", "text": json.dumps(decision)},
        ],
        "usage": usage
        or {
            "input_tokens": 10,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 200,
            "output_tokens": 20,
        },
    }


def test_request_cost_uses_opus55_cache_rates():
    usage = {
        "input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }
    assert request_cost(usage) == pytest.approx(29.2)


def test_request_cost_includes_provider_iterations_without_double_counting():
    usage = {
        "input_tokens": 10,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 30,
        "output_tokens": 40,
        "iterations": [
            {
                "type": "compaction",
                "input_tokens": 100,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 300,
                "output_tokens": 400,
            },
            {
                "type": "message",
                "input_tokens": 10,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
                "output_tokens": 40,
            },
        ],
    }
    assert billed_usage(usage) == {
        "input_tokens": 110,
        "cache_creation_input_tokens": 220,
        "cache_read_input_tokens": 330,
        "output_tokens": 440,
    }
    assert request_cost(usage) == pytest.approx(
        (110 * 4 + 220 * 5 + 330 * 0.2 + 440 * 20) / 1_000_000
    )


def test_spend_ledger_reserves_across_callers(tmp_path):
    path = tmp_path / "budget.json"
    first = SpendLedger(path, limit_usd=10)
    second = SpendLedger(path, limit_usd=10)
    reservation = first.reserve(6, label="one")
    with pytest.raises(BudgetLimitReached):
        second.reserve(5, label="two")
    first.commit(
        reservation,
        1.25,
        usage={
            "input_tokens": 0,
            "cache_creation_input_tokens": 250_000,
            "cache_read_input_tokens": 0,
            "output_tokens": 0,
        },
    )
    snapshot = second.snapshot()
    assert snapshot["spent_usd"] == pytest.approx(1.25)
    assert snapshot["reserved_usd"] == 0
    assert snapshot["remaining_unreserved_usd"] == pytest.approx(8.75)


def test_structured_decision_preserves_thinking_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr("pairing.anthropic_model.credential", lambda: "secret-not-recorded")
    model = AnthropicLearningModel(
        "instructions", tmp_path, context_policy="legacy", max_tokens=1024, workspace_id="wrkspc_test"
    )
    bodies = []

    def fake_request(body, *, timeout, purpose):
        bodies.append(body)
        payload = response()
        payload["content"][0]["thinking"] = "verbatim provider-exposed thought"
        model._record(payload, purpose=purpose, elapsed_seconds=0.1, cost=request_cost(payload["usage"]))
        return payload

    monkeypatch.setattr(model, "_request", fake_request)
    decision = model.ask('{"observation":{}}', timeout=10)
    assert decision["operation"] == "finish"
    assert bodies[0]["model"] == MODEL
    assert bodies[0]["output_config"]["effort"] == "high"
    assert bodies[0]["output_config"]["format"] == {
        "type": "json_schema",
        "schema": SCHEMA,
    }
    assert bodies[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert "context_management" not in bodies[0]
    assert bodies[0]["system"][0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "5m",
    }
    assert model.metadata["local_context_rollover"]["trigger_effective_tokens"] == ROLLOVER_AT
    assert "disabled" in model.metadata["local_context_rollover"]["provider_compaction"]
    assert model.messages[-1]["content"][0]["type"] == "thinking"
    trace = json.loads((tmp_path / "provider-thinking.jsonl").read_text())
    assert trace["provider_response_id"] == "msg_test"
    assert trace["provider_exposed_block_count"] == 1
    assert trace["plaintext_thinking_available"] is True
    assert trace["blocks"] == [
        {
            "content_block_index": 0,
            "block": {
                "type": "thinking",
                "thinking": "verbatim provider-exposed thought",
                "signature": "signed",
            },
        }
    ]
    assert model.last_provider_thinking_trace == trace
    assert "secret-not-recorded" not in json.dumps(model.metadata)
    assert "secret-not-recorded" not in (tmp_path / "model-usage.jsonl").read_text()


def test_provider_thinking_log_preserves_redacted_block(monkeypatch, tmp_path):
    monkeypatch.setattr("pairing.anthropic_model.credential", lambda: "secret-not-recorded")
    model = AnthropicLearningModel(
        "instructions", tmp_path, context_policy="legacy", max_tokens=1024, workspace_id="wrkspc_test"
    )
    payload = response()
    payload["content"][0] = {
        "type": "redacted_thinking",
        "data": "provider-opaque-payload",
    }
    model._record(payload, purpose="decision", elapsed_seconds=0.2, cost=0.0)
    trace = json.loads((tmp_path / "provider-thinking.jsonl").read_text())
    assert trace["plaintext_thinking_available"] is False
    assert trace["blocks"][0]["block"] == payload["content"][0]


def test_parse_rejects_substituted_model():
    payload = response()
    payload["model"] = "claude-not-opus"
    with pytest.raises(RuntimeError, match="substituted"):
        AnthropicLearningModel._parse_decision(payload, SCHEMA)


def test_context_archive_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr("pairing.anthropic_model.credential", lambda: "secret-not-recorded")
    model = AnthropicLearningModel(
        "instructions", tmp_path, context_policy="legacy", workspace_id="wrkspc_test"
    )
    model.messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    receipt = model.save_context(tmp_path / "context.json.gz")
    assert receipt["messages"] == 1
    assert len(receipt["sha256"]) == 64


def test_local_rollover_archives_history_without_provider_compaction(monkeypatch, tmp_path):
    monkeypatch.setattr("pairing.anthropic_model.credential", lambda: "secret-not-recorded")
    model = AnthropicLearningModel(
        "instructions", tmp_path, context_policy="legacy", workspace_id="wrkspc_test"
    )
    model.messages = [
        {"role": "user", "content": [{"type": "text", "text": "old history"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "old reply"}]},
    ]

    model.messages = [
        {"role": "user", "content": [{"type": "text", "text": "old feedback"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private", "signature": "signed"},
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "operation": "write_execute",
                            "path": "move.py",
                            "content": "large source omitted from rollover",
                            "message": "Measured the target and moved to a verified standoff.",
                        }
                    ),
                },
            ],
        },
    ]
    model.latest_context_tokens = ROLLOVER_AT
    bodies = []

    def fake_request(body, *, timeout, purpose):
        bodies.append(body)
        payload = response()
        model._record(
            payload,
            purpose=purpose,
            elapsed_seconds=0.1,
            cost=request_cost(payload["usage"]),
        )
        return payload

    monkeypatch.setattr(model, "_request", fake_request)
    decision = model.ask('{"observation":{"control_step":42}}', timeout=10)
    assert decision["operation"] == "finish"
    assert model.compactions == 0
    assert model.rollovers == 1
    assert (tmp_path / "context-before-rollover-1.json.gz").exists()
    continuity = json.loads((tmp_path / "context-rollover-1.json").read_text())
    assert continuity["recent_opus_public_decisions"] == [
        {
            "operation": "write_execute",
            "path": "move.py",
            "message": "Measured the target and moved to a verified standoff.",
        }
    ]
    sent = bodies[0]["messages"]
    assert len(sent) == 1
    assert "TRUSTED LOCAL CONTEXT ROLLOVER" in sent[0]["content"][0]["text"]
    assert '"control_step":42' in sent[0]["content"][0]["text"]
    assert "large source omitted" not in sent[0]["content"][0]["text"]
    assert "context_management" not in bodies[0]
    receipt = json.loads((tmp_path / "context-rollovers.jsonl").read_text())
    assert receipt["provider_request"] is False
    assert receipt["provider_cost_usd"] == 0.0
    assert len(model.messages) == 2


def test_explicit_refusal_is_not_infrastructure_error():
    from pairing.anthropic_model import AnthropicLearningModel, ModelRefusal, MODEL
    import pytest
    with pytest.raises(ModelRefusal) as error:
        AnthropicLearningModel._parse_decision({'model':MODEL,'stop_reason':'refusal','content':[]},{})
    assert error.value.status=='model_refusal'


def test_mixed_cache_write_ttls_use_actual_receipt_buckets():
    usage={'cache_creation_input_tokens':1_000_000,'cache_creation':{
        'ephemeral_5m_input_tokens':400_000,'ephemeral_1h_input_tokens':600_000}}
    assert request_cost(usage)==pytest.approx(6.8)
    assert request_cost(usage,cache_ttl='1h')==pytest.approx(6.8)


def test_one_hour_write_fallback_and_reservation():
    assert request_cost({'cache_creation_input_tokens':1_000_000},cache_ttl='1h')==8.0
    assert request_cost({'cache_creation_input_tokens':1_000_000})==5.0
    model=object.__new__(AnthropicLearningModel);model.context_window=1_000_000
    model.cache_ttl='1h'
    assert model._reserve_amount(32768)==pytest.approx(8+32768*20/1_000_000)
    model.cache_ttl='5m'
    assert model._reserve_amount(32768)==pytest.approx(5+32768*20/1_000_000)


def test_iterations_mixed_ttl_and_missing_details_no_double_count():
    usage={'cache_creation_input_tokens':1_000_000,'cache_creation':{'ephemeral_1h_input_tokens':1_000_000},
        'iterations':[
            {'cache_creation_input_tokens':100,'cache_creation':{'ephemeral_5m_input_tokens':25,'ephemeral_1h_input_tokens':75}},
            {'cache_creation_input_tokens':200},
            {'cache_creation':{'ephemeral_1h_input_tokens':300}}]}
    assert billed_usage(usage)['cache_creation_input_tokens']==600
    assert request_cost(usage,cache_ttl='1h')==pytest.approx((25*5+575*8)/1_000_000)


def test_partial_cache_buckets_fallback_to_configured_ttl():
    usage={'cache_creation_input_tokens':100,'cache_creation':{'ephemeral_5m_input_tokens':20}}
    assert request_cost(usage,cache_ttl='1h')==pytest.approx((20*5+80*8)/1_000_000)
    with pytest.raises(ValueError,match='inconsistent'):
        request_cost({'cache_creation_input_tokens':10,'cache_creation':{'ephemeral_1h_input_tokens':11}})


def test_one_hour_request_settles_ledger_with_configured_ttl_when_buckets_absent(monkeypatch,tmp_path):
    import io
    monkeypatch.setattr('pairing.anthropic_model.credential',lambda:'fake-test-key')
    model=AnthropicLearningModel('instructions',tmp_path,context_policy='legacy',workspace_id='wrkspc_test',
        cache_ttl='1h',max_tokens=1024,ledger_path=tmp_path/'ledger.json',spend_limit_usd=10)
    payload=response(usage={'input_tokens':0,'cache_creation_input_tokens':1_000_000,'cache_read_input_tokens':0,'output_tokens':0})
    monkeypatch.setattr('pairing.anthropic_model.urllib.request.urlopen',lambda *a,**kw:io.BytesIO(json.dumps(payload).encode()))
    model._request({'max_tokens':1024},timeout=10,purpose='unit-test')
    assert model.ledger.snapshot()['spent_usd']==8.0
    assert model.ledger.snapshot()['reserved_usd']==0
    assert model.metadata['pricing_usd_per_million_tokens']['cache_creation_input_tokens']==8.0


def _request_test_model(monkeypatch,tmp_path):
    monkeypatch.setattr('pairing.anthropic_model.credential',lambda:'private-test-key')
    return AnthropicLearningModel('instructions',tmp_path,context_policy='legacy',workspace_id='wrkspc_test',
        ledger_path=tmp_path/'ledger.json',spend_limit_usd=100)


def test_count_only_transient_retry_uses_original_deadline_and_records_cause(monkeypatch,tmp_path):
    import io,urllib.error
    model=_request_test_model(monkeypatch,tmp_path);clock=[100.0];timeouts=[]
    monkeypatch.setattr('pairing.anthropic_model.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('pairing.anthropic_model.time.sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    model._history_deadline=200
    def count(req,timeout):
        assert req.full_url.endswith('/count_tokens');timeouts.append(timeout)
        if len(timeouts)==1:
            clock[0]+=90
            raise urllib.error.URLError(TimeoutError('socket timeout private-test-key'))
        return io.BytesIO(b'{"input_tokens":123}')
    monkeypatch.setattr('pairing.anthropic_model.urllib.request.urlopen',count)
    assert model._count_history_tokens(messages=[],pending={'text':'hello','images':[]},overhead={'instructions':'i','schema':SCHEMA})==123
    assert timeouts==[90,9]
    events=(tmp_path/'provider-request-events.jsonl').read_text()
    assert 'TimeoutError' in events and 'private-test-key' not in events
    assert model.ledger.snapshot()['requests']==0 and model.ledger.snapshot()['reserved_usd']==0


def test_count_permanent_http_error_not_retried(monkeypatch,tmp_path):
    import io,urllib.error
    model=_request_test_model(monkeypatch,tmp_path);calls=[]
    def fail(*a,**k):calls.append(1);raise urllib.error.HTTPError('https://api.anthropic.com',400,'invalid',{},io.BytesIO(b''))
    monkeypatch.setattr('pairing.anthropic_model.urllib.request.urlopen',fail)
    with pytest.raises(RuntimeError,match='count-only'):model._count_history_tokens(messages=[],pending={'text':'h','images':[]},overhead={'instructions':'i','schema':SCHEMA})
    assert calls==[1]


@pytest.mark.parametrize('failure',['http503','timeout','missing_usage'])
def test_inference_never_retries_or_releases_unknown_spend(monkeypatch,tmp_path,failure):
    import io,urllib.error
    model=_request_test_model(monkeypatch,tmp_path);calls=[]
    def fail(*a,**k):
        calls.append(1)
        if failure=='http503':raise urllib.error.HTTPError('https://api.anthropic.com',503,'error',{},io.BytesIO(b''))
        if failure=='timeout':raise TimeoutError('uncertain private-test-key')
        return io.BytesIO(b'{"id":"response-without-usage"}')
    monkeypatch.setattr('pairing.anthropic_model.urllib.request.urlopen',fail)
    with pytest.raises(RuntimeError):model._request({'max_tokens':1024},timeout=20,purpose='test')
    assert calls==[1]
    assert model.ledger.snapshot()['reserved_usd']>0 and model.ledger.snapshot()['spent_usd']==0
    assert 'private-test-key' not in (tmp_path/'provider-request-events.jsonl').read_text()


def test_inference_respects_shared_original_deadline(monkeypatch,tmp_path):
    import io
    model=_request_test_model(monkeypatch,tmp_path);model._history_deadline=105;timeouts=[]
    monkeypatch.setattr('pairing.anthropic_model.time.monotonic',lambda:100)
    def post(*args,timeout):timeouts.append(timeout);return io.BytesIO(json.dumps(response()).encode())
    monkeypatch.setattr('pairing.anthropic_model.urllib.request.urlopen',post)
    model._request({'max_tokens':1024},timeout=90,purpose='test')
    assert timeouts==[5]
