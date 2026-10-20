import copy
import json
from pathlib import Path

import pytest

from pairing.sparsie_model import LANE, MODEL, PROVIDER, TOOL_NAME, SparsieLearningModel


def response(decision, *, model=MODEL, provider=PROVIDER):
    return {
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {
                                "name": TOOL_NAME,
                                "arguments": json.dumps(decision),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
        "sparsie": {
            "provider_only": True,
            "lane": LANE,
            "provider": provider,
            "stored_turns": False,
        },
    }


class FakeModel(SparsieLearningModel):
    def __init__(self, directory: Path, replies):
        super().__init__(instructions="fixed", directory=directory, context_policy="legacy")
        self.replies = list(replies)
        self.bodies = []

    def _api(self, body, *, timeout):
        self.bodies.append((copy.deepcopy(body), timeout))
        return self.replies.pop(0)


def test_fable_decision_uses_required_exact_provider_and_tool(tmp_path):
    decision = {"operation": "observe", "path": "", "content": "", "message": "Inspect."}
    model = FakeModel(tmp_path, [response(decision)])

    assert model.ask("state", timeout=42) == decision
    body, timeout = model.bodies[0]
    assert timeout == 42
    assert body["model"] == LANE
    assert body["required_provider"] == PROVIDER
    assert "tool_choice" not in body
    assert body["tools"][0]["function"]["name"] == TOOL_NAME
    assert body["parallel_tool_calls"] is False
    assert model.history[-1]["role"] == "assistant"
    assert model.usage_summary()["cumulative_token_usage"]["total_tokens"] == 24


def test_followup_is_tool_result_plus_trusted_camera_message(tmp_path):
    decision = {"operation": "observe", "path": "", "content": "", "message": "Inspect."}
    image = tmp_path / "camera.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nmock")
    model = FakeModel(tmp_path, [response(decision), response(decision)])

    model.ask("first")
    model.ask("feedback", images=[image], as_tool_output=True)

    messages = model.bodies[1][0]["messages"]
    assert messages[-3]["role"] == "assistant"
    assert messages[-2]["role"] == "tool"
    assert messages[-2]["tool_call_id"] == "tool-1"
    assert messages[-1]["role"] == "user"
    assert any(block["type"] == "image_url" for block in messages[-1]["content"])


def test_image_media_type_comes_from_bytes_not_filename(tmp_path):
    model = FakeModel(tmp_path, [])
    png = tmp_path / "camera.jpg"
    jpeg = tmp_path / "demonstration.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nmock")
    jpeg.write_bytes(b"\xff\xd8\xffmock")

    blocks = model._content("state", [png, jpeg])

    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert blocks[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_model_substitution_is_rejected(tmp_path):
    decision = {"operation": "observe", "path": "", "content": "", "message": "Inspect."}
    model = FakeModel(tmp_path, [response(decision, model="claude-fable-5")])
    with pytest.raises(RuntimeError, match="substituted model"):
        model.ask("state")


def test_provider_substitution_is_rejected(tmp_path):
    decision = {"operation": "observe", "path": "", "content": "", "message": "Inspect."}
    model = FakeModel(tmp_path, [response(decision, provider="anthropic-direct")])
    with pytest.raises(RuntimeError, match="unexpected provider"):
        model.ask("state")
