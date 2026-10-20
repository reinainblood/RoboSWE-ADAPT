import base64
import hashlib
import json
import urllib.error
from copy import deepcopy

import pytest

from roboswe.library import glimpse_judge as glimpse
from roboswe.library.video_judge import publish_visual_audit


@pytest.fixture
def clip(tmp_path, monkeypatch):
    av = pytest.importorskip("av")
    image = pytest.importorskip("PIL.Image")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key-never-log")
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for tick in range(20):
            frame = av.VideoFrame.from_image(
                image.new("RGB", (32, 32), (tick * 10, 0, 0))
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def answer(satisfied=True):
    return {
        "confidence": "medium",
        "visible_motion": "Synthetic test footage only.",
        "checks": [
            {
                "id": "task_goal",
                "satisfied": satisfied,
                "reason": "Test observation",
                "evidence": [
                    {
                        "view": "overview",
                        "start_seconds": 0.0,
                        "end_seconds": 0.5,
                        "observation": "Test observation",
                    }
                ],
            }
        ],
    }


def response(result=None, **overrides):
    return {
        "id": "test-generation",
        "model": glimpse.MODEL,
        "provider": "Together",
        "usage": {"prompt_tokens": 100, "completion_tokens": 200, "cost": 0.00066},
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(answer() if result is None else result),
                },
            }
        ],
        **overrides,
    }


def test_explicit_reasoning_effort_is_recorded_without_changing_verdict(clip, tmp_path):
    captured = []

    def request(payload, **kwargs):
        captured.append(payload)
        return response()

    report = glimpse.judge_video(
        clip, instruction="Synthetic test task", output=tmp_path / "low-effort",
        mechanical_success=True, reasoning_effort="low", request_fn=request,
    )
    assert captured[0]["reasoning"] == {"effort": "low"}
    assert report["request_settings"]["reasoning"] == {"effort": "low"}
    assert report["changes_score"] is False


def test_gemini_audit_pins_model_and_provider(clip, tmp_path):
    captured = []

    def request(payload, **kwargs):
        captured.append(payload)
        return response(model="google/gemini-4-argon")

    report = glimpse.judge_video(
        clip, instruction="Synthetic test task", output=tmp_path / "gemini-audit",
        mechanical_success=True, judge_backend="gemini", request_fn=request,
    )
    assert captured[0]["model"] == "google/gemini-4-argon"
    assert captured[0]["provider"]["only"] == ["google-vertex/global"]
    assert report["backend"] == "gemini"
    assert report["changes_score"] is False


@pytest.mark.parametrize(
    "visible,mechanical,status",
    [
        (True, True, "agrees"),
        (False, False, "agrees"),
        (True, False, "disagrees"),
        (False, True, "disagrees"),
        (None, True, "inconclusive"),
        (True, None, "uncompared"),
    ],
)
def test_native_video_is_blind_bounded_and_auditable(clip, visible, mechanical, status):
    calls = []

    def transport(payload, *, api_key, timeout):
        calls.append(payload)
        assert api_key == "unit-test-key-never-log"
        assert timeout == 180
        assert payload["model"] == "elorian/glimpse"
        assert payload["provider"]["allow_fallbacks"] is False
        assert payload["provider"]["require_parameters"] is True
        assert payload["max_tokens"] == 8192
        assert "tools" not in payload
        text = json.dumps(payload)
        assert "mechanical_success" not in text
        assert "unit-test-key-never-log" not in text
        parts = payload["messages"][1]["content"]
        prompt = json.loads(parts[0]["text"])
        assert prompt["success_criteria"] == {"sustained_control_steps": 10}
        video_parts = [part for part in parts if part["type"] == "video_url"]
        assert len(video_parts) == 2
        assert (
            base64.b64decode(video_parts[0]["video_url"]["url"].split(",")[1])
            == clip.read_bytes()
        )
        return response(answer(visible))

    report = glimpse.judge_video(
        clip,
        instruction="Test goal",
        output=clip.parent / "audit",
        mechanical_success=mechanical,
        criteria={"sustained_control_steps": 10},
        additional_videos={"wrist": clip},
        request_fn=transport,
    )
    assert report["status"] == status
    assert report["changes_score"] is False
    assert report["mechanical_success"] is mechanical
    assert report["answer"]["visible_success"] is visible
    assert len(calls) == 1
    assert (
        report["views"][0]["video_sha256"]
        == hashlib.sha256(clip.read_bytes()).hexdigest()
    )
    assert report["views"][0]["duration_seconds"] == 1.0
    assert report["coverage"] == "native_video_provider_sampling_unknown"
    saved = (clip.parent / "audit/audit.json").read_text()
    assert "unit-test-key-never-log" not in saved
    assert "data:video" not in saved
    assert json.loads(saved) == report


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["checks"][0].update(satisfied="true"),
        lambda a: a["checks"][0].update(satisfied=1),
        lambda a: a["checks"][0].update(evidence=[]),
        lambda a: a["checks"][0].update(id="invented"),
        lambda a: a["checks"].append(deepcopy(a["checks"][0])),
        lambda a: a.update(checks=[]),
        lambda a: a.update(confidence="certain"),
        lambda a: a.update(visible_success=True),
        lambda a: a["checks"][0]["evidence"][0].update(view="nonexistent-camera"),
        lambda a: a["checks"][0]["evidence"][0].update(start_seconds=-1),
        lambda a: a["checks"][0]["evidence"][0].update(start_seconds=True),
        lambda a: a["checks"][0]["evidence"][0].update(start_seconds=0.9),
        lambda a: a["checks"][0]["evidence"][0].update(end_seconds=99),
        lambda a: a["checks"][0]["evidence"][0].update(end_seconds=float("nan")),
    ],
)
def test_invalid_judgments_never_become_success(clip, mutation):
    result = answer()
    mutation(result)
    report = glimpse.judge_video(
        clip,
        instruction="Test goal",
        output=clip.parent / "audit",
        request_fn=lambda *a, **k: response(result),
    )
    assert report["status"] == "judge_failed"
    assert "answer" not in report
    assert report["changes_score"] is False


@pytest.mark.parametrize(
    "result",
    [
        {"error": {"message": "provider error"}},
        response(model="different/model"),
        response(choices=[]),
        response(choices=[{"finish_reason": "length", "message": {"content": "{}"}}]),
        response(
            choices=[{"finish_reason": "stop", "message": {"content": "not json"}}]
        ),
        response(
            choices=[
                {
                    "finish_reason": "stop",
                    "message": {"content": "{}", "tool_calls": [{}]},
                }
            ]
        ),
        response(
            choices=[
                {"finish_reason": "stop", "message": {"content": "{}", "refusal": "no"}}
            ]
        ),
    ],
)
def test_provider_failures_are_not_false_robot_failures(clip, result):
    report = glimpse.judge_video(
        clip,
        instruction="Test goal",
        output=clip.parent / "audit",
        mechanical_success=True,
        request_fn=lambda *a, **k: result,
    )
    assert report["status"] == "judge_failed"
    assert report["mechanical_success"] is True
    assert "answer" not in report


@pytest.mark.parametrize(
    "options",
    [
        {"max_video_bytes": 1},
        {"max_video_seconds": 0.5},
        {"additional_videos": {"overview": "ignored.mp4"}},
    ],
)
def test_invalid_inputs_fail_before_network(clip, options):
    calls = []
    report = glimpse.judge_video(
        clip,
        instruction="Test goal",
        output=clip.parent / "audit",
        **options,
        request_fn=lambda *a, **k: calls.append(a),
    )
    assert report["status"] == "judge_failed"
    assert not calls


def test_corrupt_video_is_not_sent(clip):
    clip.write_bytes(b"not an mp4")
    report = glimpse.judge_video(
        clip,
        instruction="Test goal",
        output=clip.parent / "audit",
        request_fn=lambda *a, **k: pytest.fail("Corrupt clip must not reach network"),
    )
    assert report["status"] == "judge_failed"


def test_missing_key_is_recorded_without_call(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    report = glimpse.judge_video(
        tmp_path / "unused.mp4",
        instruction="Test goal",
        output=tmp_path / "audit",
        request_fn=lambda *a, **k: pytest.fail("Missing key must not reach network"),
    )
    assert report["status"] == "judge_failed"
    assert "OPENROUTER_API_KEY" in report["error"]


def test_invalid_key_cannot_leak_through_header_error(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-credential\r\n")
    report = glimpse.judge_video(
        tmp_path / "unused.mp4",
        instruction="Test goal",
        output=tmp_path / "audit",
        request_fn=lambda *a, **k: pytest.fail("Invalid key must not reach network"),
    )
    assert report["status"] == "judge_failed"
    assert "private-credential" not in (tmp_path / "audit/audit.json").read_text()


def test_video_limits_cover_all_views(clip):
    report = glimpse.judge_video(
        clip,
        instruction="Test goal",
        output=clip.parent / "audit",
        additional_videos={"wrist": clip},
        max_video_seconds=1.5,
        request_fn=lambda *a, **k: pytest.fail("Aggregate budget must be enforced"),
    )
    assert report["status"] == "judge_failed"
    assert "duration budget" in report["error"]


@pytest.mark.parametrize(
    "states,expected",
    [
        ([True, True], True),
        ([True, None], None),
        ([False, None], False),
    ],
)
def test_custom_rubric_conjunction(clip, states, expected):
    rubric = [
        {"id": "placement", "description": "Bowl visibly on shelf"},
        {"id": "release", "description": "Gripper visibly releases bowl"},
    ]
    result = answer()
    result["checks"] = [
        {**deepcopy(answer(state)["checks"][0]), "id": rule["id"]}
        for state, rule in zip(states, rubric, strict=True)
    ]
    report = glimpse.judge_video(
        clip,
        instruction="Place and release bowl",
        rubric=rubric,
        output=clip.parent / "audit",
        request_fn=lambda *a, **k: response(result),
    )
    assert report["answer"]["visible_success"] is expected


@pytest.mark.parametrize("failure", [False, True])
def test_publisher_selects_glimpse_without_changing_score(
    tmp_path, monkeypatch, failure
):
    trial = {"task": "OpenOven", "verified_success": False}
    record = {"status": "verified_fail", "artifacts": {}, "video": {"views": []}}
    before = deepcopy(trial)

    def judge(video, **kwargs):
        assert "model" not in kwargs  # Solver identity must not select the judge.
        assert kwargs["mechanical_success"] is False
        if failure:
            raise TimeoutError("test timeout")
        return {"status": "disagrees", "changes_score": False}

    monkeypatch.setattr(glimpse, "judge_video", judge)
    from roboswe.library import predicate_audit
    monkeypatch.setattr(predicate_audit, "predicate_contract", lambda task: {"task": task, "rubric": None, "sources": []})
    result = publish_visual_audit(
        tmp_path,
        record,
        trial,
        model="solver-model",
        backend="glimpse",
    )
    assert result["status"] == ("judge_failed" if failure else "disagrees")
    assert record["status"] == "verified_fail"
    assert trial == before
    assert record["artifacts"]["visual_audit"] == "audit.json"


def test_http_errors_do_not_leak_body_or_retry(monkeypatch):
    calls = []

    def fail(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(glimpse.ENDPOINT, 429, "secret body", {}, None)

    monkeypatch.setattr(glimpse.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="HTTP 429") as exc:
        glimpse._request({}, api_key="secret-key", timeout=1)
    assert "secret" not in str(exc.value)
    assert len(calls) == 1


def test_cli_runs_only_requested_audit(clip, monkeypatch, capsys):
    monkeypatch.setattr(glimpse, "_request", lambda *a, **k: response())
    assert (
        glimpse.main(
            [
                str(clip),
                "--instruction",
                "Test goal",
                "--output",
                str(clip.parent / "audit"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "uncompared"


def test_existing_output_is_preserved(clip):
    output = clip.parent / "audit"
    output.mkdir()
    (output / "audit.json").write_text("existing evidence")
    with pytest.raises(FileExistsError):
        glimpse.judge_video(clip, instruction="Test goal", output=output)
    assert (output / "audit.json").read_text() == "existing evidence"
