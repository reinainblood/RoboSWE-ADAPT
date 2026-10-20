import base64

import pytest

from roboswe.campaign import check_capacity, model_input, require_execution


def plan():
    return {
        "execution": {
            "backend": "docker",
            "concurrency": 1,
            "image": "sha256:" + "a" * 64,
            "code_image": "sha256:" + "b" * 64,
        },
        "agent": {"api_key_env": None, "api_base_url": None},
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", "mutable:latest"),
        ("code_image", None),
        ("backend", "modal"),
        ("concurrency", 2),
    ],
)
def test_unsupported_execution_rejected_before_allocation(field, value):
    config = plan()
    config["execution"][field] = value
    with pytest.raises(ValueError):
        require_execution(config)


def test_login_connection_cannot_silently_use_an_api_key():
    config = plan()
    config["agent"]["api_key_env"] = "OPENAI_API_KEY"
    with pytest.raises(ValueError, match="local Codex login"):
        require_execution(config)


def test_observe_feedback_does_not_dump_base64_into_model_text(tmp_path):
    raw = base64.b64encode(b"sensor-image-bytes").decode()
    observation = {
        "images_png_base64": {"camera": raw},
        "control_step": 2,
        "sensors": {},
        "depth_npz_base64": "DO_NOT_PUT_DEPTH_BLOB_IN_MODEL_TEXT",
    }
    text, paths = model_input(observation, observation, tmp_path, 0)
    assert raw not in text
    assert "DO_NOT_PUT_DEPTH_BLOB_IN_MODEL_TEXT" not in text
    assert paths[0].read_bytes() == b"sensor-image-bytes"
    assert "depth_arrays is metadata ONLY" in text
    assert "calibration from the same observation" in text


def test_capacity_includes_separate_code_workspace_and_prevents_overlap():
    execution = {"memory_mb": 6144, "cpu": 4}
    info = {"MemTotal": 8 * 2**30, "NCPU": 8}
    check_capacity(execution, info, [])
    with pytest.raises(ValueError, match="already running"):
        check_capacity(execution, info, ["existing-simulator"])
    with pytest.raises(ValueError, match="Docker has"):
        check_capacity({**execution, "memory_mb": 16384}, info, [])


def test_reference_lease_spans_gaps_without_a_running_container(tmp_path, monkeypatch):
    from roboswe.campaign import reference_lease

    monkeypatch.setattr("roboswe.campaign.tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr("roboswe.campaign.ensure_docker_capacity", lambda _: {})
    with (
        reference_lease({}),
        pytest.raises(ValueError, match="active on this host"),
        reference_lease({}),
    ):
        pass
    with reference_lease({}):
        pass
