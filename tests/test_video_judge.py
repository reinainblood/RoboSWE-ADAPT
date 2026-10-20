import json
from copy import deepcopy

import pytest

from roboswe.library.video_judge import judge_video, publish_visual_audit


@pytest.mark.parametrize("failure", [False, True])
def test_published_audit_cannot_replace_mechanical_verdict(
    tmp_path, monkeypatch, failure
):
    trial = {"task": "OpenOven", "verified_success": False}
    before = deepcopy(trial)
    record = {
        "status": "verified_fail",
        "artifacts": {},
        "video": {
            "views": [
                {"id": "overview", "simulation": "video-overview-simulation.mp4"},
                {"id": "camera-0", "label": "wrist", "simulation": "wrist.mp4"},
            ]
        },
    }

    def audit(video, **kwargs):
        assert video == tmp_path / "video-overview-simulation.mp4"
        assert kwargs["additional_videos"] == {"wrist": tmp_path / "wrist.mp4"}
        assert kwargs["mechanical_success"] is False
        if failure:
            raise OSError("Video cannot be decoded")
        return {"status": "disagrees", "changes_score": False}

    monkeypatch.setattr("roboswe.library.video_judge.judge_video", audit)
    result = publish_visual_audit(tmp_path, record, trial, model="test-model")
    assert result["status"] == ("judge_failed" if failure else "disagrees")
    assert record["status"] == "verified_fail"
    assert trial == before
    assert record["artifacts"]["visual_audit"] == "audit.json"
    assert json.loads((tmp_path / "audit.json").read_text()) == result


def test_real_video_sampling_keeps_judge_blind_and_discloses_coverage(tmp_path):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    video = tmp_path / "clip.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for tick in range(20):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), tick * 10, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    class BlindJudge:
        def __init__(self, **kwargs):
            self.metadata = {"model": "test-only"}
            assert "blind visual auditor" in kwargs["instructions"]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def ask(self, text, *, images, **kwargs):
            prompt = json.loads(text)
            assert "mechanical_success" not in prompt
            assert prompt["coverage"] == "sampled_frames"
            assert len(images) == 16
            assert all(path.exists() for path in images)
            assert prompt["success_criteria"] == {"minimum_ticks": 10}
            return {
                "visible_success": True,
                "confidence": "low",
                "visible_motion": "Test response only",
                "reason": "Testing disagreement publication",
            }

    report = judge_video(
        video,
        instruction="Test goal",
        mechanical_success=False,
        output=tmp_path / "audit",
        max_frames=16,
        client_factory=BlindJudge,
        criteria={"minimum_ticks": 10},
    )
    assert report["status"] == "disagrees"
    assert report["changes_score"] is False
    indices = [frame["index"] for frame in report["reviewed_frames"]]
    assert indices[-11:] == list(range(9, 20))
    assert report["views"][0]["total_frames"] == 20
