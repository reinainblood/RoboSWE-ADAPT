import json
from zipfile import ZipFile

import pytest

from roboswe.evaluate import evaluate_campaign
from roboswe.provenance import snapshot_harness


def test_source_snapshot_excludes_login_config_and_unrelated_documents(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    for directory in ("roboswe", "robot_session", "docs", ".codex"):
        (root / directory).mkdir(parents=True)
    (root / "roboswe/code.py").write_text("value = 1\n")
    (root / "robot_session/core.py").write_text("value = 2\n")
    (root / ".codex/auth.json").write_text("DO_NOT_ARCHIVE")
    (root / ".env").write_text("DO_NOT_ARCHIVE")
    (root / "docs/draft.md").write_text("DO_NOT_ARCHIVE")
    monkeypatch.setattr(
        "roboswe.provenance.__file__", str(root / "roboswe/provenance.py")
    )
    output = tmp_path / "output"
    output.mkdir()
    result = snapshot_harness(output)
    with ZipFile(output / "coordinator-source.zip") as archive:
        assert set(archive.namelist()) == {"roboswe/code.py", "robot_session/core.py"}
    assert len(result["source_manifest_sha256"]) == 64


def test_evaluate_does_not_start_resources_for_active_campaign(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="active campaign"):
        evaluate_campaign(tmp_path)
