import io
import json
from pathlib import Path

import pytest

from pairing.modal_backup import inventory, upload


class FakeVolume:
    def __init__(self):
        self.files = {}
        self.writes = []
        self.corrupt = False
        self.batches = 0
        self.interrupt_after = None

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    def batch_upload(self, *, force):
        assert force is False
        self.batches += 1
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def put_file(self, source, path):
        if self.interrupt_after is not None and len(self.writes) >= self.interrupt_after:
            raise RuntimeError("simulated interruption")
        assert path not in self.files
        data = source.read() if isinstance(source, io.BytesIO) else Path(source).read_bytes()
        self.files[path] = b"corrupt" if self.corrupt else data
        self.writes.append(path)


@pytest.fixture
def job(tmp_path):
    root = tmp_path / "test-job"
    root.mkdir()
    (root / "job.json").write_text(json.dumps({"id": root.name, "finished_at": "2026-10-05T00:00:00Z", "status": "observed_fail"}))
    (root / "video-camera-0-simulation.mp4").write_bytes(b"video bytes")
    (root / "NOTES.md").write_text("retained learning")
    (root / "recordings").mkdir()
    (root / "recordings" / "duplicate.mp4").write_bytes(b"video bytes")
    return root


def test_inventory_is_stable_top_level_and_rejects_active_jobs(job):
    first = inventory(job)
    assert first["snapshot"] == inventory(job)["snapshot"]
    assert {f["path"] for f in first["manifest"]["files"]} == {"job.json", "NOTES.md", "video-camera-0-simulation.mp4"}
    metadata = json.loads((job / "job.json").read_text())
    metadata["status"] = "running"
    (job / "job.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not finalized"):
        inventory(job)


def test_upload_readback_manifest_last_and_resume_without_overwrite(job, tmp_path):
    volume = FakeVolume()
    item = inventory(job)
    result = upload(item, volume=volume, receipt_root=tmp_path / "receipts")
    assert volume.writes[-1].endswith("/backup-manifest.json")
    assert volume.batches == 2  # One artifact batch, then the verified manifest.
    assert Path(result["local_receipt"]).exists()
    count = len(volume.writes)
    resumed = upload(item, volume=volume, receipt_root=tmp_path / "receipts")
    assert len(volume.writes) == count
    assert volume.batches == 2
    assert all(f["status"] == "already_verified" for f in resumed["files"])
    assert (job / "video-camera-0-simulation.mp4").exists()


def test_corrupt_upload_never_publishes_completion_or_receipt(job, tmp_path):
    volume = FakeVolume()
    volume.corrupt = True
    with pytest.raises(ValueError, match="readback"):
        upload(inventory(job), volume=volume, receipt_root=tmp_path / "receipts")
    assert not any(p.endswith("backup-manifest.json") for p in volume.files)
    assert not (tmp_path / "receipts").exists()


def test_changed_source_and_symlink_rejected(job, tmp_path):
    item = inventory(job)
    (job / "NOTES.md").write_text("changed")
    with pytest.raises(ValueError, match="changed since inventory"):
        upload(item, volume=FakeVolume(), receipt_root=tmp_path / "receipts")
    (job / "external.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(ValueError, match="Symlink"):
        inventory(job)


def test_changed_job_creates_new_snapshot_and_remote_conflict_is_preserved(job, tmp_path):
    old = inventory(job)
    (job / "NOTES.md").write_text("new findings")
    new = inventory(job)
    assert old["remote_directory"] != new["remote_directory"]
    volume = FakeVolume()
    conflict = new["remote_directory"] + "/NOTES.md"
    volume.files[conflict] = b"preserve this"
    with pytest.raises(ValueError, match="refusing overwrite"):
        upload(new, volume=volume, receipt_root=tmp_path / "receipts")
    assert volume.files[conflict] == b"preserve this"


def test_interrupted_batch_resumes_missing_files_only(job, tmp_path):
    volume = FakeVolume()
    volume.interrupt_after = 1
    item = inventory(job)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        upload(item, volume=volume, receipt_root=tmp_path / "receipts")
    assert len(volume.files) == 1
    assert not any(path.endswith("backup-manifest.json") for path in volume.files)
    volume.interrupt_after = None
    result = upload(item, volume=volume, receipt_root=tmp_path / "receipts")
    assert len(volume.files) == item["file_count"] + 1
    assert sum(file["status"] == "already_verified" for file in result["files"]) == 1
    assert Path(result["local_receipt"]).exists()


def test_explicit_incomplete_recording_backup_preserves_warning_and_never_live_job(job):
    (job / "video.json").write_text(json.dumps({"status": "recording"}))
    with pytest.raises(ValueError, match="Recording is still active"):
        inventory(job)
    saved = inventory(job, include_incomplete_recordings=True)
    assert saved["manifest"]["unfinished_recording_manifests"] == ["video.json"]
    assert "Uncertified possibly incomplete recording" in saved["manifest"]["limitations"][0]
    metadata = json.loads((job / "job.json").read_text())
    metadata["status"] = "running"
    (job / "job.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not finalized"):
        inventory(job, include_incomplete_recordings=True)
