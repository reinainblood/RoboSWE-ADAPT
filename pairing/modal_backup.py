"""Immutable, hash-verified backups of finalized RoboSWE job artifacts in Modal FDR.

Dry-run inventory is the default. Uploads never remove local files or overwrite
remote files. A manifest is published only after every file passes readback.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

ENVIRONMENT = "FDR"
VOLUME_NAME = "roboswe-job-artifacts-v1"
ALLOWED_SUFFIXES = {".mp4", ".json", ".jsonl", ".py", ".txt", ".md", ".zip", ".gz", ".npz", ".png", ".jpg", ".html"}
ACTIVE_STATUSES = {"running", "queued", "starting", "waiting_for_current_run", "recording", "pending"}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def digest_chunks(chunks):
    digest = hashlib.sha256()
    size = 0
    for chunk in chunks:
        digest.update(chunk)
        size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def digest_file(path):
    before = path.stat()
    with path.open("rb") as stream:
        result = digest_chunks(iter(lambda: stream.read(1024 * 1024), b""))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Artifact changed while being inventoried: {path.name}")
    return result


def inventory(directory, *, include_incomplete_recordings=False):
    directory = Path(directory)
    job_path = directory / "job.json"
    if job_path.is_symlink():
        raise ValueError("Job metadata must not be a symlink")
    job = json.loads(job_path.read_text())
    if not job.get("finished_at") or job.get("status") in ACTIVE_STATUSES:
        raise ValueError("Job is not finalized; defer until finished_at is saved")
    if job.get("id") != directory.name:
        raise ValueError("Job ID must match its directory name")
    files = []
    unfinished_recordings = []
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            raise ValueError(f"Symlink is not a backup artifact: {path.name}")
        if not path.is_file() or path.name.startswith(".") or path.suffix not in ALLOWED_SUFFIXES:
            continue
        if path.name.endswith("video.json"):
            recording = json.loads(path.read_text())
            if recording.get("status") == "recording":
                if not include_incomplete_recordings:
                    raise ValueError(f"Recording is still active: {path.name}")
                unfinished_recordings.append(path.name)
        files.append({"path": path.name, **digest_file(path)})
    manifest = {"schema_version": 1, "job_id": job["id"], "files": files,
                "scope": "Finalized top-level artifacts; nested recordings and model-images excluded; local files retained"}
    if unfinished_recordings:
        manifest["unfinished_recording_manifests"] = unfinished_recordings
        manifest["limitations"] = ["Uncertified possibly incomplete recording: saved from a finalized job whose recording metadata was not finalized. Byte verification does not certify playback completeness."]
    snapshot = hashlib.sha256(encoded(manifest)).hexdigest()
    return {"directory": str(directory.resolve()), "job_id": job["id"],
            "snapshot": snapshot, "remote_directory": f"/jobs/{job['id']}/snapshots/{snapshot}",
            "file_count": len(files), "bytes": sum(f["bytes"] for f in files), "manifest": manifest}


def remote_digest(volume, path):
    try:
        return digest_chunks(volume.read_file(path))
    except FileNotFoundError:
        return None
    except Exception as exc:
        # Modal's lazy RPC iterator raises NotFoundError rather than OSError.
        if type(exc).__name__ == "NotFoundError":
            return None
        raise


def ensure_file(volume, source, remote_path, expected):
    existing = remote_digest(volume, remote_path)
    if existing is not None:
        if existing != expected:
            raise ValueError(f"Remote immutable artifact differs; refusing overwrite: {remote_path}")
        return "already_verified"
    with volume.batch_upload(force=False) as batch:
        batch.put_file(source, remote_path)
    actual = remote_digest(volume, remote_path)
    if actual != expected:
        raise ValueError(f"Remote readback verification failed: {remote_path}")
    return "uploaded_verified"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def upload(inventoried, *, volume=None, receipt_root=Path("artifacts/modal-backups")):
    if volume is None:
        import modal
        volume = modal.Volume.from_name(VOLUME_NAME, environment_name=ENVIRONMENT, create_if_missing=True)
    directory = Path(inventoried["directory"])
    results = []
    missing = []
    for item in inventoried["manifest"]["files"]:
        path = directory / item["path"]
        expected = {key: item[key] for key in ("bytes", "sha256")}
        if path.is_symlink() or digest_file(path) != expected:
            raise ValueError(f"Local artifact changed since inventory: {item['path']}")
        remote_path = inventoried["remote_directory"] + "/" + item["path"]
        existing = remote_digest(volume, remote_path)
        if existing is not None:
            if existing != expected:
                raise ValueError(f"Remote immutable artifact differs; refusing overwrite: {remote_path}")
            results.append({"path": item["path"], "status": "already_verified"})
        else:
            missing.append((item["path"], path, remote_path, expected))
    # One batch per job allows the SDK to upload independent files concurrently.
    # The terminal manifest remains a separate, later commit after readback.
    if missing:
        with volume.batch_upload(force=False) as batch:
            for _name, path, remote_path, _expected in missing:
                batch.put_file(path, remote_path)
        for name, _path, remote_path, expected in missing:
            if remote_digest(volume, remote_path) != expected:
                raise ValueError(f"Remote readback verification failed: {remote_path}")
            results.append({"path": name, "status": "uploaded_verified"})
    results.sort(key=lambda item: item["path"])
    # Publishing this terminal manifest is the completion marker. Partial upload
    # directories are resumable and do not have this marker.
    body = encoded(inventoried["manifest"])
    manifest_path = inventoried["remote_directory"] + "/backup-manifest.json"
    ensure_file(volume, io.BytesIO(body), manifest_path, digest_chunks([body]))
    receipt = {"schema_version": 1, "job_id": inventoried["job_id"],
               "snapshot": inventoried["snapshot"], "volume": VOLUME_NAME,
               "environment": ENVIRONMENT, "remote_manifest": manifest_path,
               "verified_at": datetime.now(UTC).isoformat(),
               "verification": "Every remote file read back and matched for bytes and SHA-256",
               "local_files_retained": True, "bytes": inventoried["bytes"], "files": results}
    receipt_path = Path(receipt_root) / inventoried["job_id"] / (inventoried["snapshot"] + ".json")
    atomic_json(receipt_path, receipt)
    return {**receipt, "local_receipt": str(receipt_path.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="*", type=Path, help="Selected finalized job directories")
    parser.add_argument("--all-finalized", action="store_true")
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
    parser.add_argument("--upload", action="store_true", help="Upload and verify; default only inventories local files")
    parser.add_argument("--include-incomplete-recordings", action="store_true", help="Preserve possibly incomplete recordings from finalized jobs; never includes active jobs")
    parser.add_argument("--receipt-root", type=Path, default=Path("artifacts/modal-backups"))
    args = parser.parse_args()
    if not args.jobs and not args.all_finalized:
        parser.error("Specify job directories or --all-finalized")
    directories = set(args.jobs)
    if args.all_finalized:
        directories.update(p.parent for p in args.jobs_dir.glob("*/job.json"))
    failures = 0
    for directory in sorted(directories):
        try:
            item = inventory(directory, include_incomplete_recordings=args.include_incomplete_recordings)
            result = upload(item, receipt_root=args.receipt_root) if args.upload else item
            print(json.dumps({"mode": "uploaded_verified" if args.upload else "dry_run", **result}), flush=True)
        except Exception as exc:  # noqa: BLE001 -- record each failed job, continue others, exit nonzero
            # Explicitly selected unfinished jobs are errors; all-finalized scans
            # legitimately skip jobs still being written.
            skipped = args.all_finalized and isinstance(exc, ValueError) and "not finalized" in str(exc)
            if not skipped:
                failures += 1
            print(json.dumps({"job": str(directory), "status": "skipped" if skipped else "error", "error": str(exc)}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
