"""Durable execution records. Successful execution is not a benchmark score."""

import hashlib
import json
import shutil
import subprocess
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .scene import capture_scene, scene_metadata


def now():
    return datetime.now(UTC).isoformat()


def save(job_dir, record):
    temp = job_dir / "job.json.tmp"
    temp.write_text(json.dumps(record, indent=2) + "\n")
    temp.replace(job_dir / "job.json")


def create(jobs_dir, name, kind, handle=None):
    job_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:10]
    job_dir = Path(jobs_dir).resolve() / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "id": job_id,
        "name": name,
        "kind": kind,
        "created_at": now(),
        "finished_at": None,
        "status": "running",
        "exit_code": None,
        "artifacts": {},
        "scenes": {},
        "artifact_errors": [],
    }
    if handle is not None:
        record["session"] = asdict(handle)
    save(job_dir, record)
    return job_dir, record


def capture(job_dir, record, handle, label):
    filename = f"{label}.scene.json.gz"
    try:
        metadata = capture_scene(handle, job_dir / filename)
        record["artifacts"][label] = filename
        record["scenes"][label] = metadata
    except Exception as exc:  # noqa: BLE001 -- Record provider/renderer failures without hiding code output.
        record["artifact_errors"].append(
            {"stage": label, "error": f"{type(exc).__name__}: {exc}"}
        )
    save(job_dir, record)


def run_job(
    handle,
    source,
    *,
    jobs_dir="jobs",
    timeout=600,
    name=None,
    record_video=True,
    video_width=640,
    video_height=480,
):
    """Run exact saved source once; retain logs and independent before/after captures."""
    source = Path(source)
    source_bytes = source.read_bytes()
    job_dir, record = create(jobs_dir, name or source.name, "execution", handle)
    submitted = job_dir / "source.py"
    submitted.write_bytes(source_bytes)
    record["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    record["artifacts"]["source"] = "source.py"
    record["timeout_seconds"] = timeout
    save(job_dir, record)
    capture(job_dir, record, handle, "before")
    recording_started = False
    if record_video:
        try:
            recording = json.loads(
                handle.request(
                    "/recording/start",
                    {
                        "recording_id": record["id"],
                        "width": video_width,
                        "height": video_height,
                    },
                )
            )
            recording_started = True
            record["video"] = recording
        except Exception as exc:  # noqa: BLE001 -- Do not silently execute robot code without the requested recording.
            record.update(status="recording_failed", exit_code=1, finished_at=now())
            record["artifact_errors"].append(
                {"stage": "video_start", "error": str(exc)}
            )
            save(job_dir, record)
            return job_dir, subprocess.CompletedProcess(
                [],
                1,
                b"",
                f"Recording could not start; code was not executed. Rebuild the session image for recording support, or explicitly choose --no-video.\n{exc}\n".encode(),
            )
    record["execution_started_at"] = now()
    save(job_dir, record)
    try:
        result = handle.run_code(submitted, timeout=timeout)
        record["exit_code"] = result.returncode
        record["status"] = "completed" if result.returncode == 0 else "failed"
    except Exception:  # noqa: BLE001 -- Preserve diagnostics for any backend failure at the job boundary.
        result = subprocess.CompletedProcess(
            [], 1, b"", traceback.format_exc().encode()
        )
        record["status"] = "error"
        record["exit_code"] = 1
    except KeyboardInterrupt:
        result = subprocess.CompletedProcess(
            [], 130, b"", b"Host run interrupted; remote command state is unknown.\n"
        )
        record["status"] = "interrupted"
        record["exit_code"] = 130
    record["execution_finished_at"] = now()
    (job_dir / "stdout.txt").write_bytes(result.stdout)
    (job_dir / "stderr.txt").write_bytes(result.stderr)
    record["artifacts"].update(stdout="stdout.txt", stderr="stderr.txt")
    save(job_dir, record)
    if recording_started:
        try:
            manifest = json.loads(
                handle.request(
                    "/recording/stop",
                    {"recording_id": record["id"], "reason": record["status"]},
                )
            )
            record["video"] = manifest
            for filename in manifest["files"]:
                if (
                    Path(filename).name != filename
                    or filename not in {"video.json", "video-timeline.jsonl"}
                    and not (
                        filename.startswith("video-") and filename.endswith(".mp4")
                    )
                ):
                    raise ValueError("Invalid recording filename")
                try:
                    data = handle.request(f"/recording/{record['id']}/{filename}")
                    (job_dir / filename).write_bytes(data)
                    record["artifacts"][filename] = filename
                except Exception as exc:  # noqa: BLE001 -- Retain the other videos if one transfer fails.
                    record["artifact_errors"].append(
                        {"stage": "video_download", "error": f"{filename}: {exc}"}
                    )
            for error in manifest.get("errors", []):
                record["artifact_errors"].append(
                    {"stage": "video_recording", "error": error}
                )
        except Exception as exc:  # noqa: BLE001 -- Execution evidence remains useful when recording finalization fails.
            record["artifact_errors"].append({"stage": "video_stop", "error": str(exc)})
        save(job_dir, record)
    if record["status"] != "interrupted":
        capture(job_dir, record, handle, "after")
    record["finished_at"] = now()
    save(job_dir, record)
    return job_dir, result


def capture_job(handle, *, jobs_dir="jobs", name="Session snapshot"):
    job_dir, record = create(jobs_dir, name, "snapshot", handle)
    capture(job_dir, record, handle, "after")
    record["status"] = "captured" if "after" in record["artifacts"] else "error"
    record["finished_at"] = now()
    save(job_dir, record)
    return job_dir, record


def import_scene(source, *, jobs_dir="jobs", name=None, handle=None):
    """Import a historical visual artifact without claiming to have run a job."""
    source = Path(source)
    metadata = scene_metadata(source.read_bytes())
    job_dir, record = create(jobs_dir, name or source.stem, "snapshot", handle)
    shutil.copyfile(source, job_dir / "after.scene.json.gz")
    record["artifacts"]["after"] = "after.scene.json.gz"
    record["scenes"]["after"] = metadata
    record["status"] = "captured"
    record["finished_at"] = now()
    record["imported_from"] = str(source.resolve())
    save(job_dir, record)
    return job_dir
