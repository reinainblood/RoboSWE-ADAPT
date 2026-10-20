"""Read-only loopback viewer. Never connects to a robot or executes submitted code."""

import json
import mimetypes
import re
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .scene import ASSETS

JOB_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{10}$")


def job_records(root):
    records = []
    for path in sorted(root.glob("*/job.json"), reverse=True):
        if (
            not JOB_ID.fullmatch(path.parent.name)
            or path.is_symlink()
            or path.parent.is_symlink()
        ):
            continue
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict) or record.get("schema_version") != 1:
                continue
            if record.get("id") != path.parent.name:
                continue
            if not isinstance(record.get("artifacts"), dict) or not isinstance(
                record.get("scenes"), dict
            ):
                continue
            records.append(record)
        except (OSError, ValueError):
            continue
    return records


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, jobs_dir, **kwargs):
        self.jobs_dir = Path(jobs_dir).resolve()
        super().__init__(*args, **kwargs)

    def log_message(self, *_):
        pass

    def do_GET(self):
        route = unquote(urlsplit(self.path).path)
        if route == "/api/jobs":
            body = json.dumps(job_records(self.jobs_dir)).encode()
            return self.respond(body, "application/json")
        if route == "/":
            return self.file(ASSETS / "jobs.html")
        if route.startswith("/assets/"):
            relative = route.removeprefix("/assets/")
            allowed = {
                "jobs.js",
                "scene.js",
                "video.js",
                "vendor/three.module.js",
                "vendor/three.core.min.js",
                "vendor/OrbitControls.js",
            }
            if relative in allowed:
                return self.file(ASSETS / relative)
        parts = route.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "jobs" and JOB_ID.fullmatch(parts[1]):
            job_dir = self.jobs_dir / parts[1]
            path = job_dir / parts[2]
            if (
                job_dir.is_symlink()
                or path.is_symlink()
                or not path.resolve().is_relative_to(self.jobs_dir)
            ):
                return self.send_error(404)
            allowed = {
                "job.json",
                "source.py",
                "stdout.txt",
                "stderr.txt",
                "before.scene.json.gz",
                "after.scene.json.gz",
                "video.json",
                "video-timeline.jsonl",
                "report.json",
                "plan.json",
                "audit.json",
                "library.zip",
                "NOTES.md",
                "coordinator.json",
                "coordinator-source.zip",
                "source.zip",
                "cohort.json",
                "progress.json",
                "execution-order.json",
                "timing.jsonl",
                "model-transcript.jsonl",
                "provider-responses.jsonl",
                "protocol-recovery.jsonl",
                "actions.jsonl",
                "submissions.jsonl",
                "replay-submissions.jsonl",
                "continuity.json",
            }
            if parts[2] in allowed or re.fullmatch(
                r"(?:(?:original|replay)-)?video-(overview|camera-\d+)-(wall|simulation)\.mp4", parts[2]
            ) or re.fullmatch(
                r"(?:original|replay)-video(?:-timeline\.jsonl|\.json)", parts[2]
            ):
                return self.file(path)
        self.send_error(404)

    def do_POST(self):
        self.send_error(405, "The viewer is read-only")

    def do_HEAD(self):
        self.do_GET()

    def respond(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def file(self, path):
        if not path.is_file():
            return self.send_error(404)
        content_type = (
            "application/octet-stream"
            if path.suffix == ".gz"
            else mimetypes.guess_type(path)[0] or "text/plain"
        )
        if path.suffix in (".py", ".txt"):
            content_type = "text/plain; charset=utf-8"
        if path.suffix == ".mp4":
            size = path.stat().st_size
            start, end = 0, size - 1
            value = self.headers.get("Range")
            if value:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
                try:
                    if match is None or not any(match.groups()):
                        raise ValueError
                    left, right = match.groups()
                    if not left:
                        length = int(right)
                        if length <= 0:
                            raise ValueError
                        start = max(0, size - length)
                    else:
                        start = int(left)
                        end = min(int(right), end) if right else end
                    if start > end or start >= size:
                        raise ValueError
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if value else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if value:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with path.open("rb") as source:
                source.seek(start)
                remaining = end - start + 1
                try:
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        self.respond(path.read_bytes(), content_type)


def make_server(jobs_dir, port=0):
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, jobs_dir=jobs_dir))


def serve(jobs_dir="jobs", port=0, open_browser=True):
    root = Path(jobs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with make_server(root, port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(
            f"RoboSWE jobs: {url}\nReading {root}\nPress Ctrl+C to close the viewer server.",
            flush=True,
        )
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
