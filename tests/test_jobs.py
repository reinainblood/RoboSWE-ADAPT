import gzip
import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from roboswe.jobs import capture_job, import_scene, run_job
from roboswe.scene import ASSETS, inline_scene_script
from roboswe.server import make_server


def test_original_and_replay_video_keep_distinct_files_and_timelines(tmp_path):
    from roboswe.campaign import publish_video
    from roboswe.jobs import create, save

    root = tmp_path / "jobs"
    directory, record = create(root, "Two recordings", "benchmark-task")
    for phase in ("original", "replay"):
        source = tmp_path / phase
        source.mkdir()
        manifest = {
            "views": [{"id": "overview", "wall": "video-overview-wall.mp4"}],
            "files": ["video.json", "video-timeline.jsonl", "video-overview-wall.mp4"],
        }
        (source / "video.json").write_text(json.dumps(manifest))
        (source / "video-timeline.jsonl").write_text(phase + " timeline")
        (source / "video-overview-wall.mp4").write_bytes(phase.encode())
        publish_video(source, directory, record, phase=phase)
    save(directory, record)
    assert record["default_video_run"] == "replay"
    assert (directory / "video-overview-wall.mp4").read_bytes() == b"replay"
    server = make_server(root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/jobs/{directory.name}/"
    try:
        for phase in ("original", "replay"):
            variant = record["video_runs"][phase]
            with urlopen(base + variant["views"][0]["wall"]) as response:
                assert response.read() == phase.encode()
            with urlopen(base + variant["timeline"]) as response:
                assert response.read() == (phase + " timeline").encode()
            with urlopen(base + f"{phase}-video.json") as response:
                assert json.load(response) == variant
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def scene(step=0):
    return gzip.compress(
        json.dumps(
            {
                "step": step,
                "session_id": "test-session",
                "task": "test-task",
                "meshes": {},
                "geoms": [],
                "textures": {},
            }
        ).encode()
    )


@dataclass
class FakeSession:
    backend: str = "test"
    fail_capture: bool = False
    exit_code: int = 0
    step: int = 0

    def execute(self, *args, **kwargs):
        assert args == ("python", "-")
        assert b"mj_setState" in kwargs["stdin"]
        if self.fail_capture:
            return subprocess.CompletedProcess(args, 1, b"", b"rendering unavailable")
        return subprocess.CompletedProcess(args, 0, scene(self.step), b"")

    def request(self, path, payload=None):
        if path == "/recording/start":
            return json.dumps(
                {"recording_id": payload["recording_id"], "status": "recording"}
            ).encode()
        if path == "/recording/stop":
            return json.dumps(
                {
                    "files": [
                        "video.json",
                        "video-timeline.jsonl",
                        "video-overview-wall.mp4",
                    ],
                    "views": [{"wall": "video-overview-wall.mp4"}],
                    "errors": [],
                    "status": payload["reason"],
                }
            ).encode()
        return b"saved recording artifact"

    def run_code(self, source, timeout):
        assert Path(source).read_bytes() == b"print('hello')\n"
        assert timeout == 12
        self.step = 3
        return subprocess.CompletedProcess(
            [], self.exit_code, b"hello\n", b"diagnostic\n"
        )


@pytest.mark.parametrize("exit_code,status", [(0, "completed"), (2, "failed")])
def test_real_output_contract(tmp_path, exit_code, status):
    source = tmp_path / "input.py"
    source.write_bytes(b"print('hello')\n")
    directory, result = run_job(
        FakeSession(exit_code=exit_code), source, jobs_dir=tmp_path / "jobs", timeout=12
    )
    record = json.loads((directory / "job.json").read_text())
    assert record["status"] == status
    assert (
        directory / "video-overview-wall.mp4"
    ).read_bytes() == b"saved recording artifact"
    assert result.returncode == exit_code
    assert record["scenes"]["before"]["step"] == 0
    assert record["scenes"]["after"]["step"] == 3
    assert (directory / "source.py").read_bytes() == source.read_bytes()
    assert (directory / "stdout.txt").read_bytes() == b"hello\n"
    assert (directory / "stderr.txt").read_bytes() == b"diagnostic\n"
    source.write_text("changed later")
    assert (directory / "source.py").read_bytes() == b"print('hello')\n"


def test_capture_failure_does_not_hide_execution(tmp_path):
    source = tmp_path / "input.py"
    source.write_bytes(b"print('hello')\n")
    directory, result = run_job(
        FakeSession(fail_capture=True), source, jobs_dir=tmp_path / "jobs", timeout=12
    )
    record = json.loads((directory / "job.json").read_text())
    assert result.returncode == 0 and record["status"] == "completed"
    assert [e["stage"] for e in record["artifact_errors"]] == ["before", "after"]
    assert "after" not in record["artifacts"]
    assert record["finished_at"]


def test_exception_and_timeout_retain_record(tmp_path):
    class Crashed(FakeSession):
        def run_code(self, source, timeout):
            raise TimeoutError("execution timed out")

    source = tmp_path / "input.py"
    source.write_bytes(b"print('hello')\n")
    directory, result = run_job(
        Crashed(), source, jobs_dir=tmp_path / "jobs", timeout=12
    )
    record = json.loads((directory / "job.json").read_text())
    assert result.returncode == 1 and record["status"] == "error"
    assert b"execution timed out" in (directory / "stderr.txt").read_bytes()
    assert record["scenes"]["after"]["step"] == 0


def test_snapshot_is_not_reported_as_execution(tmp_path):
    _directory, record = capture_job(FakeSession(), jobs_dir=tmp_path)
    assert record["kind"] == "snapshot" and record["status"] == "captured"
    assert record["exit_code"] is None
    assert record["scenes"]["after"]["step"] == 0


def test_read_only_server_and_path_boundaries(tmp_path):
    saved = tmp_path / "scene.gz"
    saved.write_bytes(scene())
    root = tmp_path / "jobs"
    directory = import_scene(
        saved, jobs_dir=root, name="<script>do not execute</script>"
    )
    secret = tmp_path / "outside.txt"
    secret.write_text("outside")
    (directory / "source.py").symlink_to(secret)
    server = make_server(root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/jobs") as r:
            records = json.load(r)
        assert records[0]["name"] == "<script>do not execute</script>"
        with urlopen(base + f"/jobs/{directory.name}/after.scene.json.gz") as r:
            assert r.read() == saved.read_bytes()
        for suffix in (
            f"/jobs/{directory.name}/source.py",
            "/assets/../jobs.py",
            "/jobs/../../outside.txt",
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(base + suffix)
            assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/jobs", data=b"{}", method="POST"))
        assert error.value.code == 405
        with urlopen(base + "/assets/scene.js") as r:
            assert b"https://esm.sh" not in r.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_javascript_is_bundled_for_offline_use():
    script = inline_scene_script()
    assert "https://esm.sh" not in script
    assert "from './vendor/" not in script
    assert "data:text/javascript;base64," in script
    assert (ASSETS / "vendor/THREE-LICENSE.txt").is_file()


def test_missing_recorder_does_not_run_unrecorded(tmp_path):
    class Legacy(FakeSession):
        def request(self, path, payload=None):
            raise RuntimeError("recording endpoint unavailable")

        def run_code(self, *args, **kwargs):
            pytest.fail("Must not execute after recording startup failure")

    source = tmp_path / "code.py"
    source.write_text("print('hello')\n")
    directory, result = run_job(Legacy(), source, jobs_dir=tmp_path / "jobs")
    assert result.returncode == 1
    assert (
        json.loads((directory / "job.json").read_text())["status"] == "recording_failed"
    )


def test_video_range_requests(tmp_path):
    saved = tmp_path / "scene.gz"
    saved.write_bytes(scene())
    directory = import_scene(saved, jobs_dir=tmp_path / "jobs")
    data = bytes(range(256)) * 20
    (directory / "video-overview-wall.mp4").write_bytes(data)
    server = make_server(tmp_path / "jobs")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/jobs/{directory.name}/video-overview-wall.mp4"
    try:
        for spec, expected in [
            ("bytes=0-99", data[:100]),
            ("bytes=100-", data[100:]),
            ("bytes=-100", data[-100:]),
        ]:
            with urlopen(Request(url, headers={"Range": spec})) as r:
                assert r.status == 206 and r.read() == expected
                assert r.headers["Accept-Ranges"] == "bytes"
        with urlopen(Request(url, method="HEAD")) as r:
            assert r.read() == b"" and int(r.headers["Content-Length"]) == len(data)
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, headers={"Range": "bytes=999999-"}))
        assert error.value.code == 416
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
