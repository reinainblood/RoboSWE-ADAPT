"""Explicit container lifecycle for Docker and Modal. No scoring or agent harness."""

import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .config import normalize_config

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "roboswe-365-sessions"
ENVIRONMENT = "FDR"
ENTRYPOINT = ["python", "-m", "robot_session.session"]


@dataclass
class SessionHandle:
    backend: str
    container_id: str
    image: str
    config: dict
    role: str
    initial_fingerprint: str = ""
    environment: str = ""
    app_id: str = ""
    lifetime_seconds: int = 86400
    cpu: float = 4
    memory_mb: int = 16384
    checkpointed: bool = False

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(asdict(self), indent=2) + "\n")
        temp.replace(path)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def sandbox(self):
        import modal

        if self.environment != ENVIRONMENT:
            raise ValueError("Only this project's FDR sessions are supported")
        return modal.Sandbox.from_id(self.container_id)

    def execute(self, *command, timeout=120, stdin=None):
        """Returning or failing a command never stops the session container."""
        if self.backend == "docker":
            return subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    self.container_id,
                    "timeout",
                    "--kill-after=5s",
                    str(timeout),
                    *command,
                ],
                input=stdin,
                capture_output=True,
                timeout=timeout + 30,
                check=False,
            )
        sandbox = self.sandbox()
        process = sandbox.exec(*command, timeout=timeout, text=False)
        if stdin is not None:
            process.stdin.write(stdin)
        process.stdin.write_eof()
        process.stdin.drain()
        with ThreadPoolExecutor(2) as pool:
            stdout = pool.submit(process.stdout.read)
            stderr = pool.submit(process.stderr.read)
            process.wait()
            return subprocess.CompletedProcess(
                command, process.returncode, stdout.result(), stderr.result()
            )

    def request(self, path, payload=None):
        script = (
            "import sys,json; from robot_session.client import SessionClient; "
            "sys.stdout.buffer.write(SessionClient().request(sys.argv[1], json.loads(sys.argv[2])))"
        )
        result = self.execute(
            "python", "-c", script, path, json.dumps(payload), timeout=180
        )
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return result.stdout

    def status(self):
        return json.loads(self.request("/health"))

    def finish(self):
        return json.loads(self.request("/finish", {}))

    def snapshot(self, path):
        data = self.request("/snapshot")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        # Container identity stays outside the simulation artifact, under host control.
        self.save(str(path) + ".session.json")
        return path

    def run_code(self, source, timeout=600):
        """Run exact submitted Python bytes in /workspace; simulation is a separate process."""
        source = Path(source).read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        target = f"/workspace/submission-{digest}.py"
        upload = self.execute(
            "python",
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())",
            target,
            stdin=source,
        )
        if upload.returncode:
            raise RuntimeError(upload.stderr.decode(errors="replace"))
        return self.execute("python", target, timeout=timeout)

    def stop(self):
        """Explicit teardown. Export snapshots before calling this."""
        if self.backend == "docker":
            subprocess.run(
                ["docker", "stop", "--time", "30", self.container_id],
                check=True,
                capture_output=True,
            )
            # Stop a resumed process, but retain the filesystem and CRIU data.
            if self.checkpointed:
                return
            subprocess.run(
                ["docker", "rm", self.container_id], check=True, capture_output=True
            )
        else:
            from modal.exception import SandboxTerminatedError

            sandbox = self.sandbox()
            sandbox.terminate()
            try:
                sandbox.wait()
            except SandboxTerminatedError:
                pass

    def checkpoint(self, path):
        """Suspend the full container; return a durable resume receipt.

        Run after submitted code has returned. Modal currently terminates the
        original sandbox on snapshot; Docker stops and retains its container.
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        status = self.status()
        if (status.get("recording") or {}).get("status") == "recording":
            raise ValueError("Stop video recording before checkpointing the session")
        record = {
            "kind": "container-memory",
            "state": "pending",
            "created_at": datetime.now(UTC).isoformat(),
            "session": asdict(self),
            "status": status,
        }
        if self.backend == "docker":
            info = json.loads(
                subprocess.check_output(
                    ["docker", "info", "--format", "{{json .}}"], text=True
                )
            )
            if not info.get("ExperimentalBuild") or info.get("OSType") != "linux":
                raise RuntimeError(
                    "Docker checkpoint requires a Linux daemon with experimental features and CRIU installed on the daemon host"
                )
            record["checkpoint_id"] = "roboswe-" + uuid4().hex
            record["docker_daemon_id"] = info["ID"]
            path.write_text(json.dumps(record, indent=2) + "\n")
            subprocess.run(
                [
                    "docker",
                    "checkpoint",
                    "create",
                    self.container_id,
                    record["checkpoint_id"],
                ],
                check=True,
            )
            self.checkpointed = True
            record["session"] = asdict(self)
            record["retention"] = (
                "Keep this stopped container and daemon storage; do not docker rm/prune it"
            )
        else:
            path.write_text(json.dumps(record, indent=2) + "\n")
            snapshot = self.sandbox()._experimental_snapshot()
            record["checkpoint_id"] = snapshot.object_id
            record["retention"] = (
                "Modal memory snapshots expire after 7 days; snapshot chains inherit the first expiry"
            )
        record["state"] = "ready"
        path.write_text(json.dumps(record, indent=2) + "\n")
        return record


def build_image(backend):
    if backend == "docker":
        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(ROOT / "robot_session/Dockerfile"),
                "-t",
                "roboswe-session:local",
                str(ROOT),
            ],
            check=True,
        )
        return subprocess.check_output(
            [
                "docker",
                "image",
                "inspect",
                "roboswe-session:local",
                "--format",
                "{{.Id}}",
            ],
            text=True,
        ).strip()
    import modal

    app = modal.App.lookup(
        APP_NAME, environment_name=ENVIRONMENT, create_if_missing=True
    )
    with modal.enable_output():
        image = modal.Image.from_dockerfile(
            ROOT / "robot_session/Dockerfile",
            context_dir=ROOT,
            ignore=modal.FilePatternMatcher.from_file(
                ROOT / "robot_session/Dockerfile.dockerignore"
            ),
        ).build(app)
    return image.object_id


def start(
    backend,
    image,
    config,
    *,
    role="agent",
    expected_initial=None,
    receipt=None,
    lifetime_seconds=86400,
    cpu=4,
    memory_mb=16384,
    startup_timeout=600,
):
    config = normalize_config(config)
    if backend not in ("docker", "modal"):
        raise ValueError("backend must be docker or modal")
    if not 1 <= lifetime_seconds <= 86400:
        raise ValueError(
            "Set a lifetime between 1 and 86400 seconds, including inspection time"
        )
    env = {"ROBOT_SESSION_CONFIG": json.dumps(config), "ROBOT_SESSION_ROLE": role}
    if expected_initial:
        env["ROBOT_SESSION_EXPECTED_INITIAL"] = expected_initial
    if backend == "docker":
        # Resolve mutable tags once. Replay always uses this immutable local image ID.
        image = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
        ).strip()
        command = [
            "docker",
            "run",
            "-d",
            "--init",
            "--cpus",
            str(cpu),
            "--memory",
            f"{memory_mb}m",
            "--label",
            "roboswe.component=session",
        ]
        for key, value in env.items():
            command.extend(["-e", f"{key}={value}"])
        container_id = subprocess.check_output(
            [*command, image, *ENTRYPOINT], text=True
        ).strip()
        handle = SessionHandle(
            backend,
            container_id,
            image,
            config,
            role,
            cpu=cpu,
            memory_mb=memory_mb,
            lifetime_seconds=lifetime_seconds,
        )
    else:
        import modal

        app = modal.App.lookup(
            APP_NAME, environment_name=ENVIRONMENT, create_if_missing=True
        )
        sandbox = modal.Sandbox.create(
            *ENTRYPOINT,
            image=modal.Image.from_id(image),
            app=app,
            environment_name=ENVIRONMENT,
            env=env,
            cpu=cpu,
            memory=memory_mb,
            timeout=lifetime_seconds,
            workdir="/workspace",
            _experimental_enable_snapshot=True,
            tags={"component": "roboswe-session", "role": role},
        )
        handle = SessionHandle(
            backend,
            sandbox.object_id,
            image,
            config,
            role,
            environment=ENVIRONMENT,
            app_id=app.app_id,
            cpu=cpu,
            memory_mb=memory_mb,
            lifetime_seconds=lifetime_seconds,
        )
    try:
        if receipt:
            handle.save(receipt)
        deadline = time.monotonic() + startup_timeout
        last_error = ""
        while time.monotonic() < deadline:
            try:
                status = handle.status()
                if (
                    expected_initial
                    and status["initial_fingerprint"] != expected_initial
                ):
                    raise ValueError("Initial scene mismatch")
                handle.initial_fingerprint = status["initial_fingerprint"]
                handle.config = status["config"]
                if receipt:
                    handle.save(receipt)
                return handle
            except RuntimeError as exc:
                last_error = str(exc)
                if backend == "modal" and handle.sandbox().poll() is not None:
                    raise RuntimeError(
                        "Session exited during startup: "
                        + handle.sandbox().stderr.read()
                    ) from exc
                if backend == "docker":
                    running = subprocess.check_output(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{.State.Running}}",
                            handle.container_id,
                        ],
                        text=True,
                    ).strip()
                    if running != "true":
                        logs = subprocess.run(
                            ["docker", "logs", handle.container_id],
                            capture_output=True,
                            check=False,
                        )
                        raise RuntimeError(
                            logs.stdout.decode() + logs.stderr.decode()
                        ) from exc
                time.sleep(2)
        raise TimeoutError(
            f"Session was not ready after {startup_timeout}s: {last_error}"
        )
    except BaseException:
        handle.stop()
        raise


def replay(original, *, receipt=None, startup_timeout=600):
    """Fresh writable filesystem + process + reset, never the agent's final container."""
    if not original.initial_fingerprint:
        raise ValueError("Original receipt lacks a verified initial fingerprint")
    return start(
        original.backend,
        original.image,
        original.config,
        role="verifier",
        expected_initial=original.initial_fingerprint,
        receipt=receipt,
        lifetime_seconds=original.lifetime_seconds,
        cpu=original.cpu,
        memory_mb=original.memory_mb,
        startup_timeout=startup_timeout,
    )


def resume(checkpoint, *, receipt):
    """Resume memory/filesystem state; this is deliberately separate from replay."""
    record = json.loads(Path(checkpoint).read_text())
    if record.get("kind") != "container-memory" or record.get("state") != "ready":
        raise ValueError("Not a completed container-memory checkpoint")
    handle = SessionHandle(**record["session"])
    if handle.backend == "docker":
        info = json.loads(
            subprocess.check_output(
                ["docker", "info", "--format", "{{json .}}"], text=True
            )
        )
        if info["ID"] != record["docker_daemon_id"]:
            raise ValueError(
                "Docker checkpoints must be resumed on their original daemon"
            )
        subprocess.run(
            [
                "docker",
                "start",
                "--checkpoint",
                record["checkpoint_id"],
                handle.container_id,
            ],
            check=True,
        )
    else:
        import modal

        if handle.environment != ENVIRONMENT:
            raise ValueError("Only FDR checkpoints are supported")
        snapshot = modal.SandboxSnapshot.from_id(record["checkpoint_id"])
        sandbox = modal.Sandbox._experimental_from_snapshot(snapshot)
        handle.container_id = sandbox.object_id
    # Save identity before probing so a restore failure cannot hide a live resource.
    handle.save(receipt)
    deadline = time.monotonic() + 180
    while True:
        try:
            status = handle.status()
            break
        except RuntimeError:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Restore did not become ready; resource retained in {receipt}"
                )
            time.sleep(2)
    for field in ("session_id", "initial_fingerprint", "steps", "phase"):
        if status[field] != record["status"][field]:
            raise RuntimeError(
                f"Restored {field} differs; resource retained in {receipt}"
            )
    if handle.backend == "docker":
        # CRIU restores process memory into the retained filesystem. Once that
        # filesystem is writable again, the old checkpoint is not a safe rewind.
        record["state"] = "consumed"
        record["resumed_receipt"] = str(receipt)
        Path(checkpoint).write_text(json.dumps(record, indent=2) + "\n")
    return handle
