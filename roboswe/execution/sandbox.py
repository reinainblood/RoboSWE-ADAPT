"""Docker isolation with a private persistent workspace and bounded pipe RPC.

Only immutable image IDs are executed. No host bind mounts, network, credentials,
simulator code, or verifier state enter this container. This is defense in depth,
not a claim that Docker can withstand every hostile kernel exploit.
"""

import json
import math
import os
import select
import selectors
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
from uuid import uuid4

FILE_LIMIT = 256 * 1024
LINE_LIMIT = 1024 * 1024
LOG_LIMIT = 128 * 1024


def validate_path(path, *, operation=None):
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        raise ValueError("Use a relative .py path")
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or (parsed.suffix != ".py" and path != "NOTES.md")
        or len(path) > 240
    ):
        raise ValueError("Use a relative .py path or NOTES.md without '..'")
    if operation == "execute" and parsed.suffix != ".py":
        raise ValueError("Only Python files can be executed; NOTES.md is read/write only")
    return path


def docker_path():
    executable = shutil.which("docker")
    if not executable:
        candidate = Path.home() / ".docker/bin/docker"
        if candidate.exists():
            executable = str(candidate)
    if not executable:
        raise RuntimeError("Docker is required for isolated benchmark execution")
    return executable


def build_image(tag="roboswe-code:local"):
    docker = docker_path()
    subprocess.run([docker, "build", "-t", tag, str(Path(__file__).parent)], check=True)
    return subprocess.check_output(
        [docker, "image", "inspect", "--format", "{{.Id}}", tag], text=True
    ).strip()


class CodeSandbox:
    def __init__(self, image, *, memory_mb=1024, cpu=1, lifetime_seconds=14400):
        self.docker = docker_path()
        if not image.startswith("sha256:") or len(image) != 71:
            raise ValueError("Execution requires an immutable Docker image ID")
        self.image = image
        self.memory_mb = memory_mb
        self.cpu = cpu
        self.volume = "roboswe-library-" + uuid4().hex
        self.keeper = self.volume + "-keeper"
        self.active = None
        self.closed = False
        subprocess.run(
            [
                self.docker,
                "volume",
                "create",
                "--label",
                "roboswe=agent-workspace",
                "--driver",
                "local",
                "--opt",
                "type=tmpfs",
                "--opt",
                "device=tmpfs",
                "--opt",
                "o=size=256m,uid=1000,gid=1000",
                self.volume,
            ],
            check=True,
            capture_output=True,
        )
        try:
            # A live mount preserves this bounded tmpfs between attempt containers.
            subprocess.run(
                [
                    self.docker,
                    "run",
                    "-d",
                    "--log-driver",
                    "none",
                    "--name",
                    self.keeper,
                    "--label",
                    "roboswe=workspace-keeper",
                    "--network",
                    "none",
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--pids-limit",
                    "8",
                    "--memory",
                    "512m",
                    "--memory-swap",
                    "512m",
                    "--cpus",
                    ".1",
                    "--mount",
                    f"type=volume,src={self.volume},dst=/workspace",
                    "--entrypoint",
                    "sleep",
                    self.image,
                    str(lifetime_seconds + 300),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def __enter__(self):
        return self

    def __exit__(self, _type, error, _traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note(f"Sandbox cleanup also failed: {cleanup_error}")

    def _remove_attempt(self, name):
        try:
            cleanup = subprocess.run(
                [self.docker, "rm", "-f", name],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"container_cleanup_failed: {type(exc).__name__}: {exc}"
        if cleanup.returncode and b"No such container" not in cleanup.stderr:
            detail = cleanup.stderr.decode(errors="replace").strip()[:2000]
            return f"container_cleanup_failed: {detail}"
        return None

    def close(self):
        if self.closed:
            return
        if self.active:
            subprocess.run(
                [self.docker, "rm", "-f", self.active],
                capture_output=True,
                timeout=30,
                check=False,
            )
        subprocess.run(
            [self.docker, "rm", "-f", self.keeper],
            capture_output=True,
            timeout=30,
            check=False,
        )
        subprocess.run(
            [self.docker, "volume", "rm", self.volume],
            capture_output=True,
            timeout=30,
            check=True,
        )
        self.closed = True

    def archive(self, destination):
        """Save Python sources and NOTES.md; never execute them on the host."""
        from zipfile import ZipFile

        if self.closed or self.active:
            raise RuntimeError("Cannot archive an active or closed sandbox")
        destination = Path(destination)
        name = "roboswe-source-export-" + uuid4().hex
        args = [
            self.docker,
            "run",
            "--log-driver",
            "none",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "8",
            "--memory",
            "256m",
            "--mount",
            f"type=volume,src={self.volume},dst=/workspace,readonly",
            "--entrypoint",
            "/usr/bin/timeout",
            self.image,
            "--signal=KILL",
            "35",
            "python",
            "-I",
            "-B",
            "/opt/roboswe/archive.py",
        ]
        try:
            with destination.open("xb") as output:
                subprocess.run(
                    args, stdout=output, stderr=subprocess.PIPE, timeout=40, check=True
                )
            with ZipFile(destination) as archive:
                return json.loads(archive.read("library-manifest.json"))
        finally:
            subprocess.run(
                [self.docker, "rm", "-f", name],
                capture_output=True,
                timeout=30,
                check=False,
            )

    def checkpoint(self, destination):
        if self.closed or self.active:
            raise RuntimeError('Cannot checkpoint active or closed workspace')
        return checkpoint_volume(self.image, self.volume, destination)

    def restore_checkpoint(self, source):
        if self.closed or self.active:
            raise RuntimeError('Cannot restore active or closed workspace')
        return _checkpoint_maintenance(self.image, self.volume, source=source)

    def request(self, operation, path, *, content="", bridge=None, timeout=60):
        if self.closed or self.active:
            raise RuntimeError("Sandbox is closed or an attempt is active")
        if operation not in {"read", "write", "execute"}:
            raise ValueError("Unknown sandbox operation")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Execution timeout must be positive and finite")
        validate_path(path, operation=operation)
        if len(content.encode()) > FILE_LIMIT:
            raise ValueError("File exceeds 256 KiB")
        name = "roboswe-attempt-" + uuid4().hex
        args = [
            self.docker,
            "run",
            "-i",
            "--log-driver",
            "none",
            "--name",
            name,
            "--label",
            "roboswe=code-attempt",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            f"{self.memory_mb}m",
            "--memory-swap",
            f"{self.memory_mb}m",
            "--cpus",
            str(self.cpu),
            "--ulimit",
            "fsize=67108864:67108864",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=128m",
            "--mount",
            f"type=volume,src={self.volume},dst=/workspace",
            "--entrypoint",
            "/usr/bin/timeout",
            self.image,
            "--signal=KILL",
            str(math.ceil(timeout + 30)),
            "python",
            "-I",
            "-B",
            "-u",
            "/opt/roboswe/runner.py",
        ]
        started = time.monotonic()
        deadline = started + timeout
        self.active = name
        proc = None
        output = bytearray()
        log = bytearray()
        result = None
        error = None
        infrastructure_error = None
        rpc_count = 0

        def send(data):
            payload = memoryview(json.dumps(data, allow_nan=False).encode() + b"\n")
            while payload:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, timeout)
                if not select.select([], [proc.stdin], [], min(0.1, remaining))[1]:
                    continue
                try:
                    sent = os.write(proc.stdin.fileno(), payload[:65536])
                except BlockingIOError:
                    continue
                payload = payload[sent:]

        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            os.set_blocking(proc.stdin.fileno(), False)
            send({"operation": operation, "path": path, "content": content})
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ, "rpc")
            selector.register(proc.stderr, selectors.EVENT_READ, "log")
            with selector:
                while selector.get_map():
                    if time.monotonic() >= deadline:
                        error = "execution_timeout"
                        break
                    for key, _ in selector.select(
                        min(0.1, max(0, deadline - time.monotonic()))
                    ):
                        chunk = key.fileobj.read(65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        if key.data == "log":
                            log.extend(chunk[: max(0, LOG_LIMIT - len(log))])
                            continue
                        output.extend(chunk)
                        if len(output) > LINE_LIMIT:
                            raise ValueError("Agent RPC exceeds maximum line length")
                        while b"\n" in output:
                            line, _, rest = output.partition(b"\n")
                            output = bytearray(rest)
                            message = json.loads(line)
                            if operation != "execute":
                                if result is not None or set(message) != {"result"}:
                                    raise ValueError("Invalid file-operation response")
                                result = message["result"]
                                continue
                            rpc_count += 1
                            if rpc_count > 20000:
                                raise ValueError("Attempt exceeds RPC budget")
                            if not isinstance(message, dict) or message.get(
                                "operation"
                            ) not in {"observe", "step"}:
                                raise ValueError(
                                    "Only observe and step are available to agent code"
                                )
                            expected = (
                                {"operation", "action"}
                                if message["operation"] == "step"
                                else {"operation"}
                            )
                            if set(message) != expected or bridge is None:
                                raise ValueError("Invalid robot request")
                            # The bridge owns validation, budgets and trusted evidence.
                            # Invalid calls stop the attempt, rather than returning host details.
                            response = bridge(message)
                            send({"result": response})
                if output and not error:
                    raise ValueError("Truncated agent protocol response")
            if not error:
                proc.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            error = "execution_timeout"
        except (ValueError, KeyError, RuntimeError, BrokenPipeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            # Removes the entire PID namespace, including detached children.
            infrastructure_error = self._remove_attempt(name)
            if infrastructure_error and error is None:
                error = "container_cleanup_failed"
            if proc:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    stream.close()
            # A failed removal is an isolation failure. Keep the container identity
            # for cleanup, and prohibit another request on this workspace.
            self.active = name if infrastructure_error else None
        return {
            "returncode": proc.returncode if proc else -1,
            "error": error,
            "output": log.decode(errors="replace"),
            "output_truncated": len(log) >= LOG_LIMIT,
            "result": result,
            "rpc_count": rpc_count,
            "elapsed_seconds": time.monotonic() - started,
            **(
                {"infrastructure_error": infrastructure_error}
                if infrastructure_error
                else {}
            ),
        }


# Freeze trusted maintenance code when this coordinator process imports the module.
# It is passed through -I -B -c, never loaded from the authored workspace/image.
_CHECKPOINT_HELPER = Path(__file__).with_name('workspace_checkpoint.py').read_text()


def _checkpoint_maintenance(image, volume, *, destination=None, source=None):
    import hashlib
    from .workspace_checkpoint import inspect_archive
    if not isinstance(image,str) or not __import__('re').fullmatch(r'sha256:[0-9a-f]{64}',image):
        raise ValueError('Immutable code image required')
    if not isinstance(volume,str) or not __import__('re').fullmatch(r'roboswe-library-[0-9a-f]{32}',volume):
        raise ValueError('Expected owned RoboSWE workspace volume')
    exporting=destination is not None
    if exporting == (source is not None):raise ValueError('Choose export or restore')
    if source is not None:inspect_archive(source)
    docker=docker_path()
    existing=subprocess.run([docker,'volume','inspect','--format','{{.Name}}',volume],
                            capture_output=True,text=True,timeout=15,check=True)
    if existing.stdout.strip()!=volume:
        raise ValueError('Owned workspace volume does not exist')
    name='roboswe-checkpoint-'+uuid4().hex
    command=[docker,'run','-i','--log-driver','none','--name',name,'--network','none',
             '--read-only','--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges',
             '--pids-limit','8','--memory','512m','--memory-swap','512m','--cpus','1',
             '--tmpfs','/tmp:rw,noexec,nosuid,size=300m',
             '--mount',f'type=volume,src={volume},dst=/workspace'+(',readonly' if exporting else ''),
             '--entrypoint','/usr/bin/timeout',image,'--signal=KILL','150',
             'python','-I','-B','-c',_CHECKPOINT_HELPER,'export' if exporting else 'restore']
    try:
        if exporting:
            with Path(destination).open('xb') as output:
                subprocess.run(command,stdout=output,stderr=subprocess.PIPE,timeout=160,check=True)
            manifest=inspect_archive(destination)
        else:
            with Path(source).open('rb') as input_file:
                result=subprocess.run(command,stdin=input_file,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=160,check=True)
            manifest=json.loads(result.stdout)
        return {**manifest,'maintenance_helper_sha256':hashlib.sha256(_CHECKPOINT_HELPER.encode()).hexdigest()}
    finally:
        result=subprocess.run([docker,'rm','-f',name],capture_output=True,timeout=30,check=False)
        if result.returncode and b'No such container' not in result.stderr:
            raise RuntimeError('Checkpoint maintenance cleanup unconfirmed')


def checkpoint_volume(image, volume, destination):
    """Read-only snapshot of an owned volume; caller must ensure writer quiescence."""
    return _checkpoint_maintenance(image,volume,destination=destination)
