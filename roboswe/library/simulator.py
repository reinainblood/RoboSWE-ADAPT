"""Fresh-process simulator interface; production uses an immutable Docker image."""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from roboswe.execution.sandbox import docker_path


class Digest:
    def __init__(self):
        import hashlib

        self.value = hashlib.sha256().hexdigest()

    def hexdigest(self):
        return self.value


class SimulatorProcess:
    def __init__(
        self,
        config,
        spec,
        artifacts,
        *,
        image=None,
        local_development=False,
        role="agent",
        timeout=720,
        max_steps=6000,
        lifetime_seconds=14400,
        cpu=4,
        memory_mb=16384,
    ):
        self.proc = None
        self.container = None
        self.events = queue.Queue(maxsize=2)
        self.action_digest = Digest()
        self.timeout = timeout
        self.started = None
        self.log = None
        self.sequence = 0
        directory = Path(artifacts).resolve()
        directory.parent.mkdir(parents=True, exist_ok=True)
        self.diagnostics_path = directory.parent / (directory.name + ".status.jsonl")
        if directory.exists():
            raise FileExistsError(directory)
        environment = {**os.environ, "PYTHONHASHSEED": "0", "PYNPUT_BACKEND": "dummy"}
        if local_development:
            if image is not None:
                raise ValueError("Choose local development OR a container image")
            args = [sys.executable, "-m", "robot_session.benchmark_worker"]
            artifact_path = str(directory)
        else:
            if not image or not image.startswith("sha256:") or len(image) != 71:
                raise ValueError("Simulator requires an immutable Docker image ID")
            self.docker = docker_path()
            self.container = "roboswe-simulator-" + uuid4().hex
            # This output mount is owned by the trusted simulator and never
            # mounted in the code sandbox. Assets/code remain image-contained.
            args = [
                self.docker,
                "run",
                "-i",
                "--log-driver",
                "none",
                "--name",
                self.container,
                "--network",
                "none",
                "--label",
                "roboswe=simulator",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--memory",
                f"{memory_mb}m",
                "--cpus",
                str(cpu),
                "--mount",
                f"type=bind,src={directory.parent},dst=/output",
                "--entrypoint",
                "python",
                image,
                "-m",
                "robot_session.benchmark_worker",
            ]
            artifact_path = "/output/" + directory.name
        self.log = (directory.parent / (directory.name + ".log")).open("xb")
        try:
            self.proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
                env=environment,
            )
            proc, events = self.proc, self.events

            def reader():
                for line in proc.stdout:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        event = {"error": "Non-JSON simulator protocol response"}
                    try:
                        events.put(event, timeout=5)
                    except queue.Full:
                        proc.terminate()
                        return
                try:
                    events.put({"error": "Simulator process exited"}, timeout=5)
                except queue.Full:
                    pass

            threading.Thread(target=reader, daemon=True).start()
            self._metadata = self._call(
                "initialize",
                wait=600,
                config=config,
                spec=spec,
                artifacts=artifact_path,
                role=role,
                timeout=timeout,
                max_steps=max_steps,
                lifetime_seconds=lifetime_seconds,
            )
            self._metadata["simulator_image"] = image
            self._metadata["development_runtime"] = local_development
            if spec.get("sensor_contract_version", 0) > self._metadata.get(
                "sensor_contract_version", 0
            ):
                raise ValueError(
                    "Simulator image does not implement this protocol's sensor contract; rebuild it"
                )
            if (
                spec.get("requires_boundary_fingerprints")
                and self._metadata.get("benchmark_api_version", 0) < 2
            ):
                raise ValueError(
                    "This library protocol needs a newly built simulator image with boundary fingerprints"
                )
            expected = spec.get("fixture_bindings")
            if (
                expected is not None
                and {k: v["name"] for k, v in self._metadata["bindings"].items()}
                != expected
            ):
                raise ValueError(
                    "Simulator fixture binding does not match the versioned library set"
                )
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _call(self, operation, *, wait=180, **kwargs):
        self.sequence += 1
        request_id = self.sequence
        try:
            self.proc.stdin.write(
                json.dumps(
                    {"id": request_id, "operation": operation, **kwargs},
                    allow_nan=False,
                )
                + "\n"
            )
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.diagnose("pipe_closed")
            raise RuntimeError(
                f"Simulator pipe closed during {operation}; see {self.diagnostics_path}"
            ) from exc
        deadline = time.monotonic() + wait
        try:
            while True:
                response = self.events.get(
                    timeout=max(0.001, deadline - time.monotonic())
                )
                # New workers correlate responses. Old immutable images remain
                # readable; a late correlated response can never satisfy a later call.
                if response.get("id") in (None, request_id):
                    break
                if time.monotonic() >= deadline:
                    raise queue.Empty
        except queue.Empty as exc:
            self.diagnose("response_timeout")
            raise TimeoutError("Simulator response deadline exceeded") from exc
        if "error" in response:
            raise RuntimeError(response["error"])
        self.action_digest.value = response.get(
            "action_sha256", self.action_digest.value
        )
        return response["result"]

    def diagnose(self, reason):
        state = {
            "reason": reason,
            "process_returncode": self.proc.poll() if self.proc else None,
        }
        if self.container:
            try:
                result = subprocess.run(
                    [
                        self.docker,
                        "inspect",
                        "--format",
                        "{{json .State}}",
                        self.container,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if result.returncode == 0:
                    actual = json.loads(result.stdout)
                    state["container"] = {
                        k: actual.get(k)
                        for k in (
                            "Status",
                            "Running",
                            "ExitCode",
                            "OOMKilled",
                            "Error",
                            "StartedAt",
                            "FinishedAt",
                        )
                    }
                else:
                    state["container_status_unavailable"] = True
                    state["docker_error"] = result.stderr.strip()[:2000]
            except (OSError, ValueError, subprocess.TimeoutExpired):
                state["container_status_unavailable"] = True
        with self.diagnostics_path.open("a") as output:
            output.write(json.dumps(state) + "\n")
        return state

    def metadata(self):
        return self._metadata

    def start_task(self, index):
        result = self._call("start_task", index=index)
        self.started = time.monotonic()
        return result

    def bridge(self, message):
        return self._call("bridge", message=message)

    def observe(self):
        return self._call("observe")

    def fingerprint(self):
        return self._call("fingerprint")

    def finish_task(self, *, code_errors=0):
        return self._call("finish_task", code_errors=code_errors)

    def record_start(self, recording_id, width=640, height=480):
        return self._call(
            "record_start",
            settings={"recording_id": recording_id, "width": width, "height": height},
        )

    def record_stop(self, recording_id):
        return self._call("record_stop", settings={"recording_id": recording_id})

    def close(self):
        if self.proc:
            try:
                if self.proc.poll() is None:
                    self._call("close", wait=15)
            except (OSError, RuntimeError, TimeoutError):
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
            self.diagnose("shutdown")
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    stream.close()
                except BrokenPipeError:
                    pass
            self.proc = None
        if self.container:
            subprocess.run(
                [self.docker, "rm", "-f", self.container],
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.container = None
        if self.log:
            self.log.close()
