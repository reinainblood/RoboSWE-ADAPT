"""Persistent RoboCasa process. All simulator calls stay on its owner thread."""

import argparse
import base64
import hashlib
import json
import os
import platform
import random
import signal
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .config import normalize_config


def json_bytes(value):
    return json.dumps(value, sort_keys=True, default=lambda x: x.tolist()).encode()


def npz_bytes(observation):
    result = BytesIO()
    np.savez_compressed(result, **observation)
    return result.getvalue()


class RobotSession:
    def __init__(self, config, role="agent", expected_initial=None, factory=None):
        self.config = normalize_config(config)
        self.id = str(uuid4())
        self.role = role
        self.phase = "active"
        self.steps = 0
        self.last_result = {}
        self.recording = None
        self.recordings = {}
        self.recording_root = Path(os.getenv("ROBOT_RECORDINGS_DIR", "recordings"))
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        if factory is None:
            from .environment import RobotEnvironment

            factory = RobotEnvironment
        self.env = factory(
            self.config["task"],
            robot=self.config["robot"],
            split=self.config["split"],
            seed=self.config["seed"],
            **self.config["env_kwargs"],
        )
        try:
            self.observation, self.info = self.env.reset(seed=self.config["seed"])
            self.raw = self.env.unwrapped.env
            self.initial = self.fingerprint()
            if expected_initial is not None and self.initial != expected_initial:
                raise ValueError(
                    f"Initial scene mismatch: expected {expected_initial}, got {self.initial}"
                )
        except BaseException:
            self.env.close()
            raise

    def fingerprint(self):
        """Exact starting XML + physical state, excluding renderer-dependent pixels."""
        digest = hashlib.sha256(self.raw.sim.model.get_xml().encode())
        digest.update(
            np.asarray(self.raw.sim.get_state().flatten(), dtype="<f8").tobytes()
        )
        digest.update(json_bytes(self.config))
        return digest.hexdigest()

    def status(self):
        return {
            "ready": True,
            "session_id": self.id,
            "role": self.role,
            "phase": self.phase,
            "steps": self.steps,
            "config": self.config,
            "initial_fingerprint": self.initial,
            "last_result": self.last_result,
            "api_version": 2,
            "robot": self.config["robot"],
            "action_space": self.env.action_schema(),
            "cameras": list(self.env.camera_keys),
            "recording_api": 1,
            "recording": self.recording.manifest
            if self.recording is not None
            else None,
        }

    def step(self, action):
        if self.phase != "active":
            raise ValueError(
                "Session is held for inspection; start a fresh session to replay"
            )
        try:
            converted = np.asarray(action, dtype=self.env.action_space.dtype)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Expected the selected robot's native numeric action vector"
            ) from exc
        if not np.isfinite(converted).all() or not self.env.action_space.contains(
            converted
        ):
            raise ValueError(
                f"Action must have shape {self.env.action_space.shape}, finite values and satisfy controller bounds"
            )
        try:
            obs, reward, terminated, truncated, info = self.env.step(converted)
        except Exception:
            # A failed step may already have mutated physics. Do not keep driving it.
            self.phase = "faulted"
            self.record_event("physics_error")
            if self.recording is not None:
                self.recording.stop("physics_error")
            raise
        self.observation, self.info = obs, info
        self.steps += 1
        self.last_result = {
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        }
        self.record_event("step")
        if not getattr(self, "_encode_step_observation", True):
            # Trusted in-process benchmark coordinators read self.observation
            # directly and build their own public response. HTTP clients retain
            # the default encoded observation contract.
            return dict(self.last_result)
        return {
            **self.last_result,
            "observation_npz": base64.b64encode(npz_bytes(obs)).decode(),
        }

    def finish(self):
        if self.phase == "active":
            self.phase = "held"
        return self.status()

    def snapshot(self):
        """Inspection artifact, not a promise to restore Python/controller memory."""
        import mujoco

        sim = self.raw.sim
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.empty(mujoco.mj_stateSize(sim.model._model, spec))
        mujoco.mj_getState(sim.model._model, sim.data._data, integration, spec)
        metadata = {
            **self.status(),
            "info": self.info,
            "episode_metadata": self.raw.get_ep_meta(),
            "current_fingerprint": self.fingerprint(),
            "snapshot_kind": "inspection",
            "mujoco_state_spec": int(spec),
            "runtime": {
                "python": platform.python_version(),
                "machine": platform.machine(),
                **{
                    p: version(p)
                    for p in ("mujoco", "numpy", "gymnasium", "robocasa", "robosuite")
                },
            },
        }
        output = BytesIO()
        with ZipFile(output, "w", ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json_bytes(metadata))
            archive.writestr("model.xml", sim.model.get_xml())
            archive.writestr("observations.npz", npz_bytes(self.observation))
            archive.writestr(
                "physics.npz",
                npz_bytes(
                    {
                        "sim_state": sim.get_state().flatten(),
                        "integration_state": integration,
                    }
                ),
            )
        return output.getvalue()

    def start_recording(self, payload):
        from .recording import Recording

        if self.recording is not None and self.recording.active:
            raise ValueError("A recording is already active for this session")
        if set(payload) - {"recording_id", "width", "height"}:
            raise ValueError("Unknown recording settings")
        recording = Recording(self, root=self.recording_root, **payload)
        self.recording = recording
        self.recordings[recording.manifest["recording_id"]] = recording
        recording.persist()
        return recording.manifest

    def stop_recording(self, payload):
        recording = self.recordings[payload["recording_id"]]
        return recording.stop(payload.get("reason", "completed"))

    def recording_file(self, recording_id, filename):
        recording = self.recordings[recording_id]
        if recording.active:
            raise ValueError("Stop the recording before downloading it")
        if filename not in recording.manifest["files"]:
            raise ValueError("Unknown recording artifact")
        return (recording.directory / filename).read_bytes()

    def record_event(self, event):
        if self.recording is None or not self.recording.active:
            return
        try:
            if event == "idle":
                self.recording.idle()
            else:
                self.recording.capture(event)
        except Exception as exc:  # noqa: BLE001 -- A recording failure must not misreport a completed physics step.
            self.recording.manifest["errors"].append(f"{type(exc).__name__}: {exc}")
            self.recording.stop("recording_error")

    def close(self):
        if self.recording is not None:
            self.recording.stop("session_closed")
        self.env.close()


def serve(session, host, port, stopping):
    class Handler(BaseHTTPRequestHandler):
        # Bound slow or abandoned local clients so shutdown remains possible.
        timeout = 30

        def respond(self, status, content_type, payload):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def handle_request(self):
            path = self.path.split("?", 1)[0]
            try:
                if self.command == "GET":
                    if path == "/health":
                        return self.respond(
                            200, "application/json", json_bytes(session.status())
                        )
                    if path == "/observation":
                        return self.respond(
                            200,
                            "application/octet-stream",
                            npz_bytes(session.observation),
                        )
                    if path == "/snapshot":
                        return self.respond(200, "application/zip", session.snapshot())
                    if path.startswith("/recording/"):
                        parts = path.strip("/").split("/")
                        if len(parts) == 3:
                            return self.respond(
                                200,
                                "application/octet-stream",
                                session.recording_file(parts[1], parts[2]),
                            )
                    if path.startswith("/camera/"):
                        from PIL import Image

                        camera = path.removeprefix("/camera/")
                        if camera not in session.status()["cameras"]:
                            return self.send_error(404, "Unknown camera")
                        output = BytesIO()
                        Image.fromarray(session.observation[camera]).save(
                            output, format="PNG"
                        )
                        return self.respond(200, "image/png", output.getvalue())
                elif self.command == "POST":
                    size = int(self.headers.get("Content-Length", 0))
                    if not 0 <= size <= 1024 * 1024:
                        return self.send_error(413)
                    payload = json.loads(self.rfile.read(size) or b"{}")
                    if path == "/step":
                        result = session.step(payload)
                    elif path == "/finish":
                        result = session.finish()
                    elif path == "/recording/start":
                        result = session.start_recording(payload)
                    elif path == "/recording/stop":
                        result = session.stop_recording(payload)
                    else:
                        return self.send_error(404)
                    return self.respond(200, "application/json", json_bytes(result))
                self.send_error(404)
            except (ValueError, TypeError, KeyError) as exc:
                self.respond(
                    409 if session.phase != "active" else 400,
                    "application/json",
                    json_bytes({"error": str(exc)}),
                )
            except Exception as exc:  # noqa: BLE001 -- keep inspection server alive after request failures
                traceback.print_exc()
                self.respond(500, "application/json", json_bytes({"error": str(exc)}))

        do_GET = handle_request
        do_POST = handle_request

    with HTTPServer((host, port), Handler) as server:
        server.timeout = 0.5
        print(json_bytes(session.status()).decode(), flush=True)
        while not stopping[0]:
            server.handle_request()
            session.record_event("idle")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="JSON config file; otherwise ROBOT_SESSION_CONFIG"
    )
    parser.add_argument(
        "--role",
        choices=("agent", "verifier"),
        default=os.getenv("ROBOT_SESSION_ROLE", "agent"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.config:
        with open(args.config) as source:
            config = json.load(source)
    else:
        config = json.loads(os.getenv("ROBOT_SESSION_CONFIG", "{}"))
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    session = RobotSession(
        config, args.role, os.getenv("ROBOT_SESSION_EXPECTED_INITIAL")
    )
    try:
        serve(session, args.host, args.port, stopping)
    finally:
        session.close()


if __name__ == "__main__":
    main()
