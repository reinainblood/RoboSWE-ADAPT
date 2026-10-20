"""Small client for agent code and verifier inspection inside either container."""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen


class SessionClient:
    def __init__(self, url=None, timeout=120):
        self.url = (
            url or os.getenv("ROBOT_SESSION_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.timeout = timeout

    def request(self, path, payload=None):
        data = (
            None
            if payload is None
            else json.dumps(payload, default=lambda x: x.tolist()).encode()
        )
        request = Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def status(self):
        return json.loads(self.request("/health"))

    def start_recording(self, recording_id, width=640, height=480):
        return json.loads(
            self.request(
                "/recording/start",
                {"recording_id": recording_id, "width": width, "height": height},
            )
        )

    def stop_recording(self, recording_id, reason="completed"):
        return json.loads(
            self.request(
                "/recording/stop", {"recording_id": recording_id, "reason": reason}
            )
        )

    @staticmethod
    def decode_observation(data):
        import numpy as np

        with np.load(BytesIO(data), allow_pickle=False) as observation:
            return {k: observation[k].copy() for k in observation.files}

    def observe(self):
        return self.decode_observation(self.request("/observation"))

    def action_schema(self):
        """Native vector bounds and controller input slices for this robot."""
        return self.status()["action_space"]

    def step(self, action):
        result = json.loads(self.request("/step", action))
        obs = self.decode_observation(base64.b64decode(result["observation_npz"]))
        return (
            obs,
            result["reward"],
            result["terminated"],
            result["truncated"],
            result["info"],
        )

    def finish(self):
        """Freeze further actions without closing the environment or container."""
        return json.loads(self.request("/finish", {}))

    def snapshot(self, destination):
        data = self.request("/snapshot")
        Path(destination).write_bytes(data)
        return Path(destination)
