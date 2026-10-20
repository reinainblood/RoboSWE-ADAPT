"""Local, paused-by-default Michael control panel; simulator stays on one thread."""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from .runtime import Camera, describe, make_environment, tactile

UI = Path(__file__).parent / "ui"


class Panel:
    def __init__(self, kitchen=False):
        self.env = make_environment(kitchen=kitchen)
        self.kitchen = kitchen
        self.robot = self.env.robots[0]
        self.action = self.robot.create_action_vector({})
        self.home = self.action.copy()
        for _ in range(25):
            self.env.step(self.action)
        self.camera = Camera(self.env)
        self.queue = []
        self.motion = None
        self.error = None
        self.revision = 0
        self.label = "Paused / ready"
        self.touch_guard = None
        self.touch_return = None

    def state(self):
        return {
            **describe(self.env),
            "target": self.action.tolist(),
            "running": bool(self.motion or self.queue),
            "scene": "kitchen" if self.kitchen else "workshop",
            "label": self.label,
            "error": self.error,
            "revision": self.revision,
            "camera_view": self.camera.view,
        }

    def schedule(self, parts, seconds=1.0, label="Move joints"):
        seconds = float(seconds)
        if not np.isfinite(seconds) or not 0.05 <= seconds <= 5:
            raise ValueError("Duration must be between 0.05 and 5 seconds")
        target = self.action.copy()
        slices = self.robot.composite_controller._action_split_indexes
        target[slice(*slices["base"])] = 0
        for part, values in parts.items():
            if part not in slices:
                raise ValueError(f"Unknown control group: {part}")
            values = np.asarray(values, dtype=float)
            start, end = slices[part]
            if values.shape != (end - start,):
                raise ValueError(f"{part} needs {end - start} values")
            target[start:end] = values
        low, high = self.robot.composite_controller.action_limits
        if (
            not np.isfinite(target).all()
            or np.any(target < low)
            or np.any(target > high)
        ):
            raise ValueError("Command exceeds native joint limits")
        self.queue.append(
            (target, max(1, round(seconds * self.env.control_freq)), label)
        )

    def stop(self):
        self.queue.clear()
        self.motion = None
        self.touch_guard = None
        self.action[54:57] = 0
        # Stop advancing physics immediately. This is a simulation pause.
        self.label = "Paused / physics frozen"
        self.revision += 1

    def tick(self):
        if self.motion is None and self.queue:
            target, count, label = self.queue.pop(0)
            self.motion = [self.action.copy(), target, count, 0]
            self.label = label
        if self.motion is None:
            return
        start, target, count, index = self.motion
        alpha = (index + 1) / count
        blend = alpha * alpha * (3 - 2 * alpha)
        command = start + blend * (target - start)
        try:
            self.env.step(command)
            data = self.env.sim.data._data
            if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
                raise RuntimeError("Physics warning; simulation paused")
            self.action = command
            self.revision += 1
            if self.touch_guard:
                force = tactile(self.robot)[self.touch_guard]["index_finger"][
                    "normal_N"
                ]
                if force > 0.10:
                    side = self.touch_guard
                    self.stop()
                    self.label = (
                        f"{side.title()} index felt {force:.2f} N / paused on contact"
                    )
                    return
            self.motion[3] += 1
            if self.motion[3] == count:
                self.motion = None
                self.action[54:57] = 0
                if not self.queue:
                    # Let the velocity servos brake before freezing the scene.
                    for _ in range(5):
                        self.env.step(self.action)
                    self.label = "Paused / movement complete"
        except Exception as exc:  # noqa: BLE001 - any control failure freezes this simulator
            self.stop()
            self.error = str(exc)

    def preset(self, name, side):
        if side not in ("left", "right"):
            raise ValueError("Choose left or right")
        self.stop()
        if name == "open":
            self.schedule(
                {f"{side}_gripper": np.zeros(20)}, label=f"{side.title()} hand / open"
            )
        elif name == "flex":
            hand = np.tile([0.3, 0.08, 0.25, 0.15], 5)
            hand[:4] = [0.15, -0.15, 0.2, 0.15]
            self.schedule(
                {f"{side}_gripper": hand}, label=f"{side.title()} hand / gentle flex"
            )
        elif name == "wrist":
            start, end = self.robot.composite_controller._action_split_indexes[side]
            target = self.home[start:end].copy()
            target[6] += 0.3 if side == "right" else -0.3
            self.schedule({side: target}, label=f"{side.title()} wrist / turn")
        elif name == "home":
            self.schedule(
                {
                    part: self.home[s:e]
                    for part, (
                        s,
                        e,
                    ) in self.robot.composite_controller._action_split_indexes.items()
                },
                seconds=2,
                label="Return to commissioning pose",
            )
        elif name == "hold":
            self.schedule({}, seconds=1, label="Hold pose / advance one second")
        else:
            raise ValueError("Unknown movement")

    def probe(self, side):
        if self.kitchen:
            raise ValueError("The calibration probe is in the commissioning room")
        if side not in ("right", "left", "remove"):
            raise ValueError("Choose a hand or remove the probe")
        self.stop()
        model, data = self.env.sim.model._model, self.env.sim.data._data
        mocap = model.body("tactile_probe").mocapid[0]
        if side == "remove":
            data.mocap_pos[mocap] = [2, 0, 1]
            self.env.sim.forward()
            self.schedule(
                self.touch_return or {},
                seconds=1,
                label="Probe removed / finger withdraws",
            )
            self.touch_return = None
            return
        gripper = self.robot.gripper[side]
        site = model.site(gripper.correct_naming("index_finger_tactile_site")).id
        outward = 1 if side == "right" else -1
        data.mocap_pos[mocap] = data.site_xpos[site] + data.site_xmat[site].reshape(
            3, 3
        ) @ np.array([0, outward * (model.site_size[site, 1] + 0.008), 0])
        start, end = self.robot.composite_controller._action_split_indexes[
            f"{side}_gripper"
        ]
        hand = self.action[start:end].copy()
        self.touch_return = {f"{side}_gripper": hand.copy()}
        hand[4] += 0.12
        hand[6] += 0.08
        self.schedule(
            {f"{side}_gripper": hand},
            seconds=2,
            label=f"{side.title()} index / close until touch",
        )
        self.touch_guard = side
        self.camera.set_view(side)

    def close(self):
        self.camera.close()
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8873)
    parser.add_argument("--kitchen", action="store_true")
    args = parser.parse_args()
    panel = Panel(kitchen=args.kitchen)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, value, content_type="application/json", status=200):
            payload = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/state":
                self.send(panel.state())
            elif path == "/api/frame":
                output = BytesIO()
                Image.fromarray(panel.camera.render()).save(output, "JPEG", quality=88)
                self.send(output.getvalue(), "image/jpeg")
            elif path in ("/", "/panel.js", "/panel.css"):
                file = (
                    UI
                    / {
                        "/": "index.html",
                        "/panel.js": "panel.js",
                        "/panel.css": "panel.css",
                    }[path]
                )
                mime = {
                    ".html": "text/html",
                    ".js": "text/javascript",
                    ".css": "text/css",
                }[file.suffix]
                self.send(file.read_bytes(), mime)
            else:
                self.send({"error": "Not found"}, status=404)

        def do_POST(self):
            nonlocal panel
            try:
                origin = self.headers.get("Origin")
                if origin and origin not in (
                    f"http://127.0.0.1:{args.port}",
                    f"http://localhost:{args.port}",
                ):
                    raise ValueError("Cross-origin commands are not accepted")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid command size")
                body = json.loads(self.rfile.read(length))
                if self.path == "/api/move":
                    panel.stop()
                    panel.schedule(body["parts"], body.get("seconds", 1))
                elif self.path == "/api/preset":
                    panel.preset(body["name"], body.get("side", "right"))
                elif self.path == "/api/probe":
                    panel.probe(body["side"])
                elif self.path == "/api/stop":
                    panel.stop()
                elif self.path == "/api/camera":
                    if "view" in body:
                        panel.camera.set_view(body["view"])
                    panel.camera.camera.azimuth += float(body.get("orbit", 0))
                    panel.camera.camera.distance = float(
                        np.clip(
                            panel.camera.camera.distance * float(body.get("zoom", 1)),
                            0.2,
                            5,
                        )
                    )
                    panel.revision += 1
                elif self.path == "/api/scene":
                    if body.get("scene") not in ("workshop", "kitchen"):
                        raise ValueError("Unknown scene")
                    panel.stop()
                    next_panel = Panel(kitchen=body["scene"] == "kitchen")
                    panel.close()
                    panel = next_panel
                else:
                    raise ValueError("Unknown command")
                self.send(panel.state())
            except (ValueError, TypeError, KeyError) as exc:
                self.send({"error": str(exc)}, status=400)
            except Exception as exc:  # noqa: BLE001 - return diagnostics and freeze on server failures
                panel.stop()
                panel.error = str(exc)
                self.send({"error": str(exc)}, status=500)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.timeout = 0.02
    print(f"MICHAEL_PANEL_READY http://127.0.0.1:{args.port}", flush=True)
    deadline = time.monotonic()
    try:
        while True:
            server.handle_request()
            if time.monotonic() >= deadline:
                panel.tick()
                deadline = time.monotonic() + 1 / panel.env.control_freq
    finally:
        server.server_close()
        panel.close()


if __name__ == "__main__":
    main()
