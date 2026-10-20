"""Owner-thread recording of actual control steps, with two explicit clocks."""

import json
import re
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,100}$")


class MP4Writer:
    def __init__(self, path, width, height, fps):
        import av

        self.av = av
        self.container = av.open(
            str(path),
            mode="w",
            options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
        )
        self.stream = self.container.add_stream("libx264", rate=max(1, round(fps)))
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = Fraction(1, 90000)
        self.stream.codec_context.time_base = Fraction(1, 90000)
        self.stream.codec_context.thread_count = 1
        self.stream.options = {
            "preset": "ultrafast",
            "crf": "21",
            "tune": "zerolatency",
        }
        self.last_pts = -1

    def append(self, pixels, seconds):
        frame = self.av.VideoFrame.from_ndarray(
            np.ascontiguousarray(pixels), format="rgb24"
        )
        frame.pts = max(self.last_pts + 1, round(seconds * 90000))
        frame.time_base = Fraction(1, 90000)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.last_pts = frame.pts

    def close(self):
        if self.container is not None:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                try:
                    self.container.close()
                finally:
                    self.container = None
                    self.stream = None


class Overview:
    def __init__(self, raw, width, height):
        import mujoco
        from robocasa.utils.camera_utils import DEFAULT_LAYOUT_CAM, LAYOUT_CAMS

        self.mujoco = mujoco
        self.model, self.data = raw.sim.model._model, raw.sim.data._data
        # Allocate a dedicated renderer without leaving model XML changes behind.
        old = self.model.vis.global_.offwidth, self.model.vis.global_.offheight
        try:
            self.model.vis.global_.offwidth = width
            self.model.vis.global_.offheight = height
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        finally:
            self.model.vis.global_.offwidth, self.model.vis.global_.offheight = old
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        layout = int(raw.layout_id)
        config = dict(LAYOUT_CAMS.get(layout, DEFAULT_LAYOUT_CAM))
        if layout == 1:
            config.update(
                lookat=[2.75, -1.25, 1], distance=6.6, azimuth=90, elevation=-30
            )
        for key, value in config.items():
            if key == "lookat":
                self.camera.lookat[:] = value
            else:
                setattr(self.camera, key, value)
        self.options = mujoco.MjvOption()
        self.options.geomgroup[0] = 0
        self.options.sitegroup[:] = 0
        self.hidden = {
            i
            for i in range(self.model.ngeom)
            if self.model.geom(i).name.startswith("wall_front")
            or "ceiling" in self.model.geom(i).name
        }

    def frame(self):
        self.renderer.update_scene(
            self.data, camera=self.camera, scene_option=self.options
        )
        for geom in self.renderer.scene.geoms[: self.renderer.scene.ngeom]:
            if (
                geom.objtype == self.mujoco.mjtObj.mjOBJ_GEOM
                and geom.objid in self.hidden
            ):
                geom.rgba[3] = 0
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


class Recording:
    def __init__(
        self,
        session,
        recording_id,
        root,
        width=640,
        height=480,
        *,
        clock=time.monotonic,
        overview_factory=Overview,
        writer_factory=MP4Writer,
    ):
        if not isinstance(recording_id, str) or not ID.fullmatch(recording_id):
            raise ValueError("Invalid recording_id")
        if (
            type(width) is not int
            or type(height) is not int
            or not (64 <= width <= 1920 and 64 <= height <= 1080)
            or width % 2
            or height % 2
        ):
            raise ValueError(
                "Video dimensions must be even, width 64..1920 and height 64..1080"
            )
        self.directory = Path(root) / recording_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.session, self.clock = session, clock
        self.started = clock()
        self.last_wall = 0.0
        self.start_sim = float(session.raw.sim.data.time)
        self.fps = float(session.raw.control_freq)
        self.index = 0
        self.active = True
        self.overview = None
        self.writers = {}
        self.last_frames = {}
        self.timeline = (self.directory / "video-timeline.jsonl").open("w")
        self.manifest = {
            "schema_version": 1,
            "recording_id": recording_id,
            "session_id": session.id,
            "created_at": datetime.now(UTC).isoformat(),
            "start_step": session.steps,
            "end_step": session.steps,
            "status": "recording",
            "errors": [],
            "control_fps": self.fps,
            "views": [],
            "files": ["video-timeline.jsonl", "video.json"],
            "sampling": "every completed control step; not every MuJoCo physics substep",
            "wall_clock": "actual elapsed time, including rendering/encoding/IO and waits",
            "human_view": "external cutaway spectator camera; not an agent observation",
        }
        self.manifest["trust"] = (
            "agent-side recording; not an independently trusted verifier verdict"
        )
        try:
            self.overview = overview_factory(session.raw, width, height)
            views = [("overview", "Human overview", width, height, None)]
            for i, key in enumerate(session.env.camera_keys):
                frame = np.asarray(session.observation[key])
                h, w, _ = frame.shape
                views.append((f"camera-{i}", key, w, h, key))
            for name, label, w, h, key in views:
                view = {
                    "id": name,
                    "label": label,
                    "width": w,
                    "height": h,
                    "observation_key": key,
                }
                for mode in ("wall", "simulation"):
                    filename = f"video-{name}-{mode}.mp4"
                    self.writers[name, mode] = writer_factory(
                        self.directory / filename, w, h, self.fps
                    )
                    view[mode] = filename
                    self.manifest["files"].append(filename)
                self.manifest["views"].append(view)
            self.capture("start", elapsed=0.0)
        except BaseException:
            self.stop("recording_error")
            raise

    def persist(self):
        self.manifest["end_step"] = self.session.steps
        self.manifest["step_frames"] = self.index
        temp = self.directory / "video.json.tmp"
        temp.write_text(json.dumps(self.manifest, indent=2) + "\n")
        temp.replace(self.directory / "video.json")

    def capture(self, event="step", elapsed=None):
        if not self.active:
            return
        wall = self.clock() - self.started if elapsed is None else elapsed
        frames = {"overview": self.overview.frame()}
        for view in self.manifest["views"][1:]:
            frames[view["id"]] = np.asarray(
                self.session.observation[view["observation_key"]]
            ).copy()
        for name, pixels in frames.items():
            self.writers[name, "wall"].append(pixels, wall)
            self.writers[name, "simulation"].append(pixels, self.index / self.fps)
        self.last_frames = frames
        self.event(event, wall, self.index / self.fps)
        self.index += 1
        self.last_wall = wall

    def event(self, event, wall, video_time):
        self.timeline.write(
            json.dumps(
                {
                    "event": event,
                    "step": self.session.steps,
                    "wall_seconds": wall,
                    "sim_seconds": float(self.session.raw.sim.data.time)
                    - self.start_sim,
                    "simulation_video_seconds": video_time,
                }
            )
            + "\n"
        )
        self.timeline.flush()

    def idle(self):
        if not self.active:
            return
        wall = self.clock() - self.started
        if wall - self.last_wall < 1:
            return
        # A held frame represents a real period without a control step. No render or step.
        for name, pixels in self.last_frames.items():
            self.writers[name, "wall"].append(pixels, wall)
        self.event("waiting", wall, max(0, self.index - 1) / self.fps)
        self.last_wall = wall

    def stop(self, reason="completed"):
        if not self.active:
            return self.manifest
        self.active = False
        wall = self.clock() - self.started
        self.event("stop", wall, max(0, self.index - 1) / self.fps)
        for (name, mode), writer in self.writers.items():
            try:
                if mode == "wall" and name in self.last_frames:
                    writer.append(self.last_frames[name], wall)
                writer.close()
            except Exception as exc:  # noqa: BLE001 -- Preserve all other views on encoder failure.
                self.manifest["errors"].append(str(exc))
        self.timeline.close()
        if self.overview is not None:
            self.overview.close()
            self.overview = None
        self.writers.clear()
        self.last_frames.clear()
        self.manifest.update(
            status=reason,
            wall_duration_seconds=wall,
            simulation_duration_seconds=self.index / self.fps,
        )
        self.persist()
        return self.manifest
