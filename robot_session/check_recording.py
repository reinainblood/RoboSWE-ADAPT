"""Explicit local real-simulator recording check; no existing session is touched."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from roboswe.jobs import run_job


@dataclass
class LocalValidationHandle:
    url: str
    backend: str = "local-validation"

    def request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        with urlopen(
            Request(
                self.url + path, data=body, headers={"Content-Type": "application/json"}
            ),
            timeout=120,
        ) as r:
            return r.read()

    def execute(self, *command, timeout=120, stdin=None):
        command = list(command)
        if command[0] == "python":
            command[0] = sys.executable
        return subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "ROBOT_SESSION_URL": self.url},
            check=False,
        )

    def run_code(self, source, timeout=600):
        return self.execute("python", str(source), timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = {
        "robot": "PandaOmron",
        "task": "robocasa/PickPlaceCounterToCabinet",
        "split": None,
        "seed": 0,
        "env_kwargs": {
            "layout_ids": 1,
            "style_ids": 1,
            "camera_widths": 128,
            "camera_heights": 128,
        },
    }
    log = (output / "session.log").open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "robot_session.session", "--port", str(port)],
        stdout=log,
        stderr=log,
        env={
            **os.environ,
            "ROBOT_SESSION_CONFIG": json.dumps(config),
            "ROBOT_RECORDINGS_DIR": str(output / "recordings"),
        },
    )
    h = LocalValidationHandle(f"http://127.0.0.1:{port}")
    try:
        for _ in range(180):
            if process.poll() is not None:
                raise RuntimeError(
                    "Session startup failed: " + str(output / "session.log")
                )
            try:
                initial = json.loads(h.request("/health"))
                break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("Session startup timed out")
        source = output / "recording_probe.py"
        source.write_text("""import json,time,numpy as np
from robot_session.client import SessionClient
s=SessionClient();status=s.status();a=np.zeros(status['action_space']['shape'])
parts=status['action_space']['controllers'][0]['parts']
a[parts['base'][0]]=-.4
a[parts['base_mode'][0]]=1
for tick in range(12):
    if tick==6:time.sleep(1.2)
    s.step(a)
a[:]=0
s.step(a)
print(json.dumps({'step':s.status()['steps'],'message':'New recording validation, not the earlier oven opening'}))
""")
        good, result = run_job(
            h,
            source,
            jobs_dir=output / "jobs",
            timeout=60,
            name="Recording validation · new base movement",
        )
        if result.returncode:
            raise RuntimeError(result.stderr.decode())
        source.write_text("""import time,numpy as np
from robot_session.client import SessionClient
s=SessionClient();a=np.zeros(s.status()['action_space']['shape'])
for _ in range(4):s.step(a)
time.sleep(.5)
raise RuntimeError('Intentional failure to verify video retention')
""")
        failed, result = run_job(
            h,
            source,
            jobs_dir=output / "jobs",
            timeout=60,
            name="Recording validation · intentional exception",
        )
        assert result.returncode != 0
        import av

        summary = []
        for directory in (good, failed):
            record = json.loads((directory / "job.json").read_text())
            assert not record["artifact_errors"], record["artifact_errors"]
            assert len(record["video"]["views"]) == 4
            for view in record["video"]["views"]:
                for mode in ("wall", "simulation"):
                    with av.open(directory / view[mode]) as media:
                        frames = list(media.decode(video=0))
                        assert len(frames) >= 2
                        assert frames[-1].time > frames[0].time
            events = [
                json.loads(line)
                for line in (directory / "video-timeline.jsonl")
                .read_text()
                .splitlines()
            ]
            summary.append(
                {
                    "job": str(directory),
                    "status": record["status"],
                    "step_events": sum(e["event"] == "step" for e in events),
                    "wait_events": sum(e["event"] == "waiting" for e in events),
                    "video": record["video"],
                }
            )
        assert summary[0]["step_events"] == 13 and summary[0]["wait_events"] >= 1
        assert summary[1]["step_events"] == 4 and summary[1]["status"] == "failed"
        (output / "result.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "initial_session": initial["session_id"],
                    "runs": summary,
                },
                indent=2,
            )
        )
        print(output / "result.json", flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__ == "__main__":
    main()
