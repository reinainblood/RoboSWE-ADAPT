"""Explicit real-simulator lifecycle check, independent of the live notebook.

Run from the project root with its simulation Python environment. Writes only to
--output, creates two separate processes, and always terminates both.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from zipfile import ZipFile

from .client import SessionClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((Path(__file__).parent / "config.example.json").read_text())
    config["robot"] = args.robot
    results = []
    expected = None
    for role in ("agent", "verifier"):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = {
            **os.environ,
            "PYTHONHASHSEED": "0",
            "PYNPUT_BACKEND": "dummy",
            "ROBOT_SESSION_CONFIG": json.dumps(config),
        }
        if expected:
            env["ROBOT_SESSION_EXPECTED_INITIAL"] = expected
        with (output / f"{role}.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "robot_session.session",
                    "--role",
                    role,
                    "--port",
                    str(port),
                ],
                env=env,
                stdout=log,
                stderr=log,
            )
            client = SessionClient(f"http://127.0.0.1:{port}")
            try:
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{role} exited; see {log.name}")
                    try:
                        status = client.status()
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("startup")
                        time.sleep(1)
                if expected:
                    assert status["initial_fingerprint"] == expected
                expected = status["initial_fingerprint"]
                client.snapshot(output / f"{role}-initial.zip")
                action = [0.0] * status["action_space"]["shape"][0]
                client.step(action)
                client.finish()
                assert process.poll() is None
                assert client.status()["steps"] == 1
                try:
                    client.step(action)
                except HTTPError as exc:
                    assert exc.code == 409
                else:
                    raise AssertionError("held session accepted step")
                client.snapshot(output / f"{role}-final.zip")
                with ZipFile(output / f"{role}-final.zip") as archive:
                    manifest = json.loads(archive.read("manifest.json"))
                results.append(manifest)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    assert results[0]["session_id"] != results[1]["session_id"]
    assert results[0]["current_fingerprint"] == results[1]["current_fingerprint"]
    (output / "result.json").write_text(
        json.dumps({"passed": True, "sessions": results}, indent=2)
    )
    print(
        f"PASS: fresh entrypoint, same action result, snapshots, hold, teardown. Evidence: {output}"
    )


if __name__ == "__main__":
    main()
