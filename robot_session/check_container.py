"""Explicit real-container check; never called by notebook or session startup.

Exercises submitted-code isolation, inspection, memory resume, fresh replay and
cleanup. Docker memory checkpoints require CRIU and retain daemon containers.
"""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from .backends import replay, resume, start

SOURCE = """from pathlib import Path
from robot_session.client import SessionClient
session = SessionClient()
session.step([0.0] * session.status()["action_space"]["shape"][0])
Path("/workspace/probe-marker").write_text("persisted")
print("ACTION_DONE")
"""


def manifest(path):
    with ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("docker", "modal"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((Path(__file__).parent / "config.example.json").read_text())
    config["robot"] = args.robot
    code = output / "submitted.py"
    code.write_text(SOURCE)
    handles = []
    result = {"backend": args.backend, "image": args.image, "passed": False}
    try:
        original = start(
            args.backend,
            args.image,
            config,
            receipt=output / "agent.json",
            lifetime_seconds=1800,
        )
        handles.append(original)
        print("AGENT_READY", original.container_id, flush=True)
        run = original.run_code(code)
        assert run.returncode == 0, run.stderr
        assert b"ACTION_DONE" in run.stdout
        assert original.status()["steps"] == 1
        failure = output / "failure.py"
        failure.write_text('raise RuntimeError("intentional code failure")\n')
        assert original.run_code(failure).returncode != 0
        assert original.status()["steps"] == 1
        original.snapshot(output / "agent-step1.zip")
        result["agent"] = manifest(output / "agent-step1.zip")
        print("AGENT_RUN_AND_SNAPSHOT_OK", flush=True)
        active = original
        if args.checkpoint:
            original.checkpoint(output / "checkpoint.json")
            original.save(output / "agent.json")
            print("CHECKPOINT_OK", flush=True)
            active = resume(output / "checkpoint.json", receipt=output / "resumed.json")
            if active.container_id != original.container_id:
                handles.append(active)
            marker = active.execute("cat", "/workspace/probe-marker")
            assert marker.returncode == 0 and marker.stdout == b"persisted"
            active.snapshot(output / "resumed.zip")
            assert (
                manifest(output / "resumed.zip")["current_fingerprint"]
                == result["agent"]["current_fingerprint"]
            )
            assert active.run_code(code).returncode == 0
            assert active.status()["steps"] == 2
            print("MEMORY_FILESYSTEM_RESUME_AND_STEP_OK", flush=True)
            result["checkpoint_resume"] = True
        active.finish()
        active.snapshot(output / "agent-final.zip")
        assert active.status()["phase"] == "held"
        verifier = replay(original, receipt=output / "verifier.json")
        handles.append(verifier)
        assert verifier.container_id != original.container_id
        assert verifier.status()["session_id"] != result["agent"]["session_id"]
        assert verifier.status()["steps"] == 0
        assert (
            verifier.execute("test", "!", "-e", "/workspace/probe-marker").returncode
            == 0
        )
        assert verifier.run_code(code).returncode == 0
        verifier.finish()
        verifier.snapshot(output / "verifier-final.zip")
        result["verifier"] = manifest(output / "verifier-final.zip")
        assert (
            result["verifier"]["current_fingerprint"]
            == result["agent"]["current_fingerprint"]
        )
        result["passed"] = True
        print("FRESH_REPLAY_OK", flush=True)
    finally:
        result["containers"] = [h.container_id for h in handles]
        cleanup = []
        for handle in reversed(handles):
            try:
                if handle.backend == "docker" and handle.checkpointed:
                    # Preserve CRIU data but stop any resumed process.
                    import subprocess

                    subprocess.run(
                        ["docker", "stop", handle.container_id],
                        check=True,
                        capture_output=True,
                    )
                    cleanup.append(
                        {
                            "id": handle.container_id,
                            "state": "stopped; retained for checkpoint",
                        }
                    )
                else:
                    handle.stop()
                    cleanup.append({"id": handle.container_id, "state": "stopped"})
            except Exception as exc:  # noqa: BLE001 -- continue cleanup of independently owned probes
                cleanup.append({"id": handle.container_id, "error": repr(exc)})
        result["cleanup"] = cleanup
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("CLEANUP", cleanup, flush=True)
    print("PASS", output, flush=True)


if __name__ == "__main__":
    main()
