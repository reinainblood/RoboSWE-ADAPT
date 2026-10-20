"""Private coordinator transport. Never expose this pipe to generated code."""

import json
import signal
import sys
from pathlib import Path


def main():
    # Third-party simulator imports can print to stdout. Reserve the original
    # stream for protocol output before importing or constructing the simulator.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    from roboswe.library.runtime import LibraryRuntime
    from roboswe.library.serial import SerialRuntime

    from .session import RobotSession

    session = runtime = evidence = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            try:
                operation = request["operation"]
                if operation == "initialize" and session is None:

                    def expire(*_):
                        raise SystemExit("Simulator lifetime reached")

                    signal.signal(signal.SIGALRM, expire)
                    signal.alarm(request["lifetime_seconds"])
                    directory = Path(request["artifacts"])
                    directory.mkdir(parents=True, exist_ok=False)
                    session = RobotSession(request["config"], role=request["role"])
                    session.recording_root = directory / "recordings"
                    evidence = (directory / "actions.jsonl").open("x")
                    runtime_type = (
                        SerialRuntime
                        if request["spec"].get("mode") == "serial"
                        else LibraryRuntime
                    )
                    runtime = runtime_type(
                        session,
                        request["spec"],
                        evidence=evidence,
                        timeout=request["timeout"],
                        max_steps=request["max_steps"],
                    )
                    result = runtime.metadata()
                    result["benchmark_api_version"] = 3
                    result["sensor_contract_version"] = 2
                elif runtime is None:
                    raise ValueError("Initialize exactly one simulator first")
                elif operation == "start_task":
                    result = runtime.start_task(request["index"])
                elif operation == "bridge":
                    result = runtime.bridge(request["message"])
                elif operation == "observe":
                    result = runtime.observe()
                elif operation == "fingerprint":
                    result = session.fingerprint()
                elif operation == "finish_task":
                    result = runtime.finish_task(code_errors=request["code_errors"])
                elif operation == "record_start":
                    result = session.start_recording(request["settings"])
                elif operation == "record_stop":
                    result = session.stop_recording(request["settings"])
                elif operation == "close":
                    session.close()
                    session = None
                    print(
                        json.dumps({"id": request.get("id"), "result": {}}),
                        file=protocol,
                        flush=True,
                    )
                    break
                else:
                    raise ValueError("Unknown coordinator operation")
                print(
                    json.dumps(
                        {
                            "id": request.get("id"),
                            "result": result,
                            "action_sha256": runtime.action_digest.hexdigest(),
                        },
                        allow_nan=False,
                    ),
                    file=protocol,
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 -- Report simulator failures without executing further agent actions.
                print(
                    json.dumps(
                        {
                            "id": request.get("id"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ),
                    file=protocol,
                    flush=True,
                )
    finally:
        if session is not None:
            session.close()
        if evidence is not None:
            evidence.close()


if __name__ == "__main__":
    main()
