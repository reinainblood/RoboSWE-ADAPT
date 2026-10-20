"""Host-owned submission history and chronological code replay.

The journal is evidence from a trusted coordinator, not a credential: hash chaining
detects accidental edits, but does not authenticate an arbitrary uploaded journal.
"""

import hashlib
import json
import time
from pathlib import Path

from roboswe.execution.sandbox import validate_path
from roboswe.storage import require_disk_headroom


def canonical(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x")
        self.digest = "0" * 64
        self.index = 0

    def append(self, kind, data):
        entry = {
            "index": self.index,
            "previous_sha256": self.digest,
            "kind": kind,
            "data": data,
        }
        self.digest = hashlib.sha256(canonical(entry)).hexdigest()
        entry["sha256"] = self.digest
        self.file.write(json.dumps(entry, allow_nan=False) + "\n")
        self.file.flush()
        self.index += 1

    def close(self):
        self.file.close()


def read_journal(path):
    previous = "0" * 64
    entries = []
    with Path(path).open() as source:
        for index, line in enumerate(source):
            entry = json.loads(line)
            digest = entry.pop("sha256")
            if (
                entry["index"] != index
                or entry["previous_sha256"] != previous
                or hashlib.sha256(canonical(entry)).hexdigest() != digest
            ):
                raise ValueError("Submission journal was changed or reordered")
            previous = digest
            entries.append(entry)
    return entries


class AttemptRunner:
    def __init__(self, runtime, sandbox, journal):
        self.runtime, self.sandbox, self.journal = runtime, sandbox, journal
        self.code_errors = 0
        self.execution_attempts = 0
        self.execution_errors = 0
        self.file_errors = 0
        self.task_active = False
        self.journal.append(
            "initial", {**runtime.metadata(), "code_image": sandbox.image}
        )

    def start_task(self, index):
        self.journal.append("start_task_request", {"index": index})
        result = self.runtime.start_task(index)
        self.code_errors = 0
        self.execution_attempts = 0
        self.execution_errors = 0
        self.file_errors = 0
        self.task_active = True
        self.journal.append(
            "start_task_result", {k: v for k, v in result.items() if k != "observation"}
        )
        return result

    def apply(
        self,
        decision,
        *,
        timeout=60,
        max_control_steps=None,
        replay_stop_after_steps=None,
    ):
        if not self.task_active:
            raise ValueError("No active task")
        if (
            not isinstance(decision, dict)
            or set(decision) != {"operation", "path", "content", "message"}
            or not all(isinstance(x, str) for x in decision.values())
        ):
            raise ValueError("Invalid structured decision")
        operation = decision["operation"]
        if operation not in {"write", "read", "execute", "observe", "finish"}:
            raise ValueError("Unknown operation")
        if operation in {"write", "read", "execute"}:
            validate_path(decision["path"], operation=operation)
        elif decision["path"]:
            raise ValueError("This operation takes no path")
        if operation != "write" and decision["content"]:
            raise ValueError("Only write accepts content")
        remaining = self.runtime.timeout - (time.monotonic() - self.runtime.started)
        if remaining <= 0:
            raise TimeoutError("Task wall-clock budget exhausted")
        timeout = min(timeout, remaining)
        if max_control_steps is not None and (
            type(max_control_steps) is not int or max_control_steps <= 0
        ):
            raise ValueError("Attempt control-step budget must be a positive integer")
        if replay_stop_after_steps is not None and (
            type(replay_stop_after_steps) is not int or replay_stop_after_steps < 0
        ):
            raise ValueError("Recorded interruption step count must be nonnegative")
        self.journal.append(
            "operation_request",
            {
                "decision": decision,
                "timeout": timeout,
                "max_control_steps": max_control_steps,
                "replay_stop_after_steps": replay_stop_after_steps,
            },
        )
        if operation == "finish":
            result = self.finish_task()
        elif operation == "observe":
            result = self.runtime.bridge({"operation": "observe"})
        else:
            accepted_steps = 0
            replay_boundary_reached = False

            def bridge(message):
                nonlocal accepted_steps, replay_boundary_reached
                if message.get("operation") == "step" and replay_stop_after_steps == 0:
                    replay_boundary_reached = True
                    raise RuntimeError(
                        "Recorded execution interruption boundary reached"
                    )
                if (
                    message.get("operation") == "step"
                    and max_control_steps is not None
                    and accepted_steps >= max_control_steps
                ):
                    raise RuntimeError(
                        "Execution control-step limit reached; this script has stopped. "
                        "Inspect current task feedback for the remaining episode budget before continuing."
                    )
                observation = self.runtime.bridge(message)
                if message.get("operation") == "step":
                    accepted_steps += 1
                    if (
                        replay_stop_after_steps is not None
                        and accepted_steps == replay_stop_after_steps
                    ):
                        replay_boundary_reached = True
                        raise RuntimeError(
                            "Recorded execution interruption boundary reached"
                        )
                return observation

            if operation == "execute":
                self.execution_attempts += 1
            result = self.sandbox.request(
                operation,
                decision["path"],
                content=decision["content"],
                bridge=bridge,
                timeout=timeout,
            )
            if (
                (result["returncode"] or result["error"])
                and not result.get("infrastructure_error")
                and result.get("error") != "container_cleanup_failed"
            ):
                self.code_errors += 1
                if operation == "execute":
                    self.execution_errors += 1
                else:
                    self.file_errors += 1
            result["accepted_control_steps"] = accepted_steps
            failed = bool(result.get("returncode") or result.get("error"))
            result["operation"] = operation
            result["path"] = decision["path"]
            result["physics_advanced"] = accepted_steps > 0
            if operation == "write":
                result["code_executed"] = False
                result["status"] = "write_failed" if failed else "file_saved"
                result["message"] = (
                    "File write failed; code was not executed."
                    if failed else
                    "Source file saved only. No code ran and no robot command was sent. "
                    "Use execute with this path to run it."
                )
                if not failed:
                    result["bytes_written"] = (result.get("result") or {}).get("bytes")
            elif operation == "read":
                result["code_executed"] = False
                result["status"] = "read_failed" if failed else "file_read"
                result["message"] = "File contents are in result.content; reading does not execute code."
            else:
                result["status"] = "execution_failed" if failed else "execution_completed"
                result["message"] = (
                    "Execution attempt finished. output contains captured print output and stderr; "
                    "an empty output means no captured text, not a byte count. "
                    "accepted_control_steps counts completed physics ticks, not achieved poses. "
                    "Inspect measured sensors and task_success; an error does not reset prior actions."
                )
            if replay_stop_after_steps is not None:
                result["replay_boundary_reached"] = replay_boundary_reached
        # Sensor images are separately recorded by the simulator. Keep the journal
        # focused on code decisions and trusted task/action outcomes.
        summary = (
            result
            if operation != "observe"
            else {"control_step": result["control_step"]}
        )
        self.journal.append(
            "operation_result",
            {
                "operation": operation,
                "result": summary,
                "cumulative_action_sha256": self.runtime.action_digest.hexdigest(),
            },
        )
        if (
            result.get("infrastructure_error")
            or result.get("error") == "container_cleanup_failed"
        ):
            raise RuntimeError(
                "Code sandbox infrastructure failed: "
                + str(result.get("infrastructure_error") or result["error"])
            )
        return result

    def finish_task(self, *, reason="agent_finish"):
        if not self.task_active:
            raise ValueError("No active task")
        result = self.runtime.finish_task(code_errors=self.code_errors)
        result["termination_reason"] = reason
        result.update(
            execution_attempts=self.execution_attempts,
            execution_errors=self.execution_errors,
            file_errors=self.file_errors,
            execution_error_rate=self.execution_errors / self.execution_attempts
            if self.execution_attempts
            else None,
        )
        self.task_active = False
        self.journal.append("finish_task", result)
        return result


def replay_journal(
    original, runtime, sandbox, output, *, on_start=None, on_finish=None,
    runner_factory=AttemptRunner,
):
    """Execute the original code operations in fresh simulator/sandbox instances.

    Timing is retained from the live trial; faster replay (without model thinking)
    must never be scored as a learning improvement. Divergence is explicit and
    prevents reporting a reproduced trial as a verified success.
    """
    entries = read_journal(original)
    if not entries or entries[0]["kind"] != "initial":
        raise ValueError("Missing initial provenance")
    initial = entries[0]["data"]
    current = runtime.metadata()
    for key in ("config", "initial_fingerprint", "spec_sha256", "bindings"):
        if initial[key] != current[key]:
            raise ValueError(f"Replay provenance mismatch: {key}")
    for key in ("simulator_image", "development_runtime"):
        if initial.get(key) != current.get(key):
            raise ValueError(f"Replay runtime mismatch: {key}")
    if initial["code_image"] != sandbox.image:
        raise ValueError("Replay code image mismatch")
    journal = Journal(output)
    runner = runner_factory(runtime, sandbox, journal)
    results = []
    mismatches = []
    pending = False
    replacement_sandboxes = []
    try:
        for position, entry in enumerate(entries[1:], start=1):
            require_disk_headroom(Path(output).parent)
            kind, data = entry["kind"], entry["data"]
            if kind == "reset_code_workspace":
                if runner.task_active or pending:
                    raise ValueError(
                        "Workspace resets are only allowed at task boundaries"
                    )
                from roboswe.execution.sandbox import CodeSandbox

                prior = runner.sandbox
                prior.close()
                runner.sandbox = CodeSandbox(
                    prior.image, memory_mb=prior.memory_mb, cpu=prior.cpu
                )
                replacement_sandboxes.append(runner.sandbox)
                journal.append("reset_code_workspace", {})
            elif kind == "start_task_request":
                if on_start is not None:
                    on_start(data["index"])
                runner.start_task(data["index"])
            elif kind == "operation_request":
                if pending:
                    raise ValueError("Unfinished operation in original journal")
                pending = True
                # finish() is handled at its explicit task boundary below.
                decision = data["decision"]
                if decision["operation"] == "wait":
                    if data.get("policy") != "passive_wait_v1":
                        raise ValueError("Unsupported recorded passive-wait policy")
                    actual = runner.apply(decision)
                elif decision["operation"] != "finish":
                    next_entry = (
                        entries[position + 1] if position + 1 < len(entries) else None
                    )
                    expected_result = (
                        next_entry["data"].get("result", {})
                        if next_entry and next_entry["kind"] == "operation_result"
                        else {}
                    )
                    original_error = expected_result.get("error") or ""
                    interrupted = decision["operation"] == "execute" and (
                        original_error == "execution_timeout"
                        or "Task wall-clock budget exhausted" in original_error
                    )
                    stop_steps = data.get("replay_stop_after_steps")
                    if stop_steps is None and interrupted:
                        stop_steps = expected_result.get("accepted_control_steps")
                    # Verification has its own resource allowance. Original time
                    # remains the scored time; recorded control boundaries prevent
                    # a faster replay from advancing beyond a partial attempt.
                    actual = runner.apply(
                        decision,
                        timeout=max(data["timeout"], 120)
                        if stop_steps is not None
                        else data["timeout"],
                        max_control_steps=data.get("max_control_steps"),
                        replay_stop_after_steps=stop_steps,
                    )
                else:
                    actual = None
            elif kind == "operation_result":
                if not pending:
                    raise ValueError("Unpaired operation result")
                pending = False
                if (
                    data["cumulative_action_sha256"]
                    != runtime.action_digest.hexdigest()
                ):
                    mismatches.append(
                        {"event": entry["index"], "reason": "action_sequence_diverged"}
                    )
                if data["operation"] in {"write", "read", "execute"}:
                    expected = data["result"]
                    boundary_equivalent = actual.get(
                        "replay_boundary_reached"
                    ) and actual.get("accepted_control_steps") == expected.get(
                        "accepted_control_steps"
                    )
                    if not boundary_equivalent and (
                        actual["returncode"] != expected["returncode"]
                        or bool(actual["error"]) != bool(expected["error"])
                    ):
                        mismatches.append(
                            {
                                "event": entry["index"],
                                "reason": "execution_outcome_diverged",
                            }
                        )
                elif data["operation"] == "wait":
                    expected = data["result"]
                    if actual["control_step"] != expected["control_step"] or any(
                        actual["passive_wait"][key] != expected["passive_wait"][key]
                        for key in ("requested_steps", "advanced_steps", "episode_wait_steps")
                    ):
                        mismatches.append({"event": entry["index"], "reason": "passive_wait_diverged"})
            elif kind == "finish_task":
                replayed = runner.finish_task(
                    reason=data.get("termination_reason", "agent_finish")
                )
                if any(
                    data[k] != replayed[k]
                    for k in ("observed_success", "precondition_valid")
                ):
                    mismatches.append(
                        {"event": entry["index"], "reason": "physical_verdict_diverged"}
                    )
                reproduced = (
                    data["cumulative_action_sha256"]
                    == replayed["cumulative_action_sha256"]
                    and not mismatches
                )
                infrastructure_valid = (
                    reproduced
                    and data.get("infrastructure_valid", True) is True
                    and data.get("termination_reason") != "infrastructure_interruption"
                )
                results.append(
                    {
                        **data,
                        "verified_success": bool(
                            infrastructure_valid
                            and data["observed_success"]
                            and replayed["observed_success"]
                        ),
                        "reproduced": reproduced,
                        "infrastructure_valid": infrastructure_valid,
                        "replay": replayed,
                    }
                )
                if on_finish is not None:
                    on_finish(results[-1])
            elif kind != "start_task_result":
                raise ValueError(f"Unknown journal event {kind}")
        if pending or runner.task_active:
            raise ValueError("Original journal is incomplete; cannot certify replay")
        return {
            "kind": "independent_code_replay",
            "reproduced": not mismatches,
            "mismatches": mismatches,
            "trials": results,
        }
    finally:
        journal.close()
        for replacement in replacement_sandboxes:
            replacement.close()
