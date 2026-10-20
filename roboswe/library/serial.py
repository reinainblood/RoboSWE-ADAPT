"""Independent official dataset episodes with protected upstream predicates."""

import hashlib
import json
import time

from .runtime import controller_reply, public_observation
from .spec import goal_criteria


def serial_spec(task, *, timeout, max_steps):
    spec = {
        "schema_version": 1,
        "sensor_contract_version": 2,
        "id": "serial-official-v1",
        "mode": "serial",
        "task_timeout_seconds": timeout,
        "max_control_steps_per_task": max_steps,
        "settle_steps": 10,
        "tasks": [{"index": 0, "round": 1, "pair_key": task, "task": task}],
    }
    spec["sha256"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    return spec


class SerialRuntime:
    def __init__(self, session, spec, *, timeout=None, max_steps=None, evidence=None):
        self.session, self.spec, self.evidence = session, spec, evidence
        self.timeout = timeout or spec["task_timeout_seconds"]
        self.max_steps = max_steps or spec["max_control_steps_per_task"]
        self.action_digest = hashlib.sha256()
        self.started = None
        self.finished = False
        self.consecutive = 0
        self.initial_success = bool(session.raw._check_success())
        self.latest_success = self.initial_success

    def start_task(self, index):
        if index != 0 or self.started is not None:
            raise ValueError("Serial episodes contain one task")
        self.started = time.monotonic()
        self.start_fingerprint = self.session.fingerprint()
        return {
            "index": 0,
            "task": self.spec["tasks"][0]["task"],
            "round": 1,
            "instruction": self.session.observation[
                "annotation.human.task_description"
            ],
            "precondition_valid": not self.initial_success,
            "time_budget_seconds": self.timeout,
            "control_step_budget": self.max_steps,
            "required_sustained_steps": self.spec["settle_steps"],
            "success_criteria": goal_criteria(
                self.spec["tasks"][0]["task"], settle_steps=self.spec["settle_steps"]
            ),
            "observation": self.observe(),
        }

    def success(self):
        return (
            self.session.phase == "active"
            and not self.initial_success
            and self.consecutive >= self.spec["settle_steps"]
        )

    def observe(self):
        return {**public_observation(self.session), "task_success": self.success()}

    def bridge(self, message):
        if self.finished or self.started is None:
            raise RuntimeError("No active task")
        if time.monotonic() - self.started >= self.timeout:
            raise RuntimeError("Task wall-clock budget exhausted")
        if message == {"operation": "observe"}:
            return controller_reply(self.observe(), "observe")
        if set(message) != {"operation", "action"} or message["operation"] != "step":
            raise ValueError("Only observe and step are available")
        if self.session.steps >= self.max_steps:
            raise RuntimeError("Task control-step budget exhausted")
        self.session.step(message["action"])
        self.latest_success = bool(self.session.info["success"])
        self.consecutive = self.consecutive + 1 if self.latest_success else 0
        self.action_digest.update(
            json.dumps(
                message["action"], allow_nan=False, separators=(",", ":")
            ).encode()
            + b"\n"
        )
        if self.evidence is not None:
            self.evidence.write(
                json.dumps(
                    {
                        "control_step": self.session.steps,
                        "action": message["action"],
                        "upstream_success": self.latest_success,
                        "sustained_steps": self.consecutive,
                    }
                )
                + "\n"
            )
            self.evidence.flush()
        return controller_reply(self.observe(), "step")

    def finish_task(self, *, code_errors=0):
        if self.finished or self.started is None:
            raise ValueError("No active task")
        self.finished = True
        return {
            "index": 0,
            "round": 1,
            "pair_key": self.spec["tasks"][0]["pair_key"],
            "task": self.spec["tasks"][0]["task"],
            "elapsed_seconds": time.monotonic() - self.started,
            "instruction": self.session.observation[
                "annotation.human.task_description"
            ],
            "control_steps": self.session.steps,
            "start_fingerprint": self.start_fingerprint,
            "end_fingerprint": self.session.fingerprint(),
            "code_errors": code_errors,
            "precondition_valid": not self.initial_success,
            "observed_success": self.success(),
            "verdict": {
                "predicate_source": "trusted upstream task _check_success",
                "success": self.success(),
                "sustained_steps": self.consecutive,
                "final_predicate": self.latest_success,
            },
            "cumulative_action_sha256": self.action_digest.hexdigest(),
        }

    def metadata(self):
        return {
            "config": self.session.config,
            "initial_fingerprint": self.session.initial,
            "spec_sha256": self.spec["sha256"],
            "spec": self.spec,
            "bindings": {},
            "predicate_source": "trusted upstream task _check_success",
            "mode": "serial",
        }
