import hashlib
import time

import pytest

from roboswe.library.attempts import (
    AttemptRunner,
    Journal,
    read_journal,
    replay_journal,
)


class Runtime:
    def __init__(self):
        self.timeout = 60
        self.started = time.monotonic()
        self.steps = 0
        self.action_digest = hashlib.sha256()

    def metadata(self):
        return {}

    def start_task(self, index):
        return {"index": index, "observation": {}}

    def bridge(self, message):
        if message["operation"] == "step":
            self.steps += 1
            self.action_digest.update(b"step")
        return {"control_step": self.steps}

    def finish_task(self, *, code_errors):
        return {"code_errors": code_errors}


class Sandbox:
    image = "test-image"

    def request(self, operation, path, *, content, bridge, timeout):
        try:
            for _ in range(10):
                bridge({"operation": "step", "action": [0]})
            return {"returncode": 0, "error": None}
        except RuntimeError as exc:
            return {"returncode": 1, "error": str(exc)}


def test_save_receipt_cannot_be_mistaken_for_execution_output(tmp_path):
    class FileSandbox:
        image = "test-image"

        def request(self, operation, path, *, content, bridge, timeout):
            assert operation == "write"
            return {"returncode": 0, "error": None, "output": "", "result": {"bytes": 3704}}

    journal = Journal(tmp_path / "journal.jsonl")
    runtime = Runtime()
    runner = AttemptRunner(runtime, FileSandbox(), journal)
    runner.start_task(0)
    result = runner.apply({"operation": "write", "path": "map.py", "content": "# source", "message": "save"})
    journal.close()
    assert result["operation"] == "write"
    assert result["status"] == "file_saved"
    assert result["bytes_written"] == 3704
    assert result["code_executed"] is False
    assert result["physics_advanced"] is False
    assert runtime.steps == 0
    assert runner.execution_attempts == 0


def test_infrastructure_failure_is_journaled_then_stops_the_campaign(tmp_path):
    class FailedCleanup(Sandbox):
        def request(self, operation, path, *, content, bridge, timeout):
            bridge({"operation": "step", "action": [0]})
            return {
                "returncode": 137,
                "error": "execution_timeout",
                "infrastructure_error": "container_cleanup_failed: Docker unavailable",
            }

    journal = Journal(tmp_path / "journal.jsonl")
    runner = AttemptRunner(Runtime(), FailedCleanup(), journal)
    runner.start_task(0)
    with pytest.raises(RuntimeError, match="infrastructure failed"):
        runner.apply(
            {
                "operation": "execute",
                "path": "probe.py",
                "content": "",
                "message": "test",
            }
        )
    journal.close()
    last = read_journal(journal.path)[-1]
    assert last["kind"] == "operation_result"
    assert last["data"]["result"]["accepted_control_steps"] == 1
    assert last["data"]["result"]["error"] == "execution_timeout"
    assert runner.code_errors == runner.execution_errors == 0


def test_attempt_budget_stops_script_before_task_budget_and_keeps_state(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    runtime = Runtime()
    runner = AttemptRunner(runtime, Sandbox(), journal)
    runner.start_task(0)
    decision = {
        "operation": "execute",
        "path": "probe.py",
        "content": "",
        "message": "test",
    }
    first = runner.apply(decision, max_control_steps=3)
    second = runner.apply(decision, max_control_steps=2)
    trial = runner.finish_task()
    journal.close()
    assert runtime.steps == 5
    assert first["status"] == "execution_failed"
    assert first["physics_advanced"] is True
    assert first["operation"] == "execute"
    assert (
        first["accepted_control_steps"] == 3 and second["accepted_control_steps"] == 2
    )
    assert trial["execution_attempts"] == trial["execution_errors"] == 2
    assert trial["execution_error_rate"] == 1
    recorded = [
        e["data"]["max_control_steps"]
        for e in read_journal(journal.path)
        if e["kind"] == "operation_request"
    ]
    assert recorded == [3, 2]


@pytest.mark.parametrize("reason", ["agent_finish", "infrastructure_interruption"])
def test_timeout_replay_runs_code_and_stops_at_recorded_control_boundary(
    tmp_path, reason
):
    class ReplayRuntime(Runtime):
        def metadata(self):
            return {
                "config": {},
                "initial_fingerprint": "initial",
                "spec_sha256": "spec",
                "bindings": {},
            }

        def finish_task(self, *, code_errors):
            return {
                "observed_success": False,
                "precondition_valid": True,
                "code_errors": code_errors,
                "cumulative_action_sha256": self.action_digest.hexdigest(),
            }

    class TimedOut(Sandbox):
        def request(self, operation, path, *, content, bridge, timeout):
            for _ in range(3):
                bridge({"operation": "step", "action": [0]})
            return {"returncode": 137, "error": "execution_timeout"}

    original = Journal(tmp_path / "original.jsonl")
    runner = AttemptRunner(ReplayRuntime(), TimedOut(), original)
    runner.start_task(0)
    runner.apply(
        {"operation": "execute", "path": "probe.py", "content": "", "message": "test"}
    )
    runner.finish_task(reason=reason)
    original.close()
    runtime = ReplayRuntime()
    # This sandbox would execute ten steps if the verifier merely repeated the
    # wall timeout. The recorded boundary must stop its actual execution at three.
    result = replay_journal(
        original.path, runtime, Sandbox(), tmp_path / "replay.jsonl"
    )
    assert runtime.steps == 3 and result["reproduced"]
    assert not result["trials"][0]["verified_success"]
    assert result["trials"][0]["infrastructure_valid"] == (reason == "agent_finish")
@pytest.mark.parametrize('shorten_wait',[False,True])
def test_passive_wait_replay_preserves_recorded_ticks(tmp_path,shorten_wait):
    from pairing.simulation_wait import WaitingAttemptRunner

    class WaitRuntime(Runtime):
        passive_wait_steps=0

        def metadata(self):
            return {'config':{},'initial_fingerprint':'initial','spec_sha256':'spec','bindings':{}}

        def wait(self,steps):
            actual=steps-1 if getattr(self,'shorten',False) else steps
            for _ in range(actual):self.bridge({'operation':'step','action':[0]})
            self.passive_wait_steps+=actual
            return {'control_step':self.steps,'passive_wait':{'requested_steps':steps,
                'advanced_steps':actual,'episode_wait_steps':self.passive_wait_steps,'host_seconds':2.0}}

        def finish_task(self,*,code_errors):
            return {'observed_success':True,'precondition_valid':True,'code_errors':code_errors,
                'cumulative_action_sha256':self.action_digest.hexdigest()}

    original=Journal(tmp_path/'original.jsonl')
    runner=WaitingAttemptRunner(WaitRuntime(),Sandbox(),original)
    runner.start_task(0)
    runner.apply({'operation':'execute','path':'move.py','content':'','message':'move'})
    runner.apply({'operation':'wait','path':'','content':'3','message':'settle'})
    runner.finish_task();original.close()
    runtime=WaitRuntime();runtime.shorten=shorten_wait
    result=replay_journal(original.path,runtime,Sandbox(),tmp_path/'replay.jsonl',runner_factory=WaitingAttemptRunner)
    assert result['reproduced'] is not shorten_wait
    assert result['trials'][0]['verified_success'] is not shorten_wait
    if shorten_wait:
        assert any(x['reason']=='passive_wait_diverged' for x in result['mismatches'])
