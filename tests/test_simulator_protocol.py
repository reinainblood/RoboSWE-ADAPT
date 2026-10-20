import json
import queue
from io import StringIO
from types import SimpleNamespace

import pytest

from roboswe.library.simulator import Digest, SimulatorProcess


def client():
    c = object.__new__(SimulatorProcess)
    c.sequence = 0
    c.events = queue.Queue()
    c.action_digest = Digest()
    c.proc = SimpleNamespace(stdin=StringIO(), poll=lambda: None)
    return c


def test_late_reply_cannot_be_mistaken_for_next_simulator_call():
    c = client()
    c.events.put({"id": 0, "result": {"value": "stale"}})
    c.events.put({"id": 1, "result": {"value": "current"}})
    assert c._call("observe", wait=0.1) == {"value": "current"}
    assert json.loads(c.proc.stdin.getvalue())["id"] == 1


def test_dead_pipe_keeps_exit_diagnostics(tmp_path):
    class Broken:
        def write(self, _):
            raise BrokenPipeError()

    c = client()
    c.proc = SimpleNamespace(stdin=Broken(), poll=lambda: 137)
    c.container = None
    c.diagnostics_path = tmp_path / "state.jsonl"
    with pytest.raises(RuntimeError, match="Simulator pipe closed"):
        c._call("observe")
    assert json.loads(c.diagnostics_path.read_text())["process_returncode"] == 137
