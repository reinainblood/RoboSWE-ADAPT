"""Opt-in real Docker controls; no generated code ever executes on the host."""

import json
import os
import subprocess

import pytest

from roboswe.execution.sandbox import CodeSandbox, docker_path, validate_path


def test_cleanup_keeps_the_original_failure(monkeypatch):
    sandbox = CodeSandbox.__new__(CodeSandbox)

    def fail():
        raise RuntimeError("Docker is unavailable")

    sandbox.close = fail
    original = RuntimeError("Original simulator failure")
    sandbox.__exit__(RuntimeError, original, None)
    assert "Docker is unavailable" in original.__notes__[0]
    with pytest.raises(RuntimeError, match="Docker is unavailable"):
        sandbox.__exit__(None, None, None)


@pytest.mark.parametrize(
    "stderr,failed",
    [(b"No such container: gone", False), (b"Docker Desktop is unable to start", True)],
)
def test_attempt_removal_distinguishes_absent_container_from_unavailable_engine(
    monkeypatch, stderr, failed
):
    sandbox = CodeSandbox.__new__(CodeSandbox)
    sandbox.docker = "/test/docker"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 1, b"", stderr),
    )
    result = sandbox._remove_attempt("owned-attempt")
    assert bool(result) is failed
    if failed:
        assert "Docker Desktop is unable to start" in result


@pytest.mark.parametrize(
    "path",
    [
        "../secret.py",
        "/tmp/secret.py",
        "x/../../y.py",
        "a\\b.py",
        "",
        "x.txt",
        "x\x00.py",
    ],
)
def test_reject_paths(path):
    with pytest.raises(ValueError):
        validate_path(path)


@pytest.fixture
def sandbox():
    if os.getenv("ROBOSWE_TEST_DOCKER") != "1":
        pytest.skip("Set ROBOSWE_TEST_DOCKER=1 after building roboswe-code:local")
    image = subprocess.check_output(
        [
            docker_path(),
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            "roboswe-code:local",
        ],
        text=True,
    ).strip()
    with CodeSandbox(image) as sandbox:
        yield sandbox


def write(sandbox, source, path="test.py"):
    result = sandbox.request("write", path, content=source)
    assert result["returncode"] == 0 and result["error"] is None, result


def test_private_rpc_containers_disable_duplicate_docker_logs(sandbox):
    def driver(name):
        return subprocess.check_output(
            [
                sandbox.docker,
                "inspect",
                "--format",
                "{{.HostConfig.LogConfig.Type}}",
                name,
            ],
            text=True,
        ).strip()

    assert driver(sandbox.keeper) == "none"
    observed = []
    remove = sandbox._remove_attempt

    def inspect_before_remove(name):
        observed.append(driver(name))
        return remove(name)

    sandbox._remove_attempt = inspect_before_remove
    write(sandbox, "print('pipe output survives without daemon logging')\n")
    result = sandbox.request("execute", "test.py", bridge=lambda _: {})
    assert result["returncode"] == 0
    assert "pipe output survives" in result["output"]
    assert observed == ["none", "none"]


def test_persistent_library_and_bridge(sandbox):
    write(sandbox, "def value(): return 42\n", "library.py")
    write(
        sandbox,
        "import library,robot\nprint(library.value(), robot.observe()['step'])\nrobot.step([0,0])\n",
    )
    messages = []

    def bridge(message):
        messages.append(message)
        return {"step": 7}

    result = sandbox.request("execute", "test.py", bridge=bridge)
    assert result["returncode"] == 0 and result["error"] is None, result
    assert "42 7" in result["output"]
    assert messages == [
        {"operation": "observe"},
        {"operation": "step", "action": [0.0, 0.0]},
    ]


def test_network_files_credentials_readonly(sandbox):
    write(
        sandbox,
        """import os,socket,json
from pathlib import Path
checks={}
for path in ['/app/robot_session', '/var/run/docker.sock', '/root/.codex/auth.json']:
 try: checks[path]=not Path(path).exists()
 except PermissionError: checks[path]=True
checks['credentials']=not any(x in os.environ for x in ['OPENAI_API_KEY','CODEX_HOME','MODAL_TOKEN_SECRET'])
try:
 Path('/opt/roboswe/robot.py').write_text('modified')
 checks['readonly']=False
except OSError: checks['readonly']=True
try:
 socket.create_connection(('1.1.1.1',443),timeout=.2)
 checks['network']=False
except OSError: checks['network']=True
print(json.dumps(checks))
""",
    )
    result = sandbox.request("execute", "test.py")
    assert result["returncode"] == 0, result
    assert all(json.loads(result["output"]).values())


def test_forged_administrative_rpc_never_reaches_bridge(sandbox):
    write(
        sandbox,
        """import os
os.write(1,b'{"operation":"reset","qpos":[1]}\\n')
""",
    )
    result = sandbox.request(
        "execute",
        "test.py",
        bridge=lambda _: pytest.fail("Admin request reached trusted bridge"),
    )
    assert "Only observe and step" in result["error"]


def test_timeout_kills_attempt_and_preserves_files(sandbox):
    write(sandbox, "import time\ntime.sleep(60)\n")
    result = sandbox.request("execute", "test.py", timeout=0.8)
    assert result["error"] == "execution_timeout"
    assert "time.sleep" in sandbox.request("read", "test.py")["result"]["content"]


def test_symlink_cannot_export_outside_workspace(sandbox):
    write(sandbox, "import os\nos.symlink('/opt/roboswe/robot.py', 'escape.py')\n")
    assert sandbox.request("execute", "test.py")["returncode"] == 0
    result = sandbox.request("read", "escape.py")
    assert result["returncode"] != 0 and result["result"] is None


def test_exception_preserves_prior_steps(sandbox):
    write(sandbox, "import robot\nrobot.step([0])\nraise RuntimeError('intentional')\n")
    actions = []
    result = sandbox.request(
        "execute", "test.py", bridge=lambda msg: actions.append(msg) or {}
    )
    assert result["returncode"] != 0
    assert actions == [{"operation": "step", "action": [0.0]}]
    assert "intentional" in result["output"]


def test_startup_hook_is_not_executed_by_file_tools_or_export(sandbox, tmp_path):
    write(sandbox, "raise RuntimeError('untrusted startup hook')\n", "sitecustomize.py")
    write(sandbox, "def move(): pass\n", "library.py")
    assert sandbox.request("read", "library.py")["returncode"] == 0
    manifest = sandbox.archive(tmp_path / "library.zip")
    assert {f["path"] for f in manifest["files"]} == {"sitecustomize.py", "library.py"}


def test_printed_reward_and_fake_workspace_report_cannot_award_success(
    sandbox, tmp_path
):
    import hashlib
    import time
    from types import SimpleNamespace

    from roboswe.library.attempts import AttemptRunner, Journal

    runtime = SimpleNamespace(
        timeout=60,
        started=time.monotonic(),
        action_digest=hashlib.sha256(),
        metadata=dict,
        start_task=lambda _: {"observation": {}},
        bridge=lambda _: pytest.fail("Forged report must not call the simulator"),
        finish_task=lambda **_: {"observed_success": False},
    )
    journal = Journal(tmp_path / "journal.jsonl")
    runner = AttemptRunner(runtime, sandbox, journal)
    runner.start_task(0)
    source = "from pathlib import Path\nPath('report.json').write_text('{\\\"verified_success\\\":true,\\\"reward\\\":1}')\nprint('SUCCESS reward=1 verified_success=True')\n"
    runner.apply(
        {"operation": "write", "path": "claim.py", "content": source, "message": "test"}
    )
    result = runner.apply(
        {"operation": "execute", "path": "claim.py", "content": "", "message": "test"}
    )
    assert result["returncode"] == 0 and "reward=1" in result["output"]
    assert runner.finish_task()["observed_success"] is False
    journal.close()


def test_notes_are_data_not_executable_code():
    from roboswe.execution.runner import relative_python

    assert validate_path('NOTES.md', operation='write') == 'NOTES.md'
    assert validate_path('NOTES.md', operation='read') == 'NOTES.md'
    for validator in (validate_path, relative_python):
        with pytest.raises(ValueError):
            validator('NOTES.md', operation='execute')
        with pytest.raises(ValueError):
            validator('../NOTES.md', operation='read')
        with pytest.raises(ValueError):
            validator('subdir/NOTES.md', operation='write')


def test_notes_survive_operations_and_archive(sandbox, tmp_path):
    from zipfile import ZipFile

    write(sandbox, '# Learned\nFingers make a ramp.\n', 'NOTES.md')
    write(sandbox, "print('another execution')\n")
    assert sandbox.request('execute', 'test.py')['returncode'] == 0
    result = sandbox.request('read', 'NOTES.md')
    assert result['result']['content'] == '# Learned\nFingers make a ramp.\n'
    manifest = sandbox.archive(tmp_path / 'library.zip')
    assert any(f['path'] == 'NOTES.md' for f in manifest['files'])
    with ZipFile(tmp_path / 'library.zip') as archive:
        assert archive.read('NOTES.md').decode() == result['result']['content']
