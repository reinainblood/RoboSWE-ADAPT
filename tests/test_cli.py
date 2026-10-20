import json

import pytest

from roboswe.catalog import load_catalog, select_tasks
from roboswe.cli import main, parser_for_cli
from roboswe.plan import build_plan


@pytest.fixture
def dataset(tmp_path):
    registry = tmp_path / "robocasa/utils/dataset_registry.py"
    registry.parent.mkdir(parents=True)
    registry.write_text("""
from nonexistent_simulator import never_import_this
ATOMIC_TASK_DATASETS = OrderedDict(OpenOven=dict(pretrain=dict(mg_path="demo")), CloseOven=dict(horizon=300))
COMPOSITE_TASK_DATASETS = OrderedDict(PrepareCoffee=dict(horizon=600))
PRETRAINING_TASKS = dict(pretrain50=["OpenOven", "PrepareCoffee"])
TARGET_TASKS = dict(atomic_seen=["CloseOven"], composite_seen=["PrepareCoffee"], composite_unseen=[])
LIFELONG_LEARNING_TASKS = dict(lifelong_learning_phase1=list(ATOMIC_TASK_DATASETS.keys()))
MG_DATASET_TASKS = []
TASK_SET_REGISTRY = dict(
    all_tasks=list(ATOMIC_TASK_DATASETS.keys()) + list(COMPOSITE_TASK_DATASETS.keys()),
    all_atomic_tasks=list(ATOMIC_TASK_DATASETS.keys()),
    all_composite_tasks=list(COMPOSITE_TASK_DATASETS.keys()),
    mg_dataset_tasks=MG_DATASET_TASKS,
    target50=TARGET_TASKS["atomic_seen"] + TARGET_TASKS["composite_seen"],
)
raise RuntimeError("Reading this registry must not execute it")
""")
    return tmp_path


def args(dataset, *options):
    return parser_for_cli().parse_args(
        [
            "run",
            "--mode",
            "serial",
            "--dataset-root",
            str(dataset),
            "--robot",
            "PandaOmron",
            "--agent",
            "roboswe",
            *options,
        ]
    )


def test_registry_parsed_without_importing_or_executing(dataset):
    catalog = load_catalog("robocasa-365", dataset)
    assert catalog["subsets"]["all_tasks"] == ["OpenOven", "CloseOven", "PrepareCoffee"]
    assert catalog["subsets"]["mg_dataset_tasks"] == ["OpenOven"]
    assert len(catalog["registry_sha256"]) == 64
    assert select_tasks(catalog, subset="atomic") == ["OpenOven", "CloseOven"]
    assert select_tasks(
        catalog, tasks=["robocasa/OpenOven,PrepareCoffee", "OpenOven"]
    ) == ["OpenOven", "PrepareCoffee"]


def test_default_library_plan_keeps_repeated_tasks_and_fixed_scene(dataset):
    registry = dataset / "robocasa/utils/dataset_registry.py"
    registry.write_text(
        registry.read_text().replace(
            "OrderedDict(OpenOven=",
            "OrderedDict(OpenMicrowave=dict(), CloseMicrowave=dict(), OpenDrawer=dict(), CloseDrawer=dict(), TurnOnSinkFaucet=dict(), TurnOffSinkFaucet=dict(), OpenOven=",
        )
    )
    parsed = parser_for_cli().parse_args(
        ["run", "--robot", "PandaOmron", "--dataset-root", str(dataset), "--dry-run"]
    )
    plan = build_plan(parsed)
    assert plan["mode"] == "library"
    assert plan["selection"]["task_count"] == 16
    assert plan["selection"]["tasks"][:8] == plan["selection"]["tasks"][8:]
    assert plan["environment"]["layout_ids"] == 2
    assert plan["agent"]["name"] == "roboswe"
    parsed.subset = "atomic"
    with pytest.raises(ValueError, match="serial"):
        build_plan(parsed)


def test_unknown_tasks_are_not_silently_dropped(dataset):
    catalog = load_catalog(dataset_root=dataset)
    with pytest.raises(ValueError, match="Unknown RoboCasa task"):
        select_tasks(catalog, tasks=["typo"])
    with pytest.raises(ValueError, match="outside subset"):
        select_tasks(catalog, subset="atomic", tasks=["PrepareCoffee"])


@pytest.mark.parametrize("options,enabled,backend", [
    ([], False, "codex"),
    (["--video-judge"], True, "codex"),
    (["--video-judge", "glimpse"], True, "glimpse"),
])
def test_visual_judge_is_explicit_and_keeps_legacy_default(dataset, options, enabled, backend):
    plan = build_plan(args(dataset, *options))
    assert plan["execution"]["video_judge"] is enabled
    assert plan["execution"]["video_judge_backend"] == backend


def test_task_file_and_plan_provenance(dataset, monkeypatch):
    task_file = dataset / "selected.json"
    task_file.write_text('["OpenOven", "CloseOven"]')
    monkeypatch.setenv("MODEL_KEY", "do-not-copy-this-secret")
    plan = build_plan(
        args(
            dataset,
            "--task-file",
            str(task_file),
            "--seeds",
            "10",
            "20",
            "--episodes",
            "2",
            "--api-key-env",
            "MODEL_KEY",
            "--backend",
            "modal",
        )
    )
    assert plan["episode_seeds"] == [10, 11, 20, 21]
    assert plan["planned_trials"] == 8
    assert plan["execution"]["modal_environment"] == "FDR"
    assert plan["selection"]["task_file_sha256"]
    assert "do-not-copy-this-secret" not in json.dumps(plan)
    assert plan["execution_implemented"] is False


@pytest.mark.parametrize(
    "options",
    [
        ["--cpu", "nan"],
        ["--cpu", "inf"],
        ["--max-steps", "0"],
        ["--concurrency", "-1"],
        ["--seeds", "-1"],
        ["--seeds", "0", "0"],
        ["--seeds", "0", "1", "--episodes", "2"],
        ["--api-base-url", "https://user:password@example.com"],
        ["--api-key-env", "secret=value"],
    ],
)
def test_invalid_settings_rejected(dataset, options):
    with pytest.raises(ValueError):
        build_plan(args(dataset, *options))


def test_run_without_immutable_images_does_not_create_a_job(dataset, capsys):
    output = dataset / "jobs"
    code = main(
        [
            "run",
            "--mode",
            "serial",
            "--dataset-root",
            str(dataset),
            "--robot",
            "PandaOmron",
            "--agent",
            "roboswe",
            "--jobs-dir",
            str(output),
        ]
    )
    assert code == 1
    assert "immutable Docker image" in capsys.readouterr().err
    assert not output.exists()


def test_dry_run_writes_explicit_plan_but_no_jobs(dataset, capsys):
    output = dataset / "jobs"
    plan_file = dataset / "plan.json"
    command = [
        "run",
        "--mode",
        "serial",
        "--dataset-root",
        str(dataset),
        "--robot",
        "PandaOmron",
        "--agent",
        "roboswe",
        "--dry-run",
        "--plan-out",
        str(plan_file),
        "--jobs-dir",
        str(output),
    ]
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "planned"
    assert not output.exists()
    assert main(command) == 1  # Do not silently replace a saved plan.


def test_session_command_forwards_arguments(monkeypatch):
    from robot_session import manage

    seen = []
    monkeypatch.setattr(manage, "main", lambda argv: seen.extend(argv))
    main(["session", "status", "example.json"])
    assert seen == ["status", "example.json"]


def test_reserved_commands_return_nonzero():
    assert main(["evaluate", "job-id"]) == 1
    assert main(["resume", "job-id"]) == 2


def test_jobs_subcommand_and_legacy_alias(monkeypatch):
    from roboswe import cli

    calls = []
    monkeypatch.setattr(cli, "serve", lambda *a: calls.append(a))
    assert (
        cli.main(["jobs", "--jobs-dir", "custom", "--port", "3033", "--no-browser"])
        == 0
    )
    assert calls[-1] == ("custom", 3033, False)
    assert cli.main(["--jobs", "--no-browser"]) == 0
    assert calls[-1] == ("jobs", 0, False)
