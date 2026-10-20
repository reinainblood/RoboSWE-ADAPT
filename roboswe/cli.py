"""Public RoboSWE CLI: working inspection/session tools, explicit campaign stubs."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

from .catalog import DATASETS, load_catalog, select_tasks
from .jobs import capture_job, import_scene, run_job
from .plan import build_plan, write_plan
from .server import serve


def dataset_options(parser):
    parser.add_argument(
        "--dataset",
        default="robocasa365",
        help="Dataset ID (robocasa365 / robocasa-365)",
    )
    parser.add_argument(
        "--dataset-root",
        default="upstream/robocasa",
        help="Path to official RoboCasa source checkout, not a demo HDF5 directory",
    )
    parser.add_argument(
        "--subset", help="Official TASK_SET_REGISTRY key, or all / atomic / composite"
    )
    parser.add_argument(
        "--tasks",
        action="append",
        help="Task name(s), comma-separated; may be repeated",
    )
    parser.add_argument(
        "--task-file", help="JSON list or text file with one task name per line"
    )


def parser_for_cli():
    parser = argparse.ArgumentParser(
        description="RoboSWE-365 command line: plan runs, execute session code, inspect jobs"
    )
    parser.add_argument(
        "--version", action="version", version="roboswe " + version("roboswe")
    )
    parser.add_argument(
        "--jobs", action="store_true", help="Open the local HTML jobs viewer"
    )
    parser.add_argument("--jobs-dir", default="jobs")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    commands = parser.add_subparsers(dest="command")
    jobs = commands.add_parser("jobs", help="Open the local HTML jobs and video viewer")
    jobs.add_argument("--jobs-dir", default=argparse.SUPPRESS)
    jobs.add_argument("--port", type=int, default=argparse.SUPPRESS)
    jobs.add_argument("--no-browser", action="store_true", default=argparse.SUPPRESS)
    prepare = commands.add_parser("prepare", help="Plan pinned upstream checkout and asset provisioning; --run executes")
    prepare.add_argument("--source-root", default=".")
    prepare.add_argument("--python", default="python3.11")
    prepare.add_argument("--run", action="store_true")
    prepare.add_argument("--out", help="New provisioning receipt JSON; required with --run")
    images = commands.add_parser("build-images", help="Build trusted simulator and isolated authored-code images")
    images.add_argument("-e", "--environment", choices=["docker", "modal"], default="docker")
    images.add_argument("--source-root", default=".")
    images.add_argument("--modal-profile")
    images.add_argument("--modal-environment", default="FDR")
    images.add_argument("--out", required=True, help="New JSON image receipt path")
    resume = commands.add_parser("resume-adaptive", help="Inspect a certified original-trial boundary; --run continues it")
    resume.add_argument("parent")
    resume.add_argument("--certificate", required=True)
    resume.add_argument("--run", action="store_true", help="Explicitly continue remaining original trials with paid inference")
    admit = commands.add_parser("admit", help="Qualify dataset scenes with two independent resets; no model calls")
    admit.add_argument("input")
    admit.add_argument("--config", required=True)
    admit.add_argument("--images")
    admit.add_argument("-e", "--environment", choices=["docker", "modal"])
    admit.add_argument("--out", required=True)
    admit.add_argument("--dry-run", action="store_true", help="Read and validate local files without allocating simulators")
    commands.add_parser("datasets", help="List supported dataset references")
    tasks = commands.add_parser(
        "tasks",
        help="Inspect actual task names and subsets without loading the simulator",
    )
    dataset_options(tasks)
    tasks.add_argument("--json", action="store_true")
    tasks.add_argument("--list-subsets", action="store_true")

    run = commands.add_parser(
        "run",
        help="Run or plan a persistent library sequence or independent dataset episodes",
    )
    dataset_options(run)
    run.add_argument("input", nargs="?", help="Adaptive dataset JSON or directory containing dataset.json")
    run.add_argument("--images", help="Existing build-images JSON receipt for an adaptive run")
    run.add_argument("--config", help="Adaptive run JSON configuration (required with positional dataset)")
    run.add_argument(
        "--mode",
        choices=["library", "serial"],
        default="library",
        help="Library retains workspace/context/kitchen across a set; serial uses independent trials",
    )
    run.add_argument("--library-set", default="kitchen-library-v1")
    run.add_argument(
        "--learning-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Library mode: run matched-state cold comparisons after the retained-context sequence",
    )
    run.add_argument(
        "--lifetime",
        type=int,
        default=14400,
        help="Container lifetime in seconds (default four hours)",
    )
    run.add_argument("--robot", help="Explicit robosuite robot name, e.g. PandaOmron")
    run.add_argument(
        "--agent",
        choices=["roboswe"],
        default="roboswe",
        help="The reference RoboSWE harness; arbitrary participant harnesses are not supported",
    )
    run.add_argument(
        "--model", help="Codex model ID; omit to use the configured local default"
    )
    run.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="medium",
        help="Explicit model reasoning effort, recorded in campaign provenance",
    )
    run.add_argument("--api-base-url")
    run.add_argument(
        "--api-key-env",
        help="Environment variable NAME only; secrets are never resolved into plans",
    )
    run.add_argument("--backend", "-e", "--environment", choices=["docker", "modal"], default=None)
    run.add_argument("--image", help="Existing immutable container image reference")
    run.add_argument(
        "--code-image", help="Immutable Docker image ID for the isolated code sandbox"
    )
    run.add_argument(
        "--video-judge",
        nargs="?",
        choices=["codex", "glimpse"],
        const="codex",
        help="Optional blind audit: Codex frames (default) or experimental Glimpse video via OpenRouter",
    )
    run.add_argument(
        "--split",
        choices=["pretrain", "target", "all"],
        default="all",
        help="Environment scene/object split, independent of task subset",
    )
    run.add_argument("--seeds", type=int, nargs="+", default=[0])
    run.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Consecutive episode seeds beginning at each supplied seed",
    )
    run.add_argument(
        "--timeout", type=int, default=720, help="Planned wall-clock seconds per task"
    )
    run.add_argument(
        "--max-steps",
        type=int,
        default=6000,
        help="Planned maximum simulator control steps per trial",
    )
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument(
        "--max-attempt-steps",
        type=int,
        default=50,
        help="Maximum control ticks in one code execution before another model feedback turn",
    )
    run.add_argument(
        "--attempt-timeout",
        type=int,
        default=60,
        help="Maximum wall-clock seconds per submitted code execution",
    )
    run.add_argument("--cpu", type=float, default=4)
    run.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=10,
        help="Minimum free GiB on evidence and host temporary volumes before launch (default: 10)",
    )
    run.add_argument(
        "--memory-mb",
        type=int,
        default=6144,
        help="Simulator memory limit; code/workspace containers need additional memory",
    )
    run.add_argument("--camera-width", type=int, default=640)
    run.add_argument("--camera-height", type=int, default=480)
    run.add_argument("--camera-depths", action="store_true")
    run.add_argument("--jobs-dir", default=argparse.SUPPRESS)
    run.add_argument("--name")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved plan without allocating resources",
    )
    run.add_argument(
        "--plan-out",
        help="Save a new plan JSON; requires --dry-run and refuses overwrites",
    )

    for name in ("execute", "capture", "import-scene"):
        command = commands.add_parser(
            name,
            help={
                "execute": "Run one Python file in an existing session and save job output",
                "capture": "Save a 3D snapshot of an existing session",
                "import-scene": "Import a saved 3D snapshot without a container",
            }[name],
        )
        command.add_argument(
            "input",
            help="Session receipt"
            if name != "import-scene"
            else "Saved .scene.json.gz",
        )
        command.add_argument("--name")
        command.add_argument("--jobs-dir", default=argparse.SUPPRESS)
        if name == "execute":
            command.add_argument("source")
            command.add_argument("--timeout", type=int, default=600)
            command.add_argument(
                "--no-video",
                action="store_true",
                help="Explicitly execute without video (e.g. a legacy image)",
            )
            command.add_argument("--video-width", type=int, default=640)
            command.add_argument("--video-height", type=int, default=480)
    commands.add_parser(
        "session",
        help="Existing build/start/status/snapshot/checkpoint/resume/replay/stop tools; use roboswe session --help",
        add_help=False,
    )
    evaluate = commands.add_parser(
        "evaluate",
        help="Independently re-execute a completed campaign's saved submissions",
    )
    evaluate.add_argument("job", help="Campaign job ID or directory")
    evaluate.add_argument("--jobs-dir", default=argparse.SUPPRESS)
    evaluate.add_argument(
        "--memory-mb",
        type=int,
        help="Explicit verifier memory allocation; original plan and recorded timings remain unchanged",
    )
    for name in ("resume",):
        stub = commands.add_parser(
            name, help=f"Future campaign {name} operation (not implemented)"
        )
        stub.add_argument("job", help="Campaign job ID or path (reserved interface)")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "session":
        from robot_session.manage import main as session_main

        return session_main(argv[1:])
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    if args.jobs or args.command == "jobs":
        if args.jobs and args.command:
            parser.error(
                "--jobs opens the viewer and cannot be combined with another command"
            )
        serve(args.jobs_dir, args.port, not args.no_browser)
        return 0
    if not args.command:
        parser.print_help()
        return 0
    try:
        if args.command == "prepare":
            from .provision import prepare
            print(json.dumps(prepare(args.source_root, python=args.python, run=args.run, output=args.out), indent=2))
            return 0
        if args.command == "build-images":
            from .adaptive_images import build
            print(build(args.source_root, backend=args.environment, profile=args.modal_profile,
                        environment=args.modal_environment, output=args.out))
            return 0
        if args.command == "resume-adaptive":
            from pairing.adapt_resume import validate, resume
            result = resume(args.parent, args.certificate) if args.run else validate(args.parent, args.certificate)
            print(json.dumps(str(result) if args.run else result, indent=2))
            return 0
        if args.command == "admit":
            from .admission import prepare, run as admit_run
            plan = prepare(args.input, args.config, images=args.images,
                           backend=args.environment, output=args.out)
            print(json.dumps(plan if args.dry_run else admit_run(plan), indent=2))
            return 0
        if args.command == "datasets":
            print(json.dumps(DATASETS, indent=2))
            return 0
        if args.command == "tasks":
            catalog = load_catalog(args.dataset, args.dataset_root)
            if args.list_subsets:
                result = {
                    name: len(set(tasks)) for name, tasks in catalog["subsets"].items()
                }
            else:
                result = select_tasks(
                    catalog,
                    subset=args.subset,
                    tasks=args.tasks,
                    task_file=args.task_file,
                )
            if args.json or isinstance(result, dict):
                print(json.dumps(result, indent=2))
            else:
                print("\n".join("robocasa/" + task for task in result))
            return 0
        if args.command == "run":
            if args.plan_out and not args.dry_run:
                raise ValueError("--plan-out requires --dry-run")
            if args.input or args.config or args.images:
                if not args.input or not args.config:
                    raise ValueError("Adaptive runs require both dataset path and --config")
                allowed = {"--images", "--config", "-e", "--environment", "--backend", "--dry-run", "--plan-out", "--jobs-dir"}
                supplied = {token.split("=", 1)[0] for token in argv[argv.index("run")+1:] if token.startswith("-")}
                if supplied - allowed:
                    raise ValueError("Adaptive run settings belong in --config; unsupported flags: " + ", ".join(sorted(supplied-allowed)))
                from .adaptive import build_plan as adaptive_plan, run as adaptive_run
                plan = adaptive_plan(args.input, args.config, backend=args.backend, jobs_dir=args.jobs_dir, images_path=args.images)
                if args.dry_run:
                    if args.plan_out: write_plan(args.plan_out, plan)
                    print(json.dumps(plan, indent=2))
                else:
                    print(json.dumps(adaptive_run(plan), indent=2))
                return 0
            args.backend = args.backend or "docker"
            plan = build_plan(args)
            if not args.dry_run:
                from .campaign import run_campaign

                print(run_campaign(plan))
                return 0
            if args.plan_out:
                write_plan(args.plan_out, plan)
            print(json.dumps(plan, indent=2))
            return 0
        if args.command == "evaluate":
            from .evaluate import evaluate_campaign

            directory = evaluate_campaign(
                args.job, args.jobs_dir, memory_mb=args.memory_mb
            )
            print(directory)
            return json.loads((Path(directory) / "job.json").read_text())["exit_code"]
        if args.command == "resume":
            print(
                f"roboswe {args.command}: benchmark campaign {args.command} is not implemented yet; no job or container was modified. Session checkpoint restore is available via roboswe session resume --help.",
                file=sys.stderr,
            )
            return 2
        if args.command == "import-scene":
            print(import_scene(args.input, jobs_dir=args.jobs_dir, name=args.name))
            return 0
        from robot_session.backends import SessionHandle

        handle = SessionHandle.load(args.input)
        if args.command == "capture":
            directory, record = capture_job(
                handle, jobs_dir=args.jobs_dir, name=args.name or "Session snapshot"
            )
            print(directory)
            if record["artifact_errors"]:
                print(json.dumps(record["artifact_errors"]), file=sys.stderr)
            return 0 if record["status"] == "captured" else 1
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        directory, result = run_job(
            handle,
            args.source,
            jobs_dir=args.jobs_dir,
            timeout=args.timeout,
            name=args.name,
            record_video=not args.no_video,
            video_width=args.video_width,
            video_height=args.video_height,
        )
        sys.stdout.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
        print(f"\nJob output: {directory}", file=sys.stderr)
        record = json.loads((directory / "job.json").read_text())
        for error in record["artifact_errors"]:
            print(
                f"Snapshot warning ({error['stage']}): {error['error']}",
                file=sys.stderr,
            )
        return result.returncode if result.returncode >= 0 else 1
    except (OSError, ValueError, RuntimeError, SyntaxError, KeyError) as exc:
        print(f"roboswe: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
