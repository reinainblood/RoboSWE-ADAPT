"""Validated campaign plans; deliberately no agent loop or launch behavior."""

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from .catalog import load_catalog, select_tasks
from .codex_login import CONTEXT_POLICY
from .library.spec import library_set


def build_plan(args):
    if not args.robot or not args.robot.strip():
        raise ValueError(
            "--robot is required; no robot/controller is selected implicitly"
        )
    if not args.agent or not args.agent.strip():
        raise ValueError("--agent must name the reference roboswe harness")
    for field in (
        "episodes",
        "timeout",
        "max_steps",
        "concurrency",
        "cpu",
        "memory_mb",
        "camera_width",
        "camera_height",
        "lifetime",
        "max_attempt_steps",
        "attempt_timeout",
        "min_free_disk_gb",
    ):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if any(not 0 <= s < 2**32 for s in args.seeds):
        raise ValueError("--seeds must be integers in [0, 2**32)")
    if args.min_free_disk_gb < 1:
        raise ValueError(
            "--min-free-disk-gb must be at least the 1 GiB runtime reserve"
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if max(args.seeds) + args.episodes - 1 >= 2**32:
        raise ValueError("Episode seed range exceeds 2**32 - 1")
    episode_seeds = [s + i for s in args.seeds for i in range(args.episodes)]
    if len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("--seeds and --episodes produce overlapping episode seeds")
    if args.api_base_url:
        url = urlsplit(args.api_base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "--api-base-url must be an HTTP(S) base URL without credentials, query, or fragment; use --api-key-env for authentication"
            )
    if args.api_key_env and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env
    ):
        raise ValueError(
            "--api-key-env takes an environment variable name, not a credential"
        )
    catalog = load_catalog(args.dataset, args.dataset_root)
    library = None
    if args.mode == "library":
        if args.subset or args.tasks or args.task_file:
            raise ValueError(
                "Library mode uses --library-set; use --mode serial for arbitrary dataset/subset selections"
            )
        library = library_set(args.library_set)
        library["template_sha256"] = library.pop("sha256")
        library.update(
            task_timeout_seconds=args.timeout,
            max_control_steps_per_task=args.max_steps,
            container_lifetime_seconds=args.lifetime,
        )
        library["sha256"] = hashlib.sha256(
            json.dumps(library, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if args.robot != library["robot"]:
            raise ValueError(
                f"{library['id']} currently specifies {library['robot']}; other embodiments require a separately validated set"
            )
        tasks = [t["task"] for t in library["tasks"]]
        select_tasks(catalog, tasks=tasks)
        if args.lifetime < args.timeout * len(tasks) + 300:
            raise ValueError(
                "Container lifetime must cover the full library sequence plus at least 300 seconds for setup/artifacts"
            )
    else:
        tasks = select_tasks(
            catalog, subset=args.subset, tasks=args.tasks, task_file=args.task_file
        )
    if args.lifetime > 86400:
        raise ValueError("Container lifetime cannot exceed 86400 seconds")
    plan = {
        "schema_version": 1,
        "kind": "campaign_plan",
        "status": "planned",
        "created_at": datetime.now(UTC).isoformat(),
        "name": args.name,
        "mode": args.mode,
        "library": library,
        "dataset": {
            key: catalog[key]
            for key in (
                "dataset",
                "root",
                "registry",
                "registry_sha256",
                "checkout_commit",
            )
        },
        "selection": {
            "subset": args.subset,
            "tasks": ["robocasa/" + t for t in tasks],
            "task_count": len(tasks),
        },
        "episode_seeds": episode_seeds,
        "planned_trials": len(tasks) * len(episode_seeds),
        "environment": {
            "robot": args.robot,
            "split": None if library else args.split,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
            "camera_depths": args.camera_depths,
            "layout_ids": library["layout_id"] if library else None,
            "style_ids": library["style_id"] if library else None,
        },
        "agent": {
            "name": args.agent,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "context_policy": dict(CONTEXT_POLICY),
            "api_base_url": args.api_base_url,
            "api_key_env": args.api_key_env,
        },
        "execution": {
            "backend": args.backend,
            "container_logging": "none",
            "model_feedback_transport": "tool_output_between_task_instructions",
            "image": args.image,
            "code_image": args.code_image,
            "video_judge": bool(args.video_judge),
            "video_judge_backend": args.video_judge or "codex",
            "learning_controls": args.learning_controls and library is not None,
            "concurrency": args.concurrency,
            "timeout_seconds": args.timeout,
            "max_control_steps": args.max_steps,
            "max_attempt_steps": args.max_attempt_steps,
            "attempt_timeout_seconds": args.attempt_timeout,
            "cpu": args.cpu,
            "memory_mb": args.memory_mb,
            "min_free_disk_gb": args.min_free_disk_gb,
            "lifetime_seconds": args.lifetime,
            "modal_environment": "FDR" if args.backend == "modal" else None,
        },
        "jobs_dir": str(Path(args.jobs_dir).resolve()),
        "execution_implemented": args.backend == "docker" and args.concurrency == 1,
        "validation_status": "experimental; full task-set calibration pending",
        "unimplemented": [
            "Modal campaign execution",
            "parallel campaign scheduling",
            "separate context-only versus library-only ablations",
            "campaign resume",
        ],
    }
    if args.task_file:
        path = Path(args.task_file).resolve()
        plan["selection"].update(
            task_file=str(path),
            task_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return plan


def write_plan(path, plan):
    """Explicit output only; refuse to overwrite an existing experiment plan."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(plan, output, indent=2)
        output.write("\n")
