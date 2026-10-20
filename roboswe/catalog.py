"""Read RoboCasa's local task registry without importing or executing it."""

import ast
import hashlib
import subprocess
from pathlib import Path

DATASETS = {
    "robocasa365": {
        "name": "RoboCasa365",
        "repository": "https://github.com/robocasa/robocasa",
        "registry": "robocasa/utils/dataset_registry.py",
    }
}
ALIASES = {"robocasa-365": "robocasa365", "robocasa": "robocasa365"}
SUBSETS = {
    "all": "all_tasks",
    "atomic": "all_atomic_tasks",
    "composite": "all_composite_tasks",
}


def literal(node, names):
    """Only the declarative expression forms used by the upstream registry."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(n, names) for n in node.elts]
    if isinstance(node, ast.Dict) and all(k is not None for k in node.keys):
        return {
            literal(k, names): literal(v, names)
            for k, v in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.Subscript):
        return literal(node.value, names)[literal(node.slice, names)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = literal(node.left, names), literal(node.right, names)
        if isinstance(left, list) and isinstance(right, list):
            return left + right
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if (
                node.func.id in {"dict", "OrderedDict"}
                and not node.args
                and all(k.arg for k in node.keywords)
            ):
                return {k.arg: literal(k.value, names) for k in node.keywords}
            if node.func.id == "list" and len(node.args) == 1 and not node.keywords:
                return list(literal(node.args[0], names))
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "keys"
            and not node.args
            and not node.keywords
        ):
            value = literal(node.func.value, names)
            if isinstance(value, dict):
                return list(value)
    raise ValueError(
        f"Unsupported registry expression on line {getattr(node, 'lineno', '?')}; no code was executed"
    )


def load_catalog(dataset="robocasa365", dataset_root="upstream/robocasa"):
    dataset = ALIASES.get(dataset, dataset)
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset!r}; available: {', '.join(DATASETS)}"
        )
    root = Path(dataset_root).expanduser().resolve()
    path = root / DATASETS[dataset]["registry"]
    if not path.is_file():
        raise ValueError(
            f"No RoboCasa task registry at {path}. Set --dataset-root to the official RoboCasa checkout; nothing is downloaded automatically."
        )
    raw = path.read_bytes()
    tree = ast.parse(raw, filename=str(path))
    wanted = {
        "ATOMIC_TASK_DATASETS",
        "COMPOSITE_TASK_DATASETS",
        "PRETRAINING_TASKS",
        "TARGET_TASKS",
        "LIFELONG_LEARNING_TASKS",
        "TASK_SET_REGISTRY",
    }
    names = {}
    for node in tree.body:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
        ):
            continue
        key = node.targets[0].id
        if key == "MG_DATASET_TASKS":
            merged = names.get("ATOMIC_TASK_DATASETS", {}) | names.get(
                "COMPOSITE_TASK_DATASETS", {}
            )
            names[key] = [
                task
                for task, data in merged.items()
                if data.get("pretrain", {}).get("mg_path") is not None
            ]
        elif key in wanted:
            names[key] = literal(node.value, names)
    if not all(
        k in names
        for k in (
            "ATOMIC_TASK_DATASETS",
            "COMPOSITE_TASK_DATASETS",
            "TASK_SET_REGISTRY",
        )
    ):
        raise ValueError("RoboCasa registry is missing task declarations")
    groups = names["TASK_SET_REGISTRY"]
    for subset, tasks in groups.items():
        if not isinstance(tasks, list) or not all(isinstance(t, str) for t in tasks):
            raise ValueError(f"Malformed task subset: {subset}")
    # Include named official subsets even if a task has no demonstration metadata.
    all_tasks = list(dict.fromkeys(t for tasks in groups.values() for t in tasks))
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    except OSError:
        commit = None
    return {
        "dataset": dataset,
        "root": str(root),
        "registry": str(path),
        "registry_sha256": hashlib.sha256(raw).hexdigest(),
        "checkout_commit": commit,
        "subsets": groups,
        "tasks": all_tasks,
        "metadata": names["ATOMIC_TASK_DATASETS"] | names["COMPOSITE_TASK_DATASETS"],
    }


def select_tasks(catalog, *, subset=None, tasks=None, task_file=None):
    subset = SUBSETS.get(subset, subset)
    if subset is not None and subset not in catalog["subsets"]:
        raise ValueError(
            f"Unknown subset {subset!r}; available: {', '.join(catalog['subsets'])}"
        )
    explicit = [
        t.strip().removeprefix("robocasa/")
        for arg in tasks or []
        for t in arg.split(",")
        if t.strip()
    ]
    if task_file:
        import json

        content = Path(task_file).read_text()
        entries = (
            json.loads(content)
            if Path(task_file).suffix == ".json"
            else [
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        )
        if not isinstance(entries, list) or not all(
            isinstance(t, str) and t.strip() for t in entries
        ):
            raise ValueError(
                "Task file must be a JSON list of task names or one name per text line"
            )
        explicit.extend(t.strip().removeprefix("robocasa/") for t in entries)
    invalid = set(explicit) - set(catalog["tasks"])
    if invalid:
        raise ValueError(f"Unknown RoboCasa task(s): {', '.join(sorted(invalid))}")
    if explicit:
        selected = list(dict.fromkeys(explicit))
        if subset:
            outside = set(selected) - set(catalog["subsets"][subset])
            if outside:
                raise ValueError(
                    f"Task(s) outside subset {subset}: {', '.join(sorted(outside))}"
                )
    else:
        selected = list(dict.fromkeys(catalog["subsets"][subset or "all_tasks"]))
    if not selected:
        raise ValueError("Task selection is empty")
    return selected
