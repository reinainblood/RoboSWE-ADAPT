"""Explicit, pinned upstream provisioning for a fresh RoboSWE source checkout."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPOSITORIES = {
    "robosuite": ("https://github.com/ARISE-Initiative/robosuite.git", "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"),
    "robocasa": ("https://github.com/robocasa/robocasa.git", "456174f62b89b8fca99eaaf33949c29fec9cfc2a"),
}
ASSET_FOLDERS = ("textures", "generative_textures", "fixtures", "objects/objaverse", "objects/aigen_objs", "objects/lightwheel")


def validate_checkout(path, url, revision):
    """Read only: never checkout, fetch, reset, or clean an existing repository."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    if git("rev-parse", "HEAD") != revision:
        raise ValueError(f"Pinned revision mismatch: {path}")
    if git("remote", "get-url", "origin").removesuffix(".git") != url.removesuffix(".git"):
        raise ValueError(f"Official origin mismatch: {path}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Tracked upstream changes: {path}")


def assets_present(root):
    base = root / "upstream/robocasa/robocasa/models/assets"
    return {name: any(p.is_file() for p in (base / name).rglob("*")) for name in ASSET_FOLDERS}


def plan(source_root=".", python="python3.11"):
    root = Path(source_root).expanduser().resolve()
    if not (root / "robot_session/requirements.txt").is_file():
        raise ValueError("--source-root must be a RoboSWE source checkout containing robot_session/requirements.txt")
    venv = root / ".roboswe-prepare-venv"
    interpreter = str(venv / "bin/python")
    commands = []
    existing = []
    for name, (url, revision) in REPOSITORIES.items():
        path = root / "upstream" / name
        if path.exists():
            validate_checkout(path, url, revision)
            existing.append(name)
        else:
            commands.extend([["git", "clone", "--no-checkout", url, str(path)], ["git", "-C", str(path), "checkout", "--detach", revision]])
    if existing:
        # Asset downloader extracts in-place. Never run it on a pre-existing checkout.
        if len(existing) != len(REPOSITORIES) or not all(assets_present(root).values()):
            raise ValueError("Existing upstream checkout is incomplete. Preserve it; use a fresh RoboSWE checkout for prepare --run. Existing upstream trees are never modified.")
    else:
        if venv.exists():
            raise ValueError(f"Refusing existing provisioning environment: {venv}")
        commands.extend([
            [python, "-m", "venv", str(venv)],
            [interpreter, "-m", "pip", "install", "-r", str(root / "robot_session/requirements.txt")],
            [interpreter, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(root / "upstream/robosuite"), "-e", str(root / "upstream/robocasa")],
            [interpreter, "-m", "robocasa.scripts.setup_macros"],
            [interpreter, "-m", "robocasa.scripts.download_kitchen_assets", "--type", "all"],
        ])
    return {"schema": "roboswe-provision-v1", "source_root": str(root), "repositories": {name: {"url": url, "revision": revision} for name, (url, revision) in REPOSITORIES.items()}, "commands": commands, "existing_read_only": existing, "asset_download": "Upstream interactive confirmation; six asset groups, with additional space required for extraction and images", "asset_integrity": "Folder presence only; upstream does not supply checksums in this download script"}


def prepare(source_root=".", *, python="python3.11", run=False, output=None):
    receipt = plan(source_root, python)
    receipt["status"] = "planned"
    if not run:
        return receipt
    if not output:
        raise ValueError("--run requires --out for a durable receipt")
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the receipt before any mutation; never overwrite an earlier attempt.
    with destination.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    root = Path(receipt["source_root"])
    receipt["started_at"] = datetime.now(timezone.utc).isoformat()
    receipt["completed_commands"] = 0
    try:
        (root / "upstream").mkdir(exist_ok=True)
        env = dict(os.environ, PYNPUT_BACKEND="dummy")
        for command in receipt["commands"]:
            subprocess.run(command, cwd=root, env=env, check=True)
            receipt["completed_commands"] += 1
        for name, (url, revision) in REPOSITORIES.items():
            validate_checkout(root / "upstream" / name, url, revision)
        receipt["asset_folders"] = assets_present(root)
        if not all(receipt["asset_folders"].values()):
            raise ValueError("Asset folders missing or empty: download was declined or failed. No image readiness claim.")
        receipt["status"] = "prepared"
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error_type"] = type(exc).__name__
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        destination.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
