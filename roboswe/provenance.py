"""Save coordinator Python source as data at launch; never include credentials."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def snapshot_harness(directory):
    root = Path(__file__).resolve().parents[1]
    files = []
    with ZipFile(
        directory / "coordinator-source.zip", "x", compression=ZIP_DEFLATED
    ) as archive:
        for package in ("roboswe", "robot_session"):
            for path in sorted((root / package).rglob("*.py")):
                if path.is_symlink():
                    raise ValueError("Coordinator source must not contain symlinks")
                relative = path.relative_to(root).as_posix()
                content = path.read_bytes()
                files.append(
                    {"path": relative, "sha256": hashlib.sha256(content).hexdigest()}
                )
                archive.writestr(relative, content)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    manifest = {
        "python": platform.python_version(),
        "git_head": commit.stdout.strip() if commit.returncode == 0 else None,
        "source_files": files,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "note": "Exact Python files present at launch; Git HEAD alone does not describe uncommitted work. Simulator and submitted-code images are separately pinned.",
    }
    (directory / "coordinator.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
