"""Host storage checks for durable benchmark evidence."""

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def clone_or_copy(source, destination, *, require_clone=False, preserve_destination_metadata=False):
    """Atomic independent copy, using APFS copy-on-write when available."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    with tempfile.TemporaryDirectory(prefix=".artifact-copy-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "copy"
        cloned = False
        if sys.platform == "darwin":
            result = subprocess.run(
                ["/bin/cp", "-c", str(source), str(staged)],
                capture_output=True, timeout=30, check=False,
            )
            cloned = result.returncode == 0
        if not cloned:
            if require_clone:
                raise RuntimeError("Copy-on-write cloning unavailable; destination left unchanged")
            staged.unlink(missing_ok=True)
            shutil.copyfile(source, staged)
        if preserve_destination_metadata and destination.exists():
            shutil.copystat(destination, staged, follow_symlinks=False)
        os.replace(staged, destination)
    return {"method": "copy_on_write" if cloned else "copy", "bytes": destination.stat().st_size}


def require_disk_headroom(path, minimum_gib=1):
    if not math.isfinite(minimum_gib) or minimum_gib <= 0:
        raise ValueError("Disk headroom must be positive and finite")
    checks = []
    for requested in (Path(path).resolve(), Path(tempfile.gettempdir()).resolve()):
        existing = requested
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free
        if free < minimum_gib * 2**30:
            raise RuntimeError(
                f"Insufficient host disk headroom at {requested}: "
                f"{free / 2**30:.2f} GiB free; {minimum_gib:g} GiB required. "
                "Free space before continuing; evidence must remain writable."
            )
        checks.append({"path": str(requested), "free_bytes": free})
    return checks
