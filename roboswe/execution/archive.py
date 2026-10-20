"""Export authored Python as data. Run with python -I, never import the workspace."""

import hashlib
import json
import os
import stat
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path("/workspace")
    manifest = []
    total = 0
    result = BytesIO()
    with ZipFile(result, "w", compression=ZIP_DEFLATED) as archive:
        for parent, directories, files in os.walk(root, followlinks=False):
            directories[:] = sorted(
                d for d in directories if not (Path(parent) / d).is_symlink()
            )
            for name in sorted(files):
                path = Path(parent) / name
                if (path.suffix != ".py" and path != root / "NOTES.md") or not stat.S_ISREG(path.lstat().st_mode):
                    continue
                size = path.stat().st_size
                total += size
                if total > 16 * 1024 * 1024 or len(manifest) >= 2048:
                    raise ValueError("Python source archive exceeds export budget")
                data = path.read_bytes()
                relative = path.relative_to(root).as_posix()
                archive.writestr(relative, data)
                manifest.append(
                    {
                        "path": relative,
                        "bytes": size,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
        archive.writestr(
            "library-manifest.json",
            json.dumps({"files": manifest, "total_bytes": total}, indent=2),
        )
    sys.stdout.buffer.write(result.getvalue())


if __name__ == "__main__":
    main()
