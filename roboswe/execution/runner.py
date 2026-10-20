"""Container entrypoint. Never import this module on the coordinator."""

import json
import os
import runpy
import sys
from pathlib import Path

LIMIT = 256 * 1024


def relative_python(value, *, operation=None):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or (path.suffix != ".py" and value != "NOTES.md"):
        raise ValueError("Use a relative .py path or NOTES.md without '..'")
    if operation == "execute" and path.suffix != ".py":
        raise ValueError("NOTES.md cannot be executed")
    resolved = (Path("/workspace") / path).resolve()
    if not resolved.is_relative_to(Path("/workspace")):
        raise ValueError("Path escapes workspace")
    if any(
        p.is_symlink()
        for p in [Path("/workspace") / path, *(Path("/workspace") / path).parents]
    ):
        raise ValueError("Symlink paths are not allowed")
    return resolved


def main():
    line = sys.stdin.buffer.readline(LIMIT * 8)
    request = json.loads(line)
    operation = request["operation"]
    path = relative_python(request["path"], operation=operation)
    if operation == "write":
        content = request["content"].encode()
        if len(content) > LIMIT:
            raise ValueError("File is too large")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(content)
        print(json.dumps({"result": {"bytes": len(content)}}), flush=True)
    elif operation == "read":
        with path.open("rb") as source:
            data = source.read(LIMIT + 1)
        if len(data) > LIMIT:
            raise ValueError("File is too large")
        print(json.dumps({"result": {"content": data.decode()}}), flush=True)
    elif operation == "execute":
        # stdout is the bounded RPC pipe. Ordinary Python print goes to the log.
        sys.stdout = sys.stderr
        sys.path.insert(0, "/opt/roboswe")
        sys.path.insert(0, "/workspace")
        runpy.run_path(str(path), run_name="__main__")
    else:
        raise ValueError("Unknown operation")


if __name__ == "__main__":
    main()
