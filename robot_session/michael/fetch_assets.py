"""Download manufacturer MJCF dependencies at immutable source revisions."""

import hashlib
import json
import posixpath
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).parent
SOURCES = {
    "openarm": {
        "repository": "enactic/openarm_mujoco",
        "revision": "ee1110561fa64efe041b90c7b274b93825c9bd82",
        "models": ["v2/openarm_bimanual.xml"],
    },
    "wuji": {
        "repository": "wuji-technology/wuji-description",
        "revision": "c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95",
        "models": [
            f"hand2/hand2_beta2/body/mjcf/{s}_with_mount.xml" for s in ("right", "left")
        ],
    },
}


def fetch_assets():
    manifest = {}
    for vendor, source in SOURCES.items():
        dest = ROOT / "vendor" / vendor
        url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/"

        def fetch(path, dest=dest, url=url):
            out = dest / path
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(url + path, timeout=90) as response:
                    payload = response.read()
                out.write_bytes(payload)
            payload = out.read_bytes()
            return path, {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

        paths = set(source["models"] + ["LICENSE", "README.md"])
        for model in source["models"]:
            fetch(model)
            root = ET.parse(dest / model).getroot()
            meshdir = root.find("compiler").get("meshdir", "")
            for mesh in root.findall("asset/mesh"):
                paths.add(
                    posixpath.normpath(
                        posixpath.join(
                            posixpath.dirname(model), meshdir, mesh.get("file")
                        )
                    )
                )
        with ThreadPoolExecutor(max_workers=6) as pool:
            files = dict(pool.map(fetch, sorted(paths)))
        manifest[vendor] = {**source, "files": files}
        print(
            f"{vendor}: {len(files)} files, {sum(x['bytes'] for x in files.values()) / 1e6:.1f} MB",
            flush=True,
        )
    (ROOT / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    fetch_assets()
