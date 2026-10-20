"""Export visual state without advancing physics; no host simulator required."""

import base64
import gzip
import json
from pathlib import Path

ASSETS = Path(__file__).with_name("viewer")


def scene_metadata(data):
    scene = json.loads(gzip.decompress(data))
    for key in ("step", "session_id", "task", "geoms", "meshes", "textures"):
        if key not in scene:
            raise ValueError(f"Scene is missing {key}")
    return {key: scene[key] for key in ("step", "session_id", "task")}


def capture_scene(handle, destination, timeout=120):
    result = handle.execute(
        "python",
        "-",
        timeout=timeout,
        stdin=(ASSETS / "export_scene.py").read_bytes(),
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    metadata = scene_metadata(result.stdout)
    destination = Path(destination)
    destination.write_bytes(result.stdout)
    return metadata


def inline_scene_script():
    """Bundle the pinned local Three.js modules for an offline notebook/HTML."""

    def module_url(source):
        return (
            "data:text/javascript;base64," + base64.b64encode(source.encode()).decode()
        )

    core = module_url((ASSETS / "vendor/three.core.min.js").read_text())
    three = module_url(
        (ASSETS / "vendor/three.module.js")
        .read_text()
        .replace("./three.core.min.js", core)
    )
    orbit = module_url(
        (ASSETS / "vendor/OrbitControls.js")
        .read_text()
        .replace("./three.module.js", three)
    )
    return (
        (ASSETS / "scene.js")
        .read_text()
        .replace("./vendor/OrbitControls.js", orbit)
        .replace("./vendor/three.module.js", three)
    )


def standalone_html(data):
    encoded = base64.b64encode(data).decode()
    return (
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>RoboSWE · 3D snapshot</title><style>body{margin:0;background:#14202b}"
        '.robot-view .stage{height:calc(100vh - 115px)!important}</style><div id="scene"></div>'
        '<script type="module">'
        + inline_scene_script()
        + '\nmount(document.getElementById("scene"),"'
        + encoded
        + '");</script>'
    )
