"""Optional marimo/anywidget integration; the job viewer itself needs neither."""

import anywidget
import traitlets

from .scene import inline_scene_script, standalone_html  # noqa: F401


class SceneViewer(anywidget.AnyWidget):
    _esm = inline_scene_script()
    scene_data = traitlets.Unicode("").tag(sync=True)
