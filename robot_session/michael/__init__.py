"""Opt-in OpenArm v2 / Wuji Hand 2 robot integration and commissioning tools."""


def register():
    """Register Michael's models and controller without changing upstream files."""
    from . import (
        environments,  # noqa: F401
        models,  # noqa: F401
    )


ROBOT_NAME = "MichaelOpenArmWuji"
