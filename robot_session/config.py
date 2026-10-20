"""Host-safe session configuration (no simulator dependencies)."""

import json


def normalize_config(config):
    if not isinstance(config, dict):
        raise TypeError("Session config must be an object")
    unknown = set(config) - {"robot", "task", "split", "seed", "env_kwargs"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if not isinstance(config.get("robot"), str) or not config["robot"].strip():
        raise ValueError(
            'An explicit robot name is required, e.g. "robot": "PandaOmron"'
        )
    config = {
        "task": "robocasa/PickPlaceCounterToCabinet",
        "split": "pretrain",
        "seed": 0,
        "env_kwargs": {},
        **config,
    }
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if config["split"] not in (None, "pretrain", "target", "all"):
        raise ValueError("Invalid split")
    if not isinstance(config["task"], str) or not config["task"].startswith(
        "robocasa/"
    ):
        raise ValueError("task must be a robocasa/<TaskName> ID")
    if not isinstance(config["env_kwargs"], dict):
        raise TypeError("env_kwargs must be an object")
    reserved = {
        "robot",
        "robots",
        "seed",
        "split",
        "env_name",
        "controller_configs",
        "render_onscreen",
        "has_renderer",
        "has_offscreen_renderer",
        "use_camera_obs",
    }
    if reserved & config["env_kwargs"].keys():
        raise ValueError(
            f"Reserved env_kwargs: {sorted(reserved & config['env_kwargs'].keys())}; select robot at the top level"
        )
    return json.loads(json.dumps(config, allow_nan=False))
