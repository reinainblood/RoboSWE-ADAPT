"""The entire agent-facing robot API. Contains no movement helpers."""

import json
import os


def _request(operation, **arguments):
    data = (
        json.dumps({"operation": operation, **arguments}, allow_nan=False).encode()
        + b"\n"
    )
    os.write(1, data)
    response = json.loads(input())
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["result"]


def observe():
    """JSON-compatible sensors, native action schema, and current control step."""
    return _request("observe")


def step(action):
    """One native robosuite action vector; returns post-step sensors."""
    return _request("step", action=[float(x) for x in action])
