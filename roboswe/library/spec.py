"""Versioned task sequence. Contains no agent movement implementations."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskGoal:
    task: str
    fixture: str
    predicate: str
    threshold: float
    instruction: str


GOALS = {
    "OpenMicrowave": TaskGoal(
        "OpenMicrowave", "microwave", "open", 0.90, "Open the microwave door."
    ),
    "CloseMicrowave": TaskGoal(
        "CloseMicrowave", "microwave", "close", 0.005, "Close the microwave door."
    ),
    "OpenDrawer": TaskGoal(
        "OpenDrawer",
        "drawer",
        "open",
        0.95,
        "Open the top drawer in the four-drawer stack on the sink island fully.",
    ),
    "CloseDrawer": TaskGoal(
        "CloseDrawer",
        "drawer",
        "close",
        0.05,
        "Close the top drawer in the four-drawer stack on the sink island fully.",
    ),
    "TurnOnSinkFaucet": TaskGoal(
        "TurnOnSinkFaucet", "sink", "water_on", 0.40, "Turn on the sink faucet."
    ),
    "TurnOffSinkFaucet": TaskGoal(
        "TurnOffSinkFaucet", "sink", "water_off", 0.40, "Turn off the sink faucet."
    ),
    "OpenOven": TaskGoal("OpenOven", "oven", "open", 0.90, "Open the oven door."),
    "CloseOven": TaskGoal("CloseOven", "oven", "close", 0.005, "Close the oven door."),
}
ROUND = (
    "OpenMicrowave",
    "OpenDrawer",
    "TurnOnSinkFaucet",
    "OpenOven",
    "CloseMicrowave",
    "CloseDrawer",
    "TurnOffSinkFaucet",
    "CloseOven",
)


def goal_criteria(task, *, ranges=(), settle_steps=10):
    """Published task requirements, without fixture poses or measured state."""
    task = task.removeprefix("robocasa/")
    if task not in GOALS:
        return {
            "rule": "Use the official task instruction; exact state may not be visually decidable",
            "sustained_control_steps": settle_steps,
        }
    goal = GOALS[task]
    result = {
        "predicate": goal.predicate,
        "threshold": goal.threshold,
        "sustained_control_steps": settle_steps,
        "measurement": "handle angle in radians"
        if goal.fixture == "sink"
        else "normalized fixture opening",
        "rule": f"all selected joints have normalized opening >= {goal.threshold}"
        if goal.predicate == "open"
        else f"all selected joints have normalized opening <= {goal.threshold}"
        if goal.predicate == "close"
        else "water_on iff 0.4 < (angle modulo 2*pi) < pi; water_off otherwise",
    }
    if ranges:
        result["normalization_ranges"] = list(ranges)
        result["joint_coordinate_units"] = (
            "metres" if goal.fixture == "drawer" else "radians"
        )
    return result


def library_set(name="kitchen-library-v1"):
    if name != "kitchen-library-v1":
        raise ValueError(f"Unknown library set {name!r}")
    spec = {
        "schema_version": 1,
        "requires_boundary_fingerprints": True,
        "revision": 2,
        "sensor_contract_version": 2,
        "id": name,
        "rounds": 2,
        "layout_id": 2,
        "style_id": 1,
        "robot": "PandaOmron",
        "initial_robot_base_ref": "microwave_main_group",
        "fixture_bindings": {
            "microwave": "microwave_main_group",
            "oven": "oven_main_group",
            "sink": "sink_island_group",
            "drawer": "island_cab_left_island_group_4",
        },
        "task_timeout_seconds": 720,
        "max_control_steps_per_task": 6000,
        "container_lifetime_seconds": 14400,
        "settle_steps": 10,
        "tasks": [
            {"index": i, "round": r, "pair_key": task, **asdict(GOALS[task])}
            for r in (1, 2)
            for i, task in enumerate(ROUND, start=(r - 1) * len(ROUND))
        ],
        "reset_between_tasks": False,
        "workspace_persists": True,
        "conversation_persists": True,
        "agent_motion_library": "empty",
        "protocol": "RoboSWE continuous fixture sequence, not official independent RoboCasa episodes",
    }
    spec["sha256"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return spec
