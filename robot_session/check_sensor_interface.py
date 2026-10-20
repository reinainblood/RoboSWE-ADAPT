"""Compare public RGB-D to actual renderer data without stepping the robot."""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from roboswe.library.runtime import public_observation
from roboswe.library.spec import library_set

from .session import RobotSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    spec = library_set()
    config = {
        "robot": spec["robot"],
        "task": "robocasa/Kitchen",
        "split": None,
        "seed": 0,
        "env_kwargs": {
            "layout_ids": spec["layout_id"],
            "style_ids": spec["style_id"],
            "init_robot_base_ref": spec["initial_robot_base_ref"],
            "camera_widths": 128,
            "camera_heights": 128,
            "camera_depths": True,
        },
    }
    session = RobotSession(config)
    try:
        from robosuite.utils.camera_utils import get_real_depth_map

        before = session.fingerprint()
        observed = public_observation(session)
        comparisons = {}
        with np.load(
            BytesIO(base64.b64decode(observed["depth_npz_base64"])), allow_pickle=False
        ) as depths:
            for key, meta in observed["camera_calibration"].items():
                rgb, depth = session.raw.sim.render(
                    camera_name=meta["camera"], height=128, width=128, depth=True
                )
                expected = get_real_depth_map(session.raw.sim, depth[::-1]).astype(
                    np.float32
                )
                exported_rgb = np.asarray(
                    Image.open(
                        BytesIO(base64.b64decode(observed["images_png_base64"][key]))
                    )
                )
                comparisons[key] = {
                    "rgb_max_error": int(
                        np.abs(exported_rgb.astype(int) - rgb[::-1].astype(int)).max()
                    ),
                    "depth_max_error_m": float(
                        np.abs(depths[meta["depth_key"]] - expected).max()
                    ),
                }
        assert all(
            v["rgb_max_error"] == 0 and v["depth_max_error_m"] < 1e-5
            for v in comparisons.values()
        ), comparisons
        assert before == session.fingerprint(), (
            "Interface inspection mutated the simulation"
        )
        receipt = {
            "kind": "sensor_contract_control_not_task_trial",
            "config": config,
            "initial_fingerprint": before,
            "simulator_unchanged": True,
            "comparisons": comparisons,
            "action_schema": observed["action_schema"],
            "camera_calibration": observed["camera_calibration"],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"RGB-D/controller interface control passed: {output}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
