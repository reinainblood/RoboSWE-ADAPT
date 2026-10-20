"""Read-only comparison of two RoboSWE inspection snapshot ZIPs.

Output schema v1 distinguishes exact fingerprints from normalized XML equality,
config/episode semantics and numeric closeness (rtol=0, atol=1e-8). RGB/depth
pixels are explicitly excluded. Asset bytes and future physics trajectories are
not verified by this comparison. Unknown manifest fields and ZIP members are
reported and prevent a positive initial_state_equivalent result.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

ATOL = 1e-8
ROOTS = {
    "/example/roboswe/upstream/robocasa/": "ASSET_ROOT/robocasa/",
    "/example/roboswe/upstream/robosuite/": "ASSET_ROOT/robosuite/",
    "/opt/robocasa/": "ASSET_ROOT/robocasa/",
    "/opt/robosuite/": "ASSET_ROOT/robosuite/",
}
REQUIRED = {"manifest.json", "model.xml", "physics.npz", "observations.npz"}
SEMANTIC_FIELDS = {"api_version", "cameras", "info", "last_result", "mujoco_state_spec", "phase", "ready", "recording_api", "robot", "role", "snapshot_kind", "steps"}
EXCLUDED_MANIFEST = {"session_id", "recording", "runtime", "initial_fingerprint", "current_fingerprint"}


def normalize_string(value):
    for source, target in ROOTS.items():
        if value.startswith(source):
            return target + value[len(source):]
    return value


def normalize_paths(value):
    if isinstance(value, str):
        return normalize_string(value)
    if isinstance(value, dict):
        return {k: normalize_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_paths(v) for v in value]
    return value


def canonical_model(xml):
    root = ET.fromstring(xml)
    unmapped = []
    for element in root.iter():
        for name, value in element.attrib.items():
            if name in {"file", "meshdir", "texturedir", "assetdir"}:
                normalized = normalize_string(value)
                if normalized.startswith("/"):
                    unmapped.append({"element": element.tag, "attribute": name, "value": value})
                element.set(name, normalized)
    return ET.canonicalize(ET.tostring(root, encoding="unicode"), strip_text=True), unmapped


def load_snapshot(path):
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members are unsupported")
        missing = REQUIRED - set(names)
        if missing:
            raise ValueError(f"Missing snapshot members: {sorted(missing)}")
        result = {"path": str(Path(path).resolve()), "manifest": json.loads(archive.read("manifest.json")),
                  "xml": archive.read("model.xml").decode(), "unsupported_members": sorted(set(names) - REQUIRED)}
        for name, member in (("physics", "physics.npz"), ("observations", "observations.npz")):
            with np.load(BytesIO(archive.read(member)), allow_pickle=False) as arrays:
                result[name] = {key: arrays[key].copy() for key in arrays.files}
        return result


def json_differences(left, right, path="$", *, numeric_tolerance=False):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                differences.append({"path": path + "." + key, "reason": "missing_key", "left_present": key in left, "right_present": key in right})
            else:
                differences.extend(json_differences(left[key], right[key], path + "." + key, numeric_tolerance=numeric_tolerance))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [{"path": path, "reason": "length", "left": len(left), "right": len(right)}]
        return [difference for i, (a, b) in enumerate(zip(left, right, strict=True)) for difference in json_differences(a, b, f"{path}[{i}]", numeric_tolerance=numeric_tolerance)]
    numeric = type(left) in (int, float) and type(right) in (int, float)
    if numeric and numeric_tolerance:
        same = math.isfinite(left) and math.isfinite(right) and abs(left - right) <= ATOL
    else:
        same = type(left) is type(right) and left == right
    return [] if same else [{"path": path, "reason": "value", "left": left, "right": right}]


def compare_arrays(left, right, *, exclude_pixels=False):
    results = {}
    excluded = []
    for key in sorted(left.keys() | right.keys()):
        if exclude_pixels and key.endswith(("_image", "_depth")):
            excluded.append({"name": key, "reason": "renderer-dependent RGB/depth pixels", "left_shape": list(left[key].shape) if key in left else None, "right_shape": list(right[key].shape) if key in right else None})
            continue
        if key not in left or key not in right:
            results[key] = {"close": False, "exact": False, "reason": "missing_array", "left_present": key in left, "right_present": key in right}
            continue
        a, b = left[key], right[key]
        entry = {"left_shape": list(a.shape), "right_shape": list(b.shape), "left_dtype": str(a.dtype), "right_dtype": str(b.dtype)}
        if a.shape != b.shape:
            entry.update(close=False, exact=False, reason="shape")
        elif a.dtype.kind in "biuf" and b.dtype.kind in "biuf":
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            entry.update(exact=a.dtype == b.dtype and a.tobytes() == b.tobytes(), close=finite and bool(np.all(delta <= ATOL)),
                         finite=finite, max_abs=float(delta.max()) if finite and delta.size else (0.0 if finite else None),
                         differing_elements=int(np.count_nonzero(a != b)), outside_tolerance=int(np.count_nonzero(delta > ATOL)) if finite else None)
        elif a.dtype.kind in "US" and b.dtype.kind in "US":
            entry.update(exact=a.dtype == b.dtype and a.tobytes() == b.tobytes(), close=bool(np.array_equal(a, b)), reason="non_numeric_values_compared_exactly")
        else:
            entry.update(exact=False, close=False, reason="unsupported_dtype")
        results[key] = entry
    return {"close": bool(results) and all(v["close"] for v in results.values()), "exact": bool(results) and all(v["exact"] for v in results.values()), "arrays": results, "excluded": excluded}


def fingerprint_receipt(snapshot):
    manifest = snapshot["manifest"]
    computed = None
    if "sim_state" in snapshot["physics"] and "config" in manifest:
        digest = hashlib.sha256(snapshot["xml"].encode())
        digest.update(np.asarray(snapshot["physics"]["sim_state"], dtype="<f8").tobytes())
        digest.update(json.dumps(manifest["config"], sort_keys=True).encode())
        computed = digest.hexdigest()
    return {"initial": manifest.get("initial_fingerprint"), "current": manifest.get("current_fingerprint"),
            "computed_current": computed, "current_matches_contents": computed is not None and computed == manifest.get("current_fingerprint")}


def compare(left_path, right_path):
    left, right = load_snapshot(left_path), load_snapshot(right_path)
    a, b = left["manifest"], right["manifest"]
    left_xml, left_unmapped = canonical_model(left["xml"])
    right_xml, right_unmapped = canonical_model(right["xml"])
    physics = compare_arrays(left["physics"], right["physics"])
    observations = compare_arrays(left["observations"], right["observations"], exclude_pixels=True)
    config = json_differences(a.get("config"), b.get("config"))
    metadata = json_differences(normalize_paths(a.get("episode_metadata")), normalize_paths(b.get("episode_metadata")))
    action = json_differences(normalize_paths(a.get("action_space")), normalize_paths(b.get("action_space")), numeric_tolerance=True)
    semantic = json_differences({k: a[k] for k in SEMANTIC_FIELDS if k in a}, {k: b[k] for k in SEMANTIC_FIELDS if k in b})
    known = SEMANTIC_FIELDS | EXCLUDED_MANIFEST | {"config", "episode_metadata", "action_space"}
    unsupported = {"left_manifest_fields": sorted(a.keys() - known), "right_manifest_fields": sorted(b.keys() - known),
                   "left_zip_members": left["unsupported_members"], "right_zip_members": right["unsupported_members"],
                   "left_absolute_asset_paths": left_unmapped, "right_absolute_asset_paths": right_unmapped}
    fingerprints = {"left": fingerprint_receipt(left), "right": fingerprint_receipt(right)}
    result = {"schema_version": 1, "left": left["path"], "right": right["path"], "tolerance": {"rtol": 0, "atol": ATOL},
              "geometry_equal": left_xml == right_xml, "xml_exact": left["xml"] == right["xml"],
              "config_equal": not config and "config" in a and "config" in b,
              "episode_metadata_equal": not metadata and "episode_metadata" in a and "episode_metadata" in b,
              "action_space_close": not action and "action_space" in a and "action_space" in b,
              "manifest_semantics_equal": not semantic,
              "physics_close": physics["close"], "physics_exact": physics["exact"], "observations_close": observations["close"],
              "both_initial_step_zero": a.get("steps") == b.get("steps") == 0,
              "fingerprints": fingerprints,
              "exact_current_fingerprints_equal": fingerprints["left"]["current"] is not None and fingerprints["left"]["current"] == fingerprints["right"]["current"],
              "differences": {"config": config, "episode_metadata": metadata, "action_space": action, "manifest_semantics": semantic,
                              "normalized_xml_diff": [line[:2000] for line in list(difflib.unified_diff(left_xml.replace("><", ">\n<").splitlines(), right_xml.replace("><", ">\n<").splitlines(), fromfile="left", tofile="right"))[:40]],
                              "normalized_xml_diff_limit": "First 40 diff lines, at most 2000 characters each"},
              "physics": physics, "observations": observations, "unsupported": unsupported,
              "excluded_manifest_fields": {key: {"left": a.get(key), "right": b.get(key)} for key in sorted(EXCLUDED_MANIFEST)},
              "limitations": ["Known asset root paths normalized; actual mesh/texture file bytes are not compared.",
                              "Renderer-dependent RGB and depth pixels are excluded.",
                              "Closeness of one state does not establish identical future trajectories or bitwise backend equivalence."]}
    checks = ("geometry_equal", "config_equal", "episode_metadata_equal", "action_space_close", "manifest_semantics_equal", "physics_close", "observations_close", "both_initial_step_zero")
    result["initial_state_equivalent"] = all(result[key] for key in checks) and not any(unsupported.values()) and all(x["current_matches_contents"] for x in fingerprints.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["initial_state_equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
