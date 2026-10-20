"""Compose local robosuite models from unmodified, pinned manufacturer MJCF.

The shoulder riser and wrist spacers are provisional simulation fixtures. They
are not manufacturer-certified mechanical adapters. Joint geometry, inertias,
limits and position-actuator gains remain those of the source models.
"""

from copy import deepcopy
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

import numpy as np

ROOT = Path(__file__).parent
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


def _flatten(path):
    root = ET.parse(path).getroot()
    classes = {}

    def defaults(node, inherited):
        current = deepcopy(inherited)
        for child in node:
            if child.tag != "default":
                current.setdefault(child.tag, {}).update(child.attrib)
        classes[node.get("class", "main")] = current
        for child in node.findall("default"):
            defaults(child, current)

    default = root.find("default")
    if default is not None:
        defaults(default, {})
        root.remove(default)

    def apply(node, inherited="main"):
        cls = node.attrib.pop("class", inherited)
        attributes = classes.get(cls, {}).get(node.tag, {})
        node.attrib = {**attributes, **node.attrib}
        childclass = node.attrib.pop("childclass", inherited)
        for child in node:
            apply(child, childclass)

    apply(root)
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "")
    for asset in root.findall("asset/*[@file]"):
        asset.set("file", str((path.parent / meshdir / asset.get("file")).resolve()))
    compiler.attrib.pop("meshdir", None)
    # robosuite uses group 0 for collision and group 1 for visual geometry.
    for index, geom in enumerate(root.iter("geom")):
        visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
        geom.set("group", "1" if visual else "0")
        if "name" not in geom.attrib:
            geom.set("name", f"geom_{index}")
    for joint in root.iter("joint"):
        if "range" in joint.attrib:
            joint.set("limited", "true")
    for actuator in root.findall("actuator/*"):
        if "ctrlrange" in actuator.attrib:
            actuator.set("ctrllimited", "true")
        if "forcerange" in actuator.attrib:
            actuator.set("forcelimited", "true")
    for tag in ("keyframe", "equality"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    return root


def _write(root, name):
    destination = ROOT / "generated" / name
    destination.parent.mkdir(exist_ok=True)
    ET.indent(root)
    temporary = destination.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_text(ET.tostring(root, encoding="unicode") + "\n")
    temporary.replace(destination)
    return destination


def _mesh_bounds(mesh):
    """Source STL bounds in the sensor body's coordinates (no visual guessing)."""
    data = Path(mesh.get("file")).read_bytes()
    count = int.from_bytes(data[80:84], "little")
    if len(data) != 84 + 50 * count:
        raise ValueError("Expected manufacturer's binary STL")
    triangles = np.frombuffer(
        data,
        offset=84,
        count=count,
        dtype=np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        ),
    )
    vertices = triangles["vertices"].reshape(-1, 3).astype(float)
    vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
    return vertices.min(axis=0), vertices.max(axis=0)


def build_hand(side):
    prefix = side[0]
    path = ROOT / "vendor/wuji/hand2/hand2_beta2/body/mjcf" / f"{side}_with_mount.xml"
    root = _flatten(path)
    root.set("model", f"WujiHand2{side.title()}")
    mount = root.find("worldbody/body")
    # The OpenArm wrist extends along -Z; Wuji's with-mount model does too.
    # A provisional spacer clears the retained OpenArm wrist housing.
    mount.set("pos", "0 0 -0.115")
    eef = ET.SubElement(mount, "body", name="eef", pos="0 0 0")
    for name, pos, size, kind in [
        ("grip_site", "0 0 -0.13", "0.003", "sphere"),
        ("grip_site_cylinder", "0 0 -0.13", "0.003 0.02", "cylinder"),
        ("ee", "0 0 -0.13", "0.001", "sphere"),
        ("ee_x", "0.02 0 -0.13", "0.001", "sphere"),
        ("ee_y", "0 0.02 -0.13", "0.001", "sphere"),
        ("ee_z", "0 0 -0.11", "0.001", "sphere"),
    ]:
        ET.SubElement(
            eef,
            "site",
            name=name,
            pos=pos,
            size=size,
            type=kind,
            rgba="0 0 0 0",
            group="4",
        )
    ET.SubElement(mount, "site", name="wrist_sensor", size="0.001", rgba="0 0 0 0")
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "force", name="force_ee", site="wrist_sensor")
    ET.SubElement(sensors, "torque", name="torque_ee", site="wrist_sensor")
    for finger in FINGERS:
        name = f"{prefix}_{finger}_tip_sensor_frame"
        body = root.find(f".//body[@name='{name}']")
        collision = next(g for g in body.findall("geom") if g.get("group") == "0")
        collision.set("name", f"{finger}_pad_collision")
        mesh = root.find(f"asset/mesh[@name='{collision.get('mesh')}']")
        lower, upper = _mesh_bounds(mesh)
        site = f"{finger}_tactile_site"
        ET.SubElement(
            body,
            "site",
            name=site,
            type="box",
            pos=" ".join(map(str, (lower + upper) / 2)),
            size=" ".join(map(str, (upper - lower) / 2 + 0.0003)),
            group="4",
            rgba="0 0.8 1 0",
        )
        ET.SubElement(sensors, "touch", name=f"{finger}_touch", site=site)
        ET.SubElement(sensors, "force", name=f"{finger}_force", site=site)
        ET.SubElement(sensors, "torque", name=f"{finger}_torque", site=site)
    return _write(root, f"wuji_{side}.xml")


def build_arms():
    root = _flatten(ROOT / "vendor/openarm/v2/openarm_bimanual.xml")
    root.set("model", "MichaelOpenArmWuji")
    world = root.find("worldbody")
    shoulders = {
        s: world.find(f"body[@name='openarm_{s}_base_link']") for s in ("right", "left")
    }
    world.clear()
    base = ET.SubElement(world, "body", name="base")
    ET.SubElement(
        base,
        "geom",
        name="riser",
        type="box",
        pos="0 0 0.19",
        size="0.07 0.085 0.19",
        mass="2",
        rgba="0.16 0.18 0.2 1",
        group="0",
    )
    ET.SubElement(
        base,
        "geom",
        name="riser_visual",
        type="box",
        pos="0 0 0.19",
        size="0.07 0.085 0.19",
        contype="0",
        conaffinity="0",
        rgba="0.16 0.18 0.2 1",
        group="1",
    )
    # XML traversal and actuator order must match robosuite's right / left convention.
    for side, body in shoulders.items():
        body.set("pos", f"0 {0.031 if side == 'left' else -0.031} 0.40")
        ET.SubElement(body, "site", name=f"{side}_center", size="0.002", rgba="0 0 0 0")
        wrist = body.find(f".//body[@name='openarm_{side}_ee_base_link']")
        for child in list(wrist):
            if child.tag == "body":
                wrist.remove(child)
        ET.SubElement(wrist, "body", name=f"{side}_hand")
        for site in body.iter("site"):
            site.set("rgba", "0 0 0 0")
        base.append(body)
    actuator = root.find("actuator")
    arm_actuators = [a for a in actuator if "finger" not in a.get("joint", "")]
    actuator.clear()
    for side in ("right", "left"):
        actuator.extend(a for a in arm_actuators if side in a.get("name"))
    used = {g.get("mesh") for g in root.iter("geom")}
    asset = root.find("asset")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in used:
            asset.remove(mesh)
    return _write(root, "openarm.xml")


def build_models():
    return {"robot": build_arms(), **{s: build_hand(s) for s in ("right", "left")}}


if __name__ == "__main__":
    print(build_models())
