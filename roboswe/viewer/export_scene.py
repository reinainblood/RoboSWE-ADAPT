"""Read a session snapshot and export visual geometry; run inside its container."""

import base64
import gzip
import json
import sys
from io import BytesIO
from zipfile import ZipFile

import mujoco
import numpy as np
from PIL import Image

from robot_session.client import SessionClient


def buffer(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode()


def main():
    archive = ZipFile(BytesIO(SessionClient().request("/snapshot")))
    meta = json.loads(archive.read("manifest.json"))
    model = mujoco.MjModel.from_xml_string(archive.read("model.xml").decode())
    data = mujoco.MjData(model)
    with np.load(BytesIO(archive.read("physics.npz")), allow_pickle=False) as physics:
        mujoco.mj_setState(
            model, data, physics["integration_state"], meta["mujoco_state_spec"]
        )
    mujoco.mj_forward(model, data)
    result = {
        "step": meta["steps"],
        "session_id": meta["session_id"],
        "task": meta["config"]["task"],
        "meshes": {},
        "textures": {},
        "geoms": [],
    }
    for i in range(model.ngeom):
        if model.geom_group[i] != 1 or model.geom_rgba[i, 3] == 0:
            continue
        mat = int(model.geom_matid[i])
        rgba = model.geom_rgba[i].copy()
        texture = -1
        repeat = [1, 1]
        if mat >= 0:
            if np.allclose(rgba, [0.5, 0.5, 0.5, 1]):
                rgba = model.mat_rgba[mat].copy()
            ids = model.mat_texid[mat]
            candidates = [int(t) for t in ids if t >= 0 and model.tex_type[t] == 0]
            if candidates:
                texture = candidates[0]
                repeat = model.mat_texrepeat[mat].tolist()
        if texture >= 0 and str(texture) not in result["textures"]:
            w, h, c = map(
                int,
                (
                    model.tex_width[texture],
                    model.tex_height[texture],
                    model.tex_nchannel[texture],
                ),
            )
            adr = int(model.tex_adr[texture])
            pixels = model.tex_data[adr : adr + w * h * c].reshape(h, w, c)
            im = Image.fromarray(pixels.squeeze() if c == 1 else pixels)
            im.thumbnail((512, 512))
            png = BytesIO()
            im.save(png, format="PNG")
            result["textures"][str(texture)] = base64.b64encode(png.getvalue()).decode()
        mesh_id = int(model.geom_dataid[i])
        kind = int(model.geom_type[i])
        if (
            kind == int(mujoco.mjtGeom.mjGEOM_MESH)
            and str(mesh_id) not in result["meshes"]
        ):
            va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
            vertices = model.mesh_vert[va : va + vn]
            faces = model.mesh_face[fa : fa + fn]
            uvfaces = model.mesh_facetexcoord[fa : fa + fn]
            ta = int(model.mesh_texcoordadr[mesh_id])
            mesh = {}
            if ta >= 0 and np.all(uvfaces >= 0):
                na = int(model.mesh_normaladr[mesh_id])
                nf = model.mesh_facenormal[fa : fa + fn]
                # Preserve UV/normal seams without repeating every triangle vertex.
                triples = np.column_stack(
                    (faces.reshape(-1), uvfaces.reshape(-1), nf.reshape(-1))
                )
                unique, indices = np.unique(triples, axis=0, return_inverse=True)
                mesh["positions"] = buffer(vertices[unique[:, 0]], "<f4")
                mesh["uv"] = buffer(model.mesh_texcoord[ta + unique[:, 1]], "<f4")
                mesh["indices"] = buffer(indices, "<u4")
                if na >= 0 and np.all(nf >= 0):
                    mesh["normals"] = buffer(
                        model.mesh_normal[na + unique[:, 2]], "<f4"
                    )
            else:
                mesh["positions"] = buffer(vertices, "<f4")
                mesh["indices"] = buffer(faces, "<u4")
            result["meshes"][str(mesh_id)] = mesh
        result["geoms"].append(
            {
                "name": model.geom(i).name,
                "kind": kind,
                "mesh": mesh_id,
                "position": data.geom_xpos[i].tolist(),
                "rotation": data.geom_xmat[i].tolist(),
                "size": model.geom_size[i].tolist(),
                "rgba": rgba.tolist(),
                "texture": texture,
                "repeat": repeat,
            }
        )
    sys.stdout.buffer.write(
        gzip.compress(json.dumps(result, separators=(",", ":")).encode())
    )


if __name__ == "__main__":
    main()
