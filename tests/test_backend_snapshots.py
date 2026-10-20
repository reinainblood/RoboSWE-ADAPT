import hashlib
import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest

from pairing.compare_backend_snapshots import compare


def snapshot(path, *, root="/opt/robocasa/", delta=0.0, extra=None, steps=0, missing=None, nan=False):
    xml = f'<mujoco><asset><mesh name="bread" file="{root}models/assets/bread.stl"/></asset></mujoco>'
    physics = {"sim_state": np.array([0., 1. + delta, 2.]), "integration_state": np.array([0., 3., 4.])}
    if nan:
        physics["integration_state"][0] = np.nan
    observations = {"robot0_joint_pos": np.array([1. + delta]), "robot0_eye_in_hand_image": np.full((2, 2, 3), 33 if root.startswith('/opt') else 99, dtype=np.uint8),
                    "robot0_eye_in_hand_depth": np.ones((2, 2, 1)), "annotation.human.task_description": np.array("retrieve bread")}
    if missing:
        del observations[missing]
    config = {"task": "robocasa/GetToastedBread", "seed": 0}
    digest = hashlib.sha256(xml.encode())
    digest.update(physics["sim_state"].astype('<f8').tobytes())
    digest.update(json.dumps(config, sort_keys=True).encode())
    manifest = {"config": config, "episode_metadata": {"object": "bread", "xml": root + "models/assets/bread.xml"},
                "steps": steps, "action_space": {"shape": [24]}, "session_id": root, "runtime": {"machine": root},
                "initial_fingerprint": digest.hexdigest(), "current_fingerprint": digest.hexdigest()}
    manifest.update(extra or {})
    with ZipFile(path, 'w') as z:
        z.writestr('manifest.json', json.dumps(manifest))
        z.writestr('model.xml', xml)
        for member, values in [('physics.npz', physics), ('observations.npz', observations)]:
            buf = BytesIO()
            np.savez_compressed(buf, **values)
            z.writestr(member, buf.getvalue())
    return path


def test_known_paths_normalized_pixels_excluded_fingerprints_still_differ(tmp_path):
    a = snapshot(tmp_path/'a.zip', root='/example/roboswe/upstream/robocasa/')
    b = snapshot(tmp_path/'b.zip')
    r = compare(a, b)
    assert r['initial_state_equivalent']
    assert r['geometry_equal'] and r['physics_exact'] and r['episode_metadata_equal']
    assert not r['xml_exact'] and not r['exact_current_fingerprints_equal']
    assert len(r['observations']['excluded']) == 2
    assert r['excluded_manifest_fields']['runtime']['left'] != r['excluded_manifest_fields']['runtime']['right']


def test_tolerance_distinguished_from_exact_and_large_delta_rejected(tmp_path):
    a = snapshot(tmp_path/'a.zip')
    r = compare(a, snapshot(tmp_path/'b.zip', delta=1e-9))
    assert r['initial_state_equivalent'] and not r['physics_exact']
    assert r['physics']['arrays']['sim_state']['max_abs'] > 0
    r = compare(a, snapshot(tmp_path/'c.zip', delta=1e-6))
    assert not r['initial_state_equivalent'] and not r['physics_close']
    assert r['physics']['arrays']['sim_state']['outside_tolerance'] == 1


def test_metadata_and_unknown_fields_are_visible_and_block_equivalence(tmp_path):
    a = snapshot(tmp_path/'a.zip')
    b = snapshot(tmp_path/'b.zip', extra={'episode_metadata': {'object': 'different'}, 'new_state': 'unrecognized'})
    r = compare(a, b)
    assert not r['initial_state_equivalent']
    assert not r['episode_metadata_equal']
    assert r['unsupported']['right_manifest_fields'] == ['new_state']
    assert r['differences']['episode_metadata']


def test_unknown_absolute_asset_root_not_silently_normalized(tmp_path):
    a = snapshot(tmp_path/'a.zip', root='/other/robocasa/')
    r = compare(a, a)
    assert r['geometry_equal']
    assert not r['initial_state_equivalent']
    assert r['unsupported']['left_absolute_asset_paths']


def test_nonfinite_missing_sensor_and_noninitial_states_rejected(tmp_path):
    a = snapshot(tmp_path/'a.zip')
    r = compare(a, snapshot(tmp_path/'b.zip', nan=True))
    assert not r['physics_close']
    assert r['physics']['arrays']['integration_state']['max_abs'] is None
    json.dumps(r, allow_nan=False)
    r = compare(a, snapshot(tmp_path/'c.zip', missing='robot0_joint_pos'))
    assert not r['observations_close']
    r = compare(snapshot(tmp_path/'d.zip', steps=5), snapshot(tmp_path/'e.zip', steps=5))
    assert not r['both_initial_step_zero'] and not r['initial_state_equivalent']


def test_bad_fingerprint_and_incomplete_zip_rejected(tmp_path):
    a = snapshot(tmp_path/'a.zip')
    r = compare(a, snapshot(tmp_path/'b.zip', extra={'current_fingerprint': 'forged'}))
    assert not r['initial_state_equivalent']
    assert not r['fingerprints']['right']['current_matches_contents']
    with ZipFile(tmp_path/'bad.zip', 'w') as z:
        z.writestr('manifest.json', '{}')
    with pytest.raises(ValueError, match='Missing snapshot'):
        compare(a, tmp_path/'bad.zip')
