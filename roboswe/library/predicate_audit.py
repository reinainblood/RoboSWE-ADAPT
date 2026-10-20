"""Predicate-specific visual cross-checks. Never modifies mechanical scores."""
import ast
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .glimpse_judge import judge_video

ROOT = Path(__file__).resolve().parents[2]
BLEND = 'environments/kitchen/composite/making_smoothies/blend_ingredients.py'


def predicate_contract(task, upstream=None):
    """Read source without importing/executing upstream code; no measured state."""
    root = Path(upstream or ROOT / 'upstream/robocasa/robocasa')
    candidates = []
    for path in (root / 'environments/kitchen').rglob('*.py'):
        source = path.read_text()
        tree = ast.parse(source)
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and cls.name == task:
                for fn in cls.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == '_check_success':
                        candidates.append({'path':str(path.relative_to(root)),
                            'sha256':hashlib.sha256(source.encode()).hexdigest(),
                            'source':ast.get_source_segment(source, fn)})
    if len(candidates) != 1:
        raise ValueError('Task needs one explicit upstream predicate or a reviewed inherited-predicate contract')
    rubric = None
    if task == 'BlendIngredients':
        rubric = [
            {'id':'fruit_inside','description':'The instructed fruit ends inside the blender container. Look for its placement through the opening and its final location; use null if hidden or ambiguous. The mechanical predicate uses obj_inside_of with th=0.01; pixels cannot certify that tolerance.'},
            {'id':'lid_closed','description':'The lid ends seated on the blender and upright. Mechanical activation requires lid position within 0.04 metres of the closed position and upright within the upstream th=7 check. Judge visible seating only, not exact tolerances.'},
            {'id':'activation','description':'Look for robot gripper contact with the blender power button followed by release while the lid is closed, or a clear visible active indicator. Upstream turns on at contact RELEASE and turns off when the lid is removed. Do not assume an internally stored on bit is visible. Use null if no conclusive visual evidence; false only for visible contradiction.'},
        ]
        p = root / 'models/fixtures/blender.py'
        candidates.append({'path':str(p.relative_to(root)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'source':p.read_text()})
    elif task == 'ToastBagel':
        rubric = [
            {'id':'bagel_rack_contact','description':'The bagel ends touching a toaster-oven rack or tray. Either lower or upper rack is valid when present; the helper falls back to level 0. Look for visible contact/placement and use null if occluded. The native test checks contact, not an additional containment or stable-support criterion.'},
            {'id':'door_closed','description':'The toaster-oven door ends closed. Look for visible closure; false only for a clearly open door and null when ambiguous. The native inherited is_closed check requires normalized door-joint position <=0.005; pixels cannot certify that exact tolerance.'},
            {'id':'timer_set','description':'The timer has been set: native success requires stored normalized time >=0.1. Look for visible timer-knob manipulation and a readable resulting setting; use null when the setting or threshold cannot be established visually. Knob contact alone is not proof. This is a timer-setting task, NOT a test of completed heating, elapsed toasting, browned food, or successful removal/serving; do not require or infer those outcomes.'},
            {'id':'gripper_withdrawn','description':'The robot right gripper withdraws from the bagel at the end. Native gripper_obj_far requires the right end-effector site to be more than 0.25 metres from the bagel origin. Judge clear visual separation only; use null if the exact distance is unresolved or occluded. This is not a general all-links clearance test.'},
        ]
        # Only reviewed dependencies; preserve file hashes and precise excerpts.
        dependencies = {
            'models/fixtures/toaster_oven.py': {'has_multiple_rack_levels', 'check_rack_contact', 'get_state', 'update_state'},
            'models/fixtures/fixture.py': {'is_closed', 'get_joint_state', 'door_joint_names'},
            'utils/object_utils.py': {'gripper_obj_far'},
        }
        for relative, names in dependencies.items():
            p = root / relative
            raw = p.read_bytes()
            text = raw.decode()
            excerpts = [fn for fn in ast.walk(ast.parse(text))
                        if isinstance(fn, ast.FunctionDef) and fn.name in names]
            if {fn.name for fn in excerpts} != names:
                raise ValueError('ToastBagel reviewed predicate dependency is missing')
            candidates.append({'path':relative, 'sha256':hashlib.sha256(raw).hexdigest(),
                'source':'\n\n'.join(ast.get_source_segment(text, fn) for fn in excerpts),
                'excerpts':[{'name':fn.name, 'line_start':fn.lineno, 'line_end':fn.end_lineno}
                            for fn in excerpts]})
    return {'task':task, 'sources':candidates, 'rubric':rubric,
            'purpose':'Cross-check visible evidence for predicate conditions; never certify hidden simulator state, exact ticks, or absence of hacking.'}


def comparison_status(visible, mechanical, *, failed=False):
    """Compare visual task evidence with the recorded mechanical outcome."""
    if failed:
        return "judge_failed"
    if visible is None:
        return "not_observable"
    if mechanical is None:
        return "uncompared"
    if visible != mechanical:
        return "visual_mechanical_disagreement"
    return "consistent_success" if visible else "consistent_failure"


def audit_job(directory, *, request_fn=None, upstream=None, max_tokens=8192, reasoning_effort=None, judge_backend="glimpse"):
    directory = Path(directory).resolve()
    record = json.loads((directory/'job.json').read_text())
    if not record.get('finished_at') or record.get('status') in ('running','starting'):
        raise ValueError('Only finalized episodes can be audited')
    trial = json.loads((directory/'report.json').read_text())
    contract = predicate_contract(trial['task'], upstream)
    manifest_path = directory/'video.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else record.get('video',{})
    views = {}
    for v in manifest.get('views',[]):
        name = v.get('simulation','')
        if not name or Path(name).name != name: continue
        path = directory/name
        if path.is_symlink(): raise ValueError('Symlinked video is not accepted')
        if path.is_file(): views[v['id']] = path
    if len(views) < 2: raise ValueError('Predicate audit requires at least two camera views')
    primary = views.pop('overview', None)
    if primary is None: primary = views.pop(next(iter(views)))
    verification_path = directory/'verification.json'
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
    replay_verified = record.get('status') in ('verified_pass', 'verified_fail') and verification.get('reproduced') is True
    mechanical = verification.get('verified_success') if replay_verified else trial.get('observed_success')
    output = directory/'predicate-audits'/uuid4().hex
    report = judge_video(primary, instruction=trial.get('instruction',trial['task']),
        output=output, criteria=contract, rubric=contract['rubric'],
        mechanical_success=mechanical,
        additional_videos=views, max_video_bytes=96*2**20, max_video_seconds=1200,
        max_tokens=max_tokens, reasoning_effort=reasoning_effort,
        judge_backend=judge_backend, request_fn=request_fn)
    report['predicate_contract'] = contract
    report['mechanical_outcome_source'] = 'independent_replay' if replay_verified else 'observed_upstream_predicate'
    report['recording_provenance'] = 'local finalized episode; source hashes are recorded, but historical recording identity is not independently re-certified by this audit'
    checks = report.get('answer',{}).get('checks',[])
    known = [c for c in checks if c['satisfied'] is not None]
    report['visual_support_score'] = sum(c['satisfied'] is True for c in checks)/len(checks) if checks and len(known)==len(checks) else None
    report['observable_fraction'] = len(known)/len(checks) if checks else 0
    report['contradiction_detected'] = any(c['satisfied'] is False for c in checks)
    report['review_status'] = 'judge_failed' if report['status']=='judge_failed' else 'contradicted' if report['contradiction_detected'] else 'not_observable' if len(known)<len(checks) else 'visually_supported'
    report['comparison_status'] = comparison_status(
        report.get('answer', {}).get('visible_success'), report.get('mechanical_success'),
        failed=report['status'] == 'judge_failed')
    report['predicate_disagreement'] = report['comparison_status'] == 'visual_mechanical_disagreement'
    report['contradiction_scope'] = 'contradiction_detected refers to visible task criteria, not necessarily disagreement with a mechanical failure'
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
