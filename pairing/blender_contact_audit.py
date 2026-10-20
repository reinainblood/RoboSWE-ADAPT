"""Read-only button-contact instrumentation during exact code replay."""
import hashlib
import json
from pathlib import Path

from pairing.cohort50 import ROOT, emit
from roboswe.jobs import save

SOURCE=Path('jobs/20261006T100149Z-296b83a6f2')


def contact_worker(*args):
    from pairing import overnight_suite as suite
    original=suite.fixture_diagnostics

    def diagnostics(raw):
        result=original(raw)
        blender=getattr(raw,'blender',None)
        if blender is not None:
            geom=f'{blender.name}_power_button_main'
            contacts={}
            for hand in ('left','right'):
                gripper=raw.robots[0].gripper.get(hand)
                contacts[hand]=bool(raw.check_contact(gripper,geom)) if gripper is not None else None
            result.setdefault('blender',{})['audit_button_contacts']=contacts
        return result

    suite.fixture_diagnostics=diagnostics
    try:suite.suite_worker(*args)
    finally:suite.fixture_diagnostics=original


def summarize_contacts(rows):
    result={'samples':0,'left_contact_samples':0,'right_contact_samples':0,
        'left_release_events_with_lid_closed':[],'right_release_events_with_lid_closed':[]}
    previous={'left':False,'right':False}
    for row in rows:
        state=row.get('fixture_diagnostics',{}).get('blender',{})
        contacts=state.get('audit_button_contacts')
        if contacts is None:continue
        result['samples']+=1
        for hand in ('left','right'):
            pressed=contacts.get(hand) is True
            result[f'{hand}_contact_samples']+=int(pressed)
            if previous[hand] and not pressed and state.get('lid_on_blender') is True:
                result[f'{hand}_release_events_with_lid_closed'].append(row['step'])
            previous[hand]=pressed
    return result


def main():
    from pairing import cohort50_replay as replay
    before={p.parent.name for p in Path('jobs').glob('*/job.json')}
    original_worker=replay.suite_worker
    replay.suite_worker=contact_worker
    try:replay.run(SOURCE)
    finally:
        replay.suite_worker=original_worker
        matches=[]
        for path in Path('jobs').glob('*/job.json'):
            if path.parent.name in before:continue
            record=json.loads(path.read_text())
            if record.get('source_job')==SOURCE.name:matches.append((path.parent,record))
        if len(matches)!=1:raise RuntimeError('Could not identify the contact-audit replay')
        directory,record=matches[0]
        telemetry=directory/'stage-telemetry.jsonl'
        rows=[json.loads(line) for line in telemetry.read_text().splitlines()] if telemetry.exists() else []
        result_path=directory/'report.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else {}
        audit={'source_job':SOURCE.name,'replay_job':record['id'],'replay_status':record['status'],
            'original_trace_reproduced':result.get('reproduced') is True and result.get('final_fingerprint_matches') is True,
            'instrumentation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'contacts':summarize_contacts(rows),
            'scope':'Read-only contact queries at recorded control-step endpoints. No model inference, physics changes, or success-predicate changes. Contact samples alone do not establish success under an alternative verifier; absence cannot exclude brief substep contacts.'}
        emit(directory/'contact-audit.json',audit);emit(ROOT/'blender-contact-replay-audit.json',audit)
        record['artifacts']['contact_audit']='contact-audit.json';save(directory,record)
        print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
