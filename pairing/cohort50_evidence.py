"""Register completed cohort evidence in the existing read-only job viewer."""
import json
from pathlib import Path

from pairing.run_toaster_learning import write_human_outputs
from roboswe.jobs import save


def finalize(directory,record):
    if not record.get('finished_at') or record.get('status') in ('running','starting'):
        return False
    changed=False
    if record.get('kind')=='cohort50-attempt' and (directory/'model-transcript.jsonl').exists() and (directory/'submissions.jsonl').exists():
        # Rebuild deterministically from original receipts; repeated runs do not
        # append duplicate sections or overwrite the underlying model evidence.
        provider=[]
        path=directory/'provider-responses.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                row=json.loads(line);message=row['response']['choices'][0]['message']
                reasoning=message.get('reasoning_content') or message.get('reasoning') or ''
                provider.append(f"Call {row['call']} · control step {row['control_step']} · request {row['elapsed_seconds']:.2f}s\nPROVIDER RESPONSE — NOT PROOF OF EXECUTION\nReturned reasoning:\n{reasoning}\nFinal content:\n{message.get('content','')}\n")
        write_human_outputs(directory)
        if provider:
            executed=(directory/'stdout.txt').read_text()
            temp=directory/'stdout.txt.tmp';temp.write_text('\n'.join(provider)+'\nJOURNALED ACTIONS AND EXECUTION OUTPUT\n'+executed);temp.replace(directory/'stdout.txt')
        record['artifacts']['stdout']='stdout.txt';changed=True
    for label,name in [('timing','timing.jsonl'),('transcript','model-transcript.jsonl'),('provider_responses','provider-responses.jsonl'),('actions','actions.jsonl'),('submissions','submissions.jsonl'),('continuity','continuity.json'),('protocol_recovery','protocol-recovery.jsonl')]:
        if (directory/name).exists() and record['artifacts'].get(label)!=name:
            record['artifacts'][label]=name;changed=True
    if changed:save(directory,record)
    return changed


def main():
    changed=[]
    for path in Path('jobs').glob('*/job.json'):
        record=json.loads(path.read_text())
        if record.get('kind','').startswith('cohort50-') and finalize(path.parent,record):changed.append(record['id'])
    print(json.dumps({'registered_jobs':changed}))


if __name__=='__main__':main()
