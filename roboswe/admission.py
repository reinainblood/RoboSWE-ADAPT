"""Explicit, sequential simulator admission with no model or code execution."""
import hashlib
import json
import time
from pathlib import Path

from .adaptive import build_plan


def prepare(dataset, config, *, images=None, backend=None, output):
    plan = build_plan(dataset, config, images_path=images, backend=backend,
                      include_prompts=False)
    target = Path(output).resolve()
    evidence = target.with_name(target.name + '.admission')
    rejected = target.with_name(target.name + '.rejected.json')
    if any(p.exists() for p in (target, evidence, rejected)):
        raise ValueError('Admission output/evidence already exists; refusing overwrite')
    plan.update(kind='no-model-admission-v1', output=str(target),
                evidence=str(evidence), rejected_output=str(rejected),
                config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
                model_calls=0, resets_per_task=2, diagnostic_steps=5,
                maximum_simulators=2 * len(plan['tasks']), concurrent_simulators=1,
                startup_timeout_seconds=900, active_seconds=120,
                simulator_lifetime_seconds=1020)
    return plan


def _save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(plan, *, factory=None):
    if plan.get('kind') != 'no-model-admission-v1':
        raise ValueError('Expected admission plan')
    if factory is None:
        from .adaptive_transport import episode_factory
        factory = episode_factory
    from .library.serial import serial_spec
    output, evidence, rejected = map(Path, (plan['output'], plan['evidence'], plan['rejected_output']))
    if any(p.exists() for p in (output, evidence, rejected)):
        raise ValueError('Admission output/evidence already exists; refusing overwrite')
    evidence.mkdir(parents=True, exist_ok=False)
    _save(evidence/'plan.json', plan)
    approved, failures = [], []
    stopped = False
    for index, task in enumerate(plan['tasks']):
        row = task['cohort']
        if stopped:
            failures.append({'id':row['id'], 'status':'not_attempted_after_infrastructure_failure'})
            continue
        receipts = []
        for number in (1, 2):
            directory = evidence/f'task-{index:04d}'/f'reset-{number}'
            directory.mkdir(parents=True, exist_ok=False)
            execution = plan['execution']
            spec = serial_spec(row['task'], timeout=120, max_steps=5)
            spec.update(execution_backend=execution['backend'],
                        transport_optimizations='sensor-serialization-cache-v1',
                        sensor_documentation='nested-rgb-v1',
                        simulator_startup_timeout_seconds=900,
                        simulator_lifetime_seconds=1020)
            spec.update({k:v for k,v in execution.items() if k.startswith('modal_')})
            if execution['backend'] == 'modal':
                spec.update(modal_image_id=execution['simulator_image'], artifact_transfer='remote-video')
            else:
                spec['docker_simulator_image'] = execution['simulator_image']
            if row.get('counter_region_reference') is not None:
                spec['counter_region_reference'] = row['counter_region_reference']
            spec['sha256'] = hashlib.sha256(json.dumps({k:v for k,v in spec.items() if k!='sha256'},sort_keys=True).encode()).hexdigest()
            receipt = dict(task=row['task'], config=row['config'], number=number,
                           execution=execution, model_calls=0, steps=0, ok=False)
            runtime = None
            began = time.monotonic()
            try:
                runtime = factory(row['config'], spec, directory, number, [])
                fingerprint = runtime.initial['fingerprint']
                if not isinstance(fingerprint, str) or not fingerprint:
                    raise ValueError('Missing initial fingerprint')
                receipt['initial_fingerprint'] = fingerprint
                initial = runtime.start_task(0)
                receipt['precondition_valid'] = initial.get('precondition_valid') is True
                if not receipt['precondition_valid']:
                    raise ValueError('Initial predicate is already true or precondition unavailable')
                reference = row.get('initial_fingerprint') or (receipts[0]['initial_fingerprint'] if receipts else fingerprint)
                if fingerprint != reference:
                    raise ValueError('Exact initial fingerprint mismatch')
                observation = initial['observation']
                if row['config']['robot'] == 'PandaOmron':
                    action = [0.0] * observation['action_schema']['shape'][0]
                elif row['config']['robot'] == 'GR1':
                    action = observation['whole_body_ik']['current_pose_hold_action']
                else:
                    raise ValueError('No validated diagnostic action for this robot')
                for _ in range(5):
                    result = runtime.bridge({'operation':'step', 'action':action})
                    receipt['steps'] += 1
                    if result['control_step'] != receipt['steps']:
                        raise RuntimeError('Diagnostic control-step receipt mismatch')
                receipt['finish'] = runtime.finish_task(code_errors=0)
                receipt['ok'] = True
            except Exception as exc:
                receipt.update(error_type=type(exc).__name__, error=str(exc)[:1000])
                # No implicit transport or scene retries, including the second reset.
                if not isinstance(exc, ValueError):
                    stopped = True
            finally:
                if runtime is not None:
                    try:
                        receipt['close'] = runtime.close()
                        receipt['close_completed'] = True
                        if execution['backend'] == 'modal' and receipt['close'].get('remote_artifacts',{}).get('persisted_readback_verified') is not True:
                            raise RuntimeError('Full remote artifact readback unconfirmed')
                    except Exception as exc:
                        receipt.update(ok=False, cleanup_error_type=type(exc).__name__,
                                       cleanup_error=str(exc)[:1000])
                        stopped = True
                else:
                    # Constructor cleanup cannot be certified here; stop the batch.
                    stopped = True
                receipt['elapsed_seconds'] = time.monotonic() - began
                _save(directory/'admission-receipt.json', receipt)
                receipts.append(receipt)
            if not receipt['ok']:
                break
        admitted = len(receipts)==2 and all(r['ok'] for r in receipts)
        summary = {'id':row['id'], 'task':row['task'], 'admitted':admitted,
                   'receipts':[str(evidence/f'task-{index:04d}'/f'reset-{r["number"]}'/'admission-receipt.json') for r in receipts]}
        _save(evidence/f'task-{index:04d}'/'result.json', summary)
        if admitted:
            approved.append({**row, 'initial_fingerprint':receipts[0]['initial_fingerprint'],
                             'admission':{'status':'two-independent-resets-five-native-steps',
                                          'execution':execution,'receipt':str(evidence/f'task-{index:04d}'/'result.json')}})
        else:
            failures.append({**summary, 'status':'environment_admission_rejected_not_model_outcome'})
    _save(rejected, {'tasks':failures, 'model_calls':0, 'stopped_on_infrastructure_failure':stopped})
    approved_ids = {row['id'] for row in approved}
    _save(output, {'tasks':approved, 'model_calls':0,
                   'source_task_count':len(plan['tasks']), 'admitted_task_count':len(approved),
                   'rejected_or_unattempted_task_count':len(failures),
                   'source_planned_attempts':plan['planned_attempts'],
                   'admitted_planned_attempts':sum(task['trials'] for task in plan['tasks'] if task['cohort']['id'] in approved_ids),
                   'source_dataset_sha256':plan['tasks'][0]['dataset_sha256'],
                   'image_build_receipt':plan.get('image_build_receipt'), 'admission_plan':str(evidence/'plan.json'),
                   'rejected_report':str(rejected), 'status':'admitted' if not failures else 'partial_or_rejected'})
    return {'output':str(output), 'admitted':len(approved), 'rejected_or_unattempted':len(failures),
            'model_calls':0, 'stopped_on_infrastructure_failure':stopped}
