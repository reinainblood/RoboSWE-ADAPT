"""Portable, explicit adaptive dataset plans and sequential public dispatch.

Planning never resolves secrets, imports simulator packages, or allocates resources.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
from pathlib import Path

PROTOCOL = 'adaptive-public-sqlite-v1'


def read_object(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def build_plan(dataset_path, config_path, *, backend=None, jobs_dir='jobs', images_path=None, include_prompts=True):
    source = Path(dataset_path).resolve()
    if source.is_dir():
        source = source / 'dataset.json'
    manifest = read_object(source)
    config = read_object(config_path)
    if set(config) - {'participant', 'execution', 'limits', 'trials', 'video_judge'}:
        raise ValueError('Unknown run configuration field; credentials belong in environment variables, not config')
    participant = config.get('participant', {})
    allowed = {'provider', 'model', 'context_window', 'max_output_tokens', 'effort', 'reasoning_summary',
               'workspace_id', 'cache_ttl', 'api_key_env', 'upstream_provider', 'context_policy', 'ledger_path', 'spend_limit_usd', 'input_price_per_million', 'output_price_per_million'}
    if set(participant) - allowed:
        raise ValueError('Unknown participant configuration field; never put API keys in config')
    if participant.get('provider') not in {'openai', 'anthropic', 'openrouter', 'google'} or not participant.get('model'):
        raise ValueError('Explicit participant provider and model required')
    context, output = participant.get('context_window'), participant.get('max_output_tokens')
    if type(context) is not int or type(output) is not int or output <= 0 or context <= output + 4096:
        raise ValueError('Explicit valid context_window and max_output_tokens required')
    key_env = participant.get('api_key_env') or {'openai':'OPENAI_API_KEY','anthropic':'ANTHROPIC_API_KEY','openrouter':'OPENROUTER_API_KEY','google':'GEMINI_API_KEY'}[participant['provider']]
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key_env):
        raise ValueError('api_key_env must be an environment variable name')
    participant = {**participant, 'api_key_env':key_env}
    policy=participant.get('context_policy','sqlite-sliding-v1')
    if policy not in ('sqlite-sliding-v1','sqlite-sliding-or-conservative-v1','sqlite-sliding-or-google-count-v2'):raise ValueError('Unsupported context policy')
    if policy!='sqlite-sliding-v1' and participant['provider']!='openrouter':raise ValueError('Approximate policy is OpenRouter-only')
    if policy.startswith('sqlite-sliding-or-') and not all(participant.get(k) is not None for k in ('ledger_path','spend_limit_usd','input_price_per_million','output_price_per_million')):
        raise ValueError('Conservative OR runs require explicit shared ledger, spending cap, and pinned input/output prices')
    execution = dict(config.get('execution', {}))
    if set(execution) - {'backend','simulator_image','code_image','modal_profile','modal_environment','modal_sdk_python','modal_app','modal_volume','artifact_transfer'}:
        raise ValueError('Unknown execution field; named provider secrets require a remote trusted coordinator, not simulator injection')
    image_provenance = None
    if images_path is not None:
        receipt_path = Path(images_path).resolve()
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        fields = {'backend', 'simulator_image', 'code_image', 'modal_profile', 'modal_environment'}
        if not isinstance(receipt, dict) or set(receipt) - fields:
            raise ValueError('Unsupported image receipt fields; only build-images execution metadata is accepted')
        if not {'backend', 'simulator_image', 'code_image'} <= set(receipt):
            raise ValueError('Image receipt requires backend, simulator_image and code_image')
        if any(not isinstance(v, str) or not v for v in receipt.values()):
            raise ValueError('Image receipt fields must be nonempty strings')
        selected_backend = backend or execution.get('backend') or receipt['backend']
        if selected_backend != receipt['backend']:
            raise ValueError('Image receipt backend does not match selected execution backend')
        if receipt['backend'] == 'docker' and set(receipt) & {'modal_profile', 'modal_environment'}:
            raise ValueError('Docker image receipt must not contain Modal settings')
        if receipt['backend'] == 'modal':
            if not {'modal_profile', 'modal_environment'} <= set(receipt):
                raise ValueError('Modal image receipt requires explicit modal_profile and modal_environment')
            for key in ('modal_profile', 'modal_environment'):
                if not re.fullmatch(r'[A-Za-z0-9_.-]+', receipt[key]):
                    raise ValueError('Invalid Modal profile or environment in image receipt')
        # Explicit --images replaces only image-build fields; no arbitrary config merge.
        execution.update(receipt)
        image_provenance = {'path': str(receipt_path),
                            'sha256': hashlib.sha256(receipt_bytes).hexdigest(),
                            'source': 'build-images-receipt'}
    execution['backend'] = backend or execution.get('backend', 'docker')
    if execution['backend'] not in {'docker','modal'}:
        raise ValueError('Environment must be docker or modal')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', execution.get('code_image','')):
        raise ValueError('execution.code_image must be an immutable Docker sha256 image ID')
    image = execution.get('simulator_image','')
    if execution['backend']=='docker' and not re.fullmatch(r'sha256:[0-9a-f]{64}',image):
        raise ValueError('Docker simulator_image must be an immutable sha256 image ID')
    if execution['backend']=='modal':
        execution.setdefault('modal_volume','roboswe-job-artifacts-v1')
        if not isinstance(execution['modal_volume'],str) or not execution['modal_volume'].strip():
            raise ValueError('modal_volume must be a nonempty volume name')
        execution.setdefault('artifact_transfer','remote-video')
        if execution['artifact_transfer']!='remote-video':raise ValueError('Public Modal runs require remote-video full remote hash readback')
        if not re.fullmatch(r'im-[A-Za-z0-9]+', image):
            raise ValueError('Modal simulator_image must be a Modal image ID')
        if not execution.get('modal_environment') or not execution.get('modal_profile'):
            raise ValueError('Explicit modal_environment and modal_profile required (FDR for this workspace)')
    limits = {'active_seconds':1800,'control_steps':6000,'per_script_steps':50,'per_script_seconds':60}
    overrides = config.get('limits', {})
    if set(overrides)-set(limits):raise ValueError('Unknown budget field')
    limits.update(overrides)
    if any(type(v) is not int or v<=0 for v in limits.values()):raise ValueError('Budgets must be positive integers')
    if limits['per_script_steps']>limits['control_steps']:raise ValueError('Script step budget exceeds episode budget')
    trials = config.get('trials',5)
    if type(trials) is not int or not 1<=trials<=20:raise ValueError('trials must be 1..20')
    judge = config.get('video_judge')
    if judge is not None:
        if not isinstance(judge, dict) or set(judge) != {'backend', 'ledger_path', 'spend_limit_usd'}:
            raise ValueError('video_judge requires backend, ledger_path, and spend_limit_usd')
        if judge['backend'] not in {'gemini', 'glimpse'}:
            raise ValueError('Unsupported video judge backend')
        if not isinstance(judge['ledger_path'], str) or not judge['ledger_path']:
            raise ValueError('Explicit video judge ledger_path required')
        cap = judge['spend_limit_usd']
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
            raise ValueError('Video judge spend_limit_usd must be positive and finite')
    rows=manifest.get('tasks')
    if not isinstance(rows,list) or not rows:raise ValueError('Dataset must contain nonempty tasks array')
    ids=set(); plans=[]
    if include_prompts:
        from pairing.cohort50 import prompt_for
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('id'),str) or row['id'] in ids:
            raise ValueError('Every task needs a unique string id')
        ids.add(row['id'])
        task=row.get('task'); scene=row.get('config')
        if not isinstance(task,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',task):raise ValueError('Task must be a RoboCasa class name')
        if not isinstance(scene,dict) or scene.get('task')!='robocasa/'+task or not scene.get('robot'):
            raise ValueError('Every task requires matching config.task and explicit config.robot')
        if type(scene.get('seed')) is not int or not 0<=scene['seed']<2**32:raise ValueError('Explicit scene seed required')
        prompt = None
        if include_prompts:
            prompt=prompt_for(task).replace('2-attempt',f'{trials}-attempt').replace('any compaction summary, ','')
            prompt=prompt.replace('1800-second',f'{limits["active_seconds"]}-second').replace('50-step, 60-second',f'{limits["per_script_steps"]}-step, {limits["per_script_seconds"]}-second')
            prompt=prompt.replace('50-step/60-second',f'{limits["per_script_steps"]}-step/{limits["per_script_seconds"]}-second')
            prompt+='\nExact history is archived in SQLite, never summarized. history_search becomes available only after a sliding-window event.\n'
        plans.append({'protocol':PROTOCOL,'context_policy':policy,'cohort':row,'participant':participant,
            'trials':trials,'limits':limits,'code_image':execution['code_image'],'prompt':prompt,
            'initial_fingerprint':row.get('initial_fingerprint'),'counter_region_reference':row.get('counter_region_reference'),
            'execution':execution,'video_judge':judge,'new_campaign':True,'legacy_resume':False,
            'reset_policy':'Fresh simulator each attempt; exact match to first reset or explicitly pinned fingerprint before model input',
            'budget_source':'Explicit protocol budgets; not inferred upstream defaults',
            'mechanical_grade':'Automatic sustained upstream predicate plus independent code replay; visual audit optional',
            'dataset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            **({'image_build_receipt':image_provenance} if image_provenance else {})})
    return {'protocol':PROTOCOL,'dataset':str(source),'tasks':plans,'jobs_dir':str(Path(jobs_dir).resolve()),
            'planned_attempts':len(plans)*trials,'execution':execution,
            **({'image_build_receipt':image_provenance} if image_provenance else {}),
            'limitations':['OpenRouter conservative context policy is approximate and differs from native-count providers; disclose separately',
                            'No crash resume; no fresh-context control arm',
                            'Modal simulator still uses local trusted coordinator and local Docker code sandbox']}


def run(plan):
    from pairing.adaptive_run import run as run_series
    from roboswe.adaptive_transport import episode_factory
    participant=plan['tasks'][0]['participant']
    provider=participant['provider']
    if provider=='openrouter' and participant.get('context_policy') not in ('sqlite-sliding-or-conservative-v1','sqlite-sliding-or-google-count-v2'):raise ValueError('OpenRouter requires a verified model-aware token counter; no simulation or paid call started')
    canonical={'openai':'OPENAI_API_KEY','anthropic':'ANTHROPIC_API_KEY','openrouter':'OPENROUTER_API_KEY','google':'GEMINI_API_KEY'}[provider]
    value=os.environ.get(participant['api_key_env'])
    if not value:raise ValueError(f'Required trusted coordinator environment variable {participant["api_key_env"]} is absent')
    previous=os.environ.get(canonical)
    os.environ[canonical]=value
    try:
        return [str(run_series(task,jobs_dir=plan['jobs_dir'],episode_factory=episode_factory)) for task in plan['tasks']]
    finally:
        if previous is None:os.environ.pop(canonical,None)
        else:os.environ[canonical]=previous
