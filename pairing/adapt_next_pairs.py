"""Two explicitly planned task pairs after an owned predecessor; never retry a lane."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from pairing.adapt_campaign import write
from pairing.adapt_followon import live_simulators
from pairing.adapt_modal_pair import classify, pid_live

PROVIDERS = ('openrouter', 'anthropic')
TASKS = {'beverage-sorting': 'BeverageSorting', 'date-night': 'DateNight'}
PINS = {
    'openrouter': {'provider': 'openrouter', 'model': 'google/gemini-4-argon',
        'context_window': 1048576, 'max_output_tokens': 32768, 'effort': 'high',
        'context_policy': 'sqlite-sliding-or-google-count-v2',
        'upstream_provider': 'google-vertex/global', 'spend_limit_usd': 150},
    'anthropic': {'provider': 'anthropic', 'model': 'claude-opus-5-5',
        'context_window': 1000000, 'max_output_tokens': 32768, 'effort': 'high',
        'spend_limit_usd': 100},
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(path, campaign_root):
    pairs = read(path).get('pairs', [])
    if not 1 <= len(pairs) <= 2 or len({p['id'] for p in pairs}) != len(pairs):
        raise ValueError('One or two distinct explicitly planned pairs required')
    hashes = {str(Path(path).resolve()): digest(path)}
    for pair in pairs:
        if pair['id'] not in TASKS or set(pair['configs']) != set(PROVIDERS):
            raise ValueError('Only BeverageSorting and DateNight with both pinned providers')
        paths = [pair['dataset'], *pair['configs'].values()]
        if any(not Path(p).is_absolute() for p in paths):
            raise ValueError('Plan paths must be absolute')
        tasks = read(pair['dataset']).get('tasks', [])
        if len(tasks) != 1 or tasks[0].get('task') != TASKS[pair['id']]:
            raise ValueError('Each dataset must contain exactly its named native task')
        if tasks[0].get('config', {}).get('task') != 'robocasa/' + TASKS[pair['id']]:
            raise ValueError('Native simulator task mismatch')
        if not tasks[0].get('initial_fingerprint'):
            raise ValueError('Admitted initial fingerprint required')
        configs = {p: read(pair['configs'][p]) for p in PROVIDERS}
        left, right = configs.values()
        for field in ('execution', 'limits', 'trials'):
            if left[field] != right[field]:
                raise ValueError('Paired execution, limits, and trial counts must match')
        if left['trials'] != 5:
            raise ValueError('Exactly five serial trials per provider required')
        execution = left['execution']
        if execution.get('backend') != 'modal' or execution.get('modal_environment') != 'FDR' or execution.get('modal_profile') != 'snorkelai':
            raise ValueError('Explicit Modal FDR on snorkelai required')
        if execution.get('artifact_transfer') != 'remote-video':
            raise ValueError('Remote-video durability required')
        for provider, config in configs.items():
            participant = config['participant']
            if any(participant.get(k) != v for k, v in PINS[provider].items()):
                raise ValueError('Provider/model/context/budget pin changed: ' + provider)
            if provider == 'anthropic' and (participant.get('context_policy', 'sqlite-sliding-v1') != 'sqlite-sliding-v1' or participant.get('cache_ttl', '5m') != '5m'):
                raise ValueError('Anthropic native-count policy changed')
            if Path(participant.get('ledger_path', '')).resolve() != (campaign_root / (provider + '-spend.json')).resolve():
                raise ValueError('Existing campaign ledger required; no fresh budget')
            if config.get('video_judge') is not None:
                raise ValueError('No automatic paid video judge in this queue')
        hashes.update({p: digest(p) for p in paths})
    return pairs, hashes


def validate_series(series, jobs_dir):
    series = Path(series)
    parent = read(series / 'job.json')
    report = read(series / 'report.json')
    if parent.get('status') != 'completed_unscored' or not parent.get('finished_at') or parent.get('artifact_errors') or len(report.get('trials', [])) != 5:
        raise ValueError('Parent is not a finalized complete five-trial series')
    children = []
    for path in Path(jobs_dir).glob('*/job.json'):
        child = read(path)
        if child.get('campaign_id') == parent['id'] and child.get('kind') == 'adaptive-direct-attempt':
            children.append((path.parent, child))
    if len(children) != 5 or {j.get('attempt_number') for _, j in children} != set(range(1, 6)):
        raise ValueError('Exactly five unique child trial receipts required')
    for directory, child in children:
        if child.get('status') not in {'verified_pass', 'verified_fail'} or not child.get('finished_at') or child.get('artifact_errors'):
            raise ValueError('Child is not a verified terminal artifact-valid trial')
        verification = read(directory / 'verification.json')
        trials = verification.get('trials', [])
        if verification.get('kind') != 'independent_code_replay' or verification.get('reproduced') is not True or verification.get('infrastructure_valid') is False or verification.get('mismatches') or len(trials) != 1:
            raise ValueError('Missing independent reproduced replay')
        t = trials[0]
        if t.get('reproduced') is not True or t.get('infrastructure_valid') is not True or t.get('precondition_valid') is not True or t.get('replay', {}).get('precondition_valid') is not True:
            raise ValueError('Replay or task precondition invalid')
        expected = child['status'] == 'verified_pass'
        if any(v is not expected for v in [verification.get('verified_success'), t.get('verified_success'), t.get('observed_success'), t.get('replay', {}).get('observed_success')]):
            raise ValueError('Contradictory verified success receipts')
    return parent['id']


def lane_result(provider, lane, code, jobs_dir):
    try:
        receipt = read(lane / 'active.json')
        status = receipt.get('child_status', receipt.get('status'))
        category = ('provider_blocked' if status in {'model_refusal', 'provider_error', 'model_error', 'budget_exhausted', 'funding_blocked'} else classify(status, receipt.get('error', '')))
        result = {'provider': provider, 'status': status, 'classification': category,
                  'returncode': code, 'receipt': str(lane / 'active.json'), 'completed': receipt.get('completed', [])}
        if not receipt.get('finished_epoch'):
            raise ValueError('Lane has no terminal receipt')
        if category == 'completed':
            if code != 0 or len(result['completed']) != 1:
                raise ValueError('Completed lane returned incorrect series count or exit code')
            validate_series(result['completed'][0], jobs_dir)
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'provider': provider, 'returncode': code, 'classification': 'blocked_infrastructure',
                'status': 'invalid_receipt', 'validation_error': str(exc)}


def predecessor_blocked(receipt_path, pid, jobs_dir):
    receipt = read(receipt_path)
    if receipt.get('pid') != pid or not receipt.get('finished_epoch') or receipt.get('cleanup_pending'):
        raise ValueError('Predecessor owned PID and terminal cleanup receipt required')
    if receipt.get('status') not in {'completed', 'partial_stopped', 'partial_provider_blocked'}:
        raise ValueError('Predecessor stopped on infrastructure or control')
    if any(pid_live(c['pid']) for c in receipt.get('children', [])):
        raise ValueError('Predecessor still has live child processes')
    results = receipt.get('results', [])
    if len(results) != 2 or {r.get('provider') for r in results} != set(PROVIDERS):
        raise ValueError('Predecessor needs both provider results')
    blocked = set()
    for result in results:
        if result.get('classification') == 'provider_blocked':
            blocked.add(result['provider'])
        elif result.get('classification') == 'completed':
            completed = result.get('completed', [])
            if len(completed) != 1:
                raise ValueError('Predecessor completed series missing')
            validate_series(completed[0], jobs_dir)
        else:
            raise ValueError('Predecessor infrastructure/verifier/artifact failure')
    return blocked


def stop_requested(folder, root):
    return (folder / 'STOP').exists() or (root / 'STOP').exists()


def child_command(pair, provider, lane, jobs_dir):
    return [sys.executable, '-u', '-m', 'pairing.adapt_campaign', '--dataset', pair['dataset'],
            '--config', pair['configs'][provider], '--state-dir', str(lane), '--max-series', '1', '--jobs-dir', str(jobs_dir)]


def cleanup(children):
    pending = []
    for provider, process in children:
        if process.poll() is None:
            # SIGINT invokes adapt_campaign/adaptive_run finally cleanup; never SIGKILL sandboxes.
            process.send_signal(signal.SIGINT)
    for provider, process in children:
        if process.poll() is None:
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                pending.append({'provider': provider, 'pid': process.pid})
    return pending


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'state-dir', 'campaign-root', 'predecessor-state'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--predecessor-pid', type=int, required=True)
    parser.add_argument('--jobs-dir', default='jobs')
    parser.add_argument('--wait-hours', type=float, default=12)
    args = parser.parse_args(argv)
    if args.predecessor_pid <= 0 or not 0 < args.wait_hours <= 48:
        raise ValueError('Positive predecessor PID and bounded wait of at most 48 hours required')
    folder, root, jobs = Path(args.state_dir).resolve(), Path(args.campaign_root).resolve(), Path(args.jobs_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        owned = stack.enter_context((folder / 'owned.lock').open('a+'))
        fcntl.flock(owned, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'active.json').exists():
            raise RuntimeError('Existing queue receipt; no implicit restart')
        pairs, hashes = validate_plan(args.plan, root)
        state = {'pid': os.getpid(), 'status': 'waiting_for_predecessor', 'started_epoch': time.time(),
                 'predecessor_pid': args.predecessor_pid, 'predecessor_state': args.predecessor_state,
                 'input_sha256': hashes, 'results': [], 'children': [], 'max_task_pairs': len(pairs),
                 'max_parallel_provider_series': 2, 'trials_per_provider': 5, 'retry_policy': 'none'}
        children = []
        write(folder / 'active.json', state)
        try:
            deadline = time.monotonic() + args.wait_hours * 3600
            while pid_live(args.predecessor_pid):
                if stop_requested(folder, root):
                    raise RuntimeError('STOP before launch')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Predecessor wait deadline reached; no launch')
                time.sleep(10)
            blocked = predecessor_blocked(args.predecessor_state, args.predecessor_pid, jobs)
            global_guard = stack.enter_context((root / 'global-campaign.lock').open('a+'))
            fcntl.flock(global_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if live_simulators():
                raise RuntimeError('Unexpected live local simulator')
            for pair in pairs:
                if stop_requested(folder, root):
                    raise RuntimeError('STOP at task boundary')
                if any(digest(p) != sha for p, sha in hashes.items()):
                    raise RuntimeError('Frozen queue inputs changed; no launch')
                current = []
                state.update(status='launching_pair', current_pair=pair['id'])
                # Refuse reuse before launching either provider.
                for provider in PROVIDERS:
                    if (folder / pair['id'] / provider).exists():
                        raise RuntimeError('Existing lane directory; no reuse or retry')
                for provider in PROVIDERS:
                    if provider in blocked:
                        state['results'].append({'pair': pair['id'], 'provider': provider, 'classification': 'provider_blocked', 'status': 'skipped_provider_blocked'})
                        continue
                    if stop_requested(folder, root):
                        raise RuntimeError('STOP immediately before paid launch')
                    lane = folder / pair['id'] / provider
                    lane.mkdir(parents=True)
                    env = {**os.environ, 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
                    if provider == 'openrouter' and not env.get('GEMINI_API_KEY'):
                        key = subprocess.run(['/usr/bin/security', 'find-generic-password', '-a', 'GEMINI_API_KEY', '-w'], capture_output=True, timeout=20)
                        if key.returncode or not key.stdout.strip():
                            raise RuntimeError('Count-only credential unavailable; output suppressed')
                        env['GEMINI_API_KEY'] = key.stdout.decode().strip()
                    with (lane / 'supervisor.log').open('ab') as log:
                        proc = subprocess.Popen(child_command(pair, provider, lane, jobs), stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
                    children.append((provider, proc)); current.append((provider, proc, lane))
                    state['children'].append({'pair': pair['id'], 'provider': provider, 'pid': proc.pid, 'lane': str(lane)})
                    write(folder / 'active.json', state)
                state['status'] = 'running_pair'; write(folder / 'active.json', state)
                while current:
                    if stop_requested(folder, root):
                        raise RuntimeError('STOP during active pair')
                    for provider, proc, lane in list(current):
                        code = proc.poll()
                        if code is None:
                            continue
                        result = lane_result(provider, lane, code, jobs)
                        result['pair'] = pair['id']; state['results'].append(result)
                        current.remove((provider, proc, lane)); write(folder / 'active.json', state)
                        if result['classification'] == 'blocked_infrastructure':
                            raise RuntimeError('Infrastructure/verifier/artifact failure; stopping all owned lanes')
                        if result['classification'] == 'provider_blocked':
                            blocked.add(provider)
                    if current:
                        time.sleep(10)
            state['status'] = 'completed' if not blocked else 'partial_provider_blocked'
        except BaseException as exc:
            state.update(status='stopped', error_type=type(exc).__name__, error=str(exc)[:1500])
        finally:
            pending = cleanup(children)
            if pending:
                state.update(status='cleanup_pending', cleanup_pending=pending, finished_epoch=None, supervisor_finished_epoch=time.time())
            else:
                state['finished_epoch'] = time.time()
            write(folder / 'active.json', state)
    return state


if __name__ == '__main__':
    main()
