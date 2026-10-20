"""Portable simulator transports for the public adaptive runner.

Provider credentials remain in the trusted host coordinator. Neither simulator
container nor authored-code container receives provider secrets.
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from pairing.modal_episode import RemoteEpisode
from roboswe.execution.sandbox import docker_path


def docker_command(config, spec, directory, name):
    image = spec.get('docker_simulator_image', '')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('docker_simulator_image must be an immutable sha256 Docker image ID')
    # Trusted simulator root has an ephemeral writable layer: upstream creates
    # generated object XML beside its assets. Only evidence is host-mounted. No source tree,
    # credentials, Docker socket, or authored-code workspace is mounted.
    return [docker_path(), 'run', '--rm', '-i', '--name', name,
            '--network', 'none', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges',
            '--tmpfs', '/tmp:rw,nosuid,size=2g',
            '--cpus', str(spec.get('simulator_cpus', 4)),
            '--memory', str(spec.get('simulator_memory_mb', 16384)) + 'm',
            '--mount', f'type=bind,source={Path(directory).resolve()},target=/evidence/job',
            '-e', 'PYTHONHASHSEED=' + str(config.get('seed', 0)),
            '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1',
            '-e', 'MUJOCO_GL=osmesa', '-e', 'PYOPENGL_PLATFORM=osmesa',
            '-e', 'PYNPUT_BACKEND=dummy', '-e', 'HOME=/tmp',
            '--workdir', '/app', image,
            'python', '-u', '-m', 'pairing.modal_episode', 'worker']


class DockerEpisode(RemoteEpisode):
    """Fresh isolated simulator container, JSON pipe, local durable evidence."""
    def __init__(self, config, spec, directory, number, previous, *, worker=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.name = 'roboswe-simulator-' + uuid4().hex
        command = docker_command(config, spec, self.directory, self.name)
        self.docker = command[0]
        self.log = (self.directory / 'docker-simulator.log').open('ab')
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log,
                                     text=True, bufsize=1)
        self.incoming = queue.Queue()
        def reader():
            try:
                for line in self.proc.stdout:
                    self.incoming.put(json.loads(line))
            except (OSError, ValueError) as exc:
                self.incoming.put({'error': f'Docker simulator protocol error: {exc}'})
            finally:
                self.incoming.put({'error': 'Docker simulator exited'})
        threading.Thread(target=reader, daemon=True).start()
        payload = dict(config=config, spec=spec, number=number, previous=previous,
                       remote_directory='/evidence/job',
                       worker='toast' if worker and worker.__name__ == 'toast_worker' else 'suite')
        self.timeout = spec['task_timeout_seconds']
        self.started = None
        self.passive_wait_steps = 0
        self.has_stepped = False
        (self.directory / 'docker-session.json').write_text(json.dumps({
            'container_name': self.name, 'image': spec['docker_simulator_image'],
            'network': 'none', 'simulator_backend': 'docker',
            'provider_secrets_in_simulator': False}, indent=2))
        try:
            self.proc.stdin.write(json.dumps(payload) + '\n')
            self.proc.stdin.flush()
            self.initial = self._receive(spec.get('simulator_startup_timeout_seconds', 1200))
        except BaseException:
            self._cleanup()
            raise
        self.action_digest = SimpleNamespace(hexdigest=lambda: self.call('digest'))

    def _cleanup(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            subprocess.run([self.docker, 'rm', '-f', self.name],
                           capture_output=True, timeout=30, check=True)
            self.proc.wait(timeout=30)
        finally:
            self.log.close()

    def close(self):
        try:
            result = self.call('close')
            recording = Path(result['recording_dir']).relative_to('/evidence/job')
            result['recording_dir'] = str(self.directory / recording)
            return result
        finally:
            self._cleanup()


def episode_factory(config, spec, directory, number, previous, *, worker=None):
    """FreshEpisode-compatible factory, selected by spec.execution_backend."""
    backend = spec.get('execution_backend')
    if backend == 'docker':
        return DockerEpisode(config, spec, directory, number, previous, worker=worker)
    if backend == 'modal':
        if not spec.get('modal_image_id'):
            raise ValueError('modal_image_id is required; build/provide your own simulator image')
        public_spec = dict(spec)
        public_spec.setdefault('artifact_transfer','remote-video')
        public_spec['retain_local_videos']=True  # Public viewer/judge still consumes local playback files.
        if public_spec['artifact_transfer']!='remote-video':
            raise ValueError('Public Modal transport requires verified remote-video artifacts')
        # Public path never falls back to the author's private Modal profile or
        # installation. An absent profile uses the user's standard Modal login.
        public_spec.setdefault('modal_profile', os.environ.get('MODAL_PROFILE'))
        public_spec.setdefault('modal_environment', 'FDR')
        public_spec.setdefault('modal_sdk_python', sys.executable)
        return RemoteEpisode(config, public_spec, directory, number, previous, worker=worker)
    raise ValueError('execution_backend must be docker or modal')


class ArtifactDurabilityError(RuntimeError):
    status = 'artifact_durability_error'


def archive_finalized_attempt(directory, execution):
    """Future Modal runs archive SQLite after finalization, using pinned auth."""
    if execution.get('backend') != 'modal':return None
    profile=execution.get('modal_profile');environment=execution.get('modal_environment');volume=execution.get('modal_volume')
    if not all(isinstance(v,str) and v.strip() for v in (profile,environment,volume)):
        raise ArtifactDurabilityError('ADAPT artifact archiver requires explicit profile, environment and volume')
    try:
        result=subprocess.run([execution.get('modal_sdk_python',sys.executable),'-m',
            'pairing.adapt_artifact_archive',str(Path(directory).resolve()),'--profile',profile,'--environment',environment,'--volume',volume],
            env={**os.environ,'MODAL_ENVIRONMENT':environment,'MODAL_PROFILE':profile},
            capture_output=True,text=True,timeout=300)
        if result.returncode:
            (Path(directory)/'artifact-archive-error.txt').write_text(result.stderr[-8000:])
            raise ArtifactDurabilityError('Finalized Modal evidence archive failed; preserved locally; no model retry')
        receipt=json.loads(result.stdout)
        expected={'job_id':Path(directory).resolve().name,'scope':'finalized-job',
                  'environment':environment,'modal_profile':profile,'volume':volume}
        archive=receipt.get('archive',{}) if isinstance(receipt,dict) else {}
        if (not isinstance(receipt,dict) or any(receipt.get(k)!=v for k,v in expected.items())
            or receipt.get('remote_readback_verified') is not True or receipt.get('restore_test_passed') is not True
            or not isinstance(archive,dict) or not re.fullmatch(r'[0-9a-f]{64}',str(archive.get('sha256','')))
            or type(archive.get('bytes')) is not int or archive['bytes']<=0):
            raise ArtifactDurabilityError('Finalized evidence archive verification unconfirmed')
        return receipt
    except (subprocess.TimeoutExpired,json.JSONDecodeError) as exc:
        raise ArtifactDurabilityError('Artifact archive outcome uncertain; preserve local files and inspect before retry') from exc
