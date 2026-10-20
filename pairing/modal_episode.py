"""Trusted Modal simulator transport; generated Python stays in CodeSandbox.

Run image builds / SDK proxy with the separate Modal tool Python, avoiding the
simulation environment's legacy protobuf dependency. All resources are FDR.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

ENVIRONMENT = 'FDR'
APP = 'roboswe-parallel-trials'
VOLUME = 'roboswe-job-artifacts-v1'
SDK_PYTHON = os.environ.get('ROBOSWE_MODAL_PYTHON', str(Path.home()/'.local/share/uv/tools/modal/bin/python'))
BASE_IMAGE = 'im-JkBCuOs89zM0zHyeU7CVLz'


def cleanup_startup_sandbox(directory, *, run=subprocess.run, environment=ENVIRONMENT, profile="snorkelai", sdk_python=SDK_PYTHON):
    """Terminate only the recorded owned sandbox after its proxy has exited."""
    directory=Path(directory)
    session=directory/'modal-session.json'
    if not session.exists():return
    metadata=json.loads(session.read_text())
    if metadata.get('environment')!=environment or not metadata.get('sandbox_id','').startswith('sb-'):
        raise ValueError('Invalid owned sandbox cleanup metadata')
    script='import modal,json,sys; s=modal.Sandbox.from_id(sys.argv[1]); before=s.poll(); s.terminate(wait=True) if before is None else None; print(json.dumps({"exit_before":before,"exit_after":s.poll()}))'
    result=run([sdk_python,'-c',script,metadata['sandbox_id']],capture_output=True,text=True,timeout=60,
               env={**os.environ,'MODAL_ENVIRONMENT':environment,**({'MODAL_PROFILE':profile} if profile else {})})
    if result.returncode:raise RuntimeError('Owned sandbox cleanup failed; inspect Modal state')
    receipt={'sandbox_id':metadata['sandbox_id'],'environment':environment,**json.loads(result.stdout)}
    (directory/'startup-cleanup.json').write_text(json.dumps(receipt,indent=2))


def build(receipt_path='artifacts/modal-trials-image.json'):
    import modal
    root=Path.cwd()
    app=modal.App.lookup(APP,environment_name=ENVIRONMENT,create_if_missing=True)
    image=modal.Image.from_id(BASE_IMAGE).pip_install_from_requirements(str(root/'robot_session/requirements.txt'))
    for name in ('robot_session','roboswe','pairing'):
        image=image.add_local_dir(root/name,'/app/'+name,copy=True,
            ignore=lambda p: Path(p).suffix not in {'.py','.json','.txt'})
    for name in ('robocasa','robosuite'):
        image=image.add_local_dir(root/'upstream'/name,'/opt/'+name,copy=True,
            ignore=lambda p: Path(p).suffix not in {'.py','.json','.xml','.yaml','.yml'})
    image=image.env({'PYTHONHASHSEED':'0','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
        'LP_NUM_THREADS':'4','MUJOCO_GL':'osmesa','PYOPENGL_PLATFORM':'osmesa','PYNPUT_BACKEND':'dummy'})
    with modal.enable_output(): built=image.build(app)
    receipt={'image':built.object_id,'base_image':BASE_IMAGE,'app_id':app.app_id,'environment':ENVIRONMENT,
        'cpu':4,'memory_mb':16384,'gpu':'L4','backend':'Modal trusted simulator; local isolated Docker code execution'}
    Path(receipt_path).write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


class RemoteEpisode:
    """FreshEpisode-compatible coordinator, bridged through an SDK subprocess."""
    def __init__(self,config,spec,directory,number,previous,*,worker=None):
        directory=Path(directory).resolve()
        environment=spec.get('modal_environment', ENVIRONMENT)
        profile=spec.get('modal_profile', 'snorkelai')
        sdk_python=spec.get('modal_sdk_python', SDK_PYTHON)
        image=spec.get('modal_image_id') or json.loads(Path('artifacts/modal-trials-image.json').read_text())['image']
        worker_name='toast' if worker and worker.__name__=='toast_worker' else 'suite'
        payload={'config':config,'spec':spec,'directory':str(directory),'number':number,'previous':previous,
            'environment':environment,'app':spec.get('modal_app',APP),'volume':spec.get('modal_volume',VOLUME),'image':image,'worker':worker_name,'lifetime':int(spec.get('simulator_lifetime_seconds',spec['task_timeout_seconds']+1800)),
            'artifact_transfer':spec.get('artifact_transfer','all')}
        self.log=(directory/'modal-transport.log').open('ab')
        self.proc=subprocess.Popen([sdk_python,'-u','-m','pairing.modal_episode','proxy'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,
            env={**os.environ,'MODAL_ENVIRONMENT':environment,**({'MODAL_PROFILE':profile} if profile else {})})
        self.incoming=queue.Queue()
        def reader():
            try:
                for line in self.proc.stdout:self.incoming.put(json.loads(line))
            except (OSError, ValueError) as exc: self.incoming.put({'error':f'Modal transport reader: {exc}'})
            finally:self.incoming.put({'error':'Modal transport exited'})
        threading.Thread(target=reader,daemon=True).start()
        self.proc.stdin.write(json.dumps(payload)+'\n');self.proc.stdin.flush()
        self.timeout=spec['task_timeout_seconds'];self.started=None
        self.passive_wait_steps=0;self.has_stepped=False
        try:self.initial=self._receive(int(spec.get('simulator_startup_timeout_seconds',1200)))
        except BaseException:
            try:self.proc.stdin.close()
            except OSError:pass  # The proxy may already have closed its pipe.
            try:self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=30)
            self.log.close()
            try:cleanup_startup_sandbox(directory, environment=environment, profile=profile, sdk_python=sdk_python)
            except Exception as cleanup_error:
                (directory/'startup-cleanup-error.json').write_text(json.dumps({'error_type':type(cleanup_error).__name__,'error':str(cleanup_error)}))
            raise
        self.action_digest=SimpleNamespace(hexdigest=lambda:self.call('digest'))
    def _receive(self,timeout=180):
        try:reply=self.incoming.get(timeout=timeout)
        except queue.Empty:raise TimeoutError('Modal simulator response deadline exceeded') from None
        if 'error' in reply:
            cls={'ValueError':ValueError,'TimeoutError':TimeoutError}.get(reply.get('error_type'),RuntimeError)
            raise cls(reply['error'])
        return reply
    def call(self,operation,**args):
        self.proc.stdin.write(json.dumps({'operation':operation,**args})+'\n');self.proc.stdin.flush()
        return self._receive(900 if operation=='close' else 180)['ok']
    def metadata(self):return self.initial['ok']
    def start_task(self,index):
        if index!=0:raise ValueError('One episode per sandbox')
        self.started=time.monotonic();return self.call('start_task')
    def observe(self):return self.call('observe')
    def bridge(self,message):
        result=self.call('bridge',message=message)
        if message.get('operation')=='step':self.has_stepped=True
        return result
    def wait(self,steps):
        if not self.has_stepped:raise ValueError('Execute a native step before waiting')
        began=time.monotonic()
        self.proc.stdin.write(json.dumps({'operation':'wait','steps':steps})+'\n');self.proc.stdin.flush()
        result=self._receive(900)['ok'];self.started+=time.monotonic()-began
        self.passive_wait_steps=result['passive_wait']['episode_wait_steps'];return result
    def finish_task(self,*,code_errors):return self.call('finish_task',code_errors=code_errors)
    def close(self):
        try:return self.call('close')
        finally:
            try:self.proc.stdin.close()
            except OSError:pass  # Preserve the original transport error and reap.
            try:self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=30)
            self.log.close()


def remote_worker():
    """Only trusted coordinator requests enter this process, never solver source."""
    wire=sys.stdout
    sys.stdout=sys.stderr
    payload=json.loads(sys.stdin.readline())
    directory=Path(payload['remote_directory']);directory.mkdir(parents=True,exist_ok=True)
    from robot_session.session import RobotSession
    original_init=RobotSession.__init__
    def audited_init(session,*args,**kwargs):
        original_init(session,*args,**kwargs)
        (directory/'initial-snapshot.zip').write_bytes(session.snapshot())
    RobotSession.__init__=audited_init
    def serial(value):
        if hasattr(value,'tolist'):return value.tolist()
        raise TypeError(type(value).__name__)
    class Pipe:
        def send(self,value):
            wire.write(json.dumps(value,default=serial,allow_nan=False)+'\n');wire.flush()
        def recv(self):return json.loads(sys.stdin.readline())
        def close(self):pass
    if payload['worker']=='toast':
        from pairing.run_qwen_trials import toast_worker as worker
    else:
        from pairing.overnight_suite import suite_worker as worker
    worker(Pipe(),payload['config'],payload['spec'],directory,payload['number'],payload['previous'])


def verify_volume_files(volume, remote, files, *, workers=4, on_verified=None):
    """Read every persisted byte, with bounded parallelism across independent files."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    if not 1<=workers<=4:raise ValueError('Readback concurrency must be between one and four')
    def verify(item):
        began=time.monotonic()
        for attempt in range(3):
            digest=hashlib.sha256();size=0
            for chunk in volume.read_file(remote.rstrip('/')+'/'+item['path']):
                digest.update(chunk);size+=len(chunk)
            if size==item['bytes'] and digest.hexdigest()==item['sha256']:break
            if attempt==2:
                raise RuntimeError('Persisted Modal artifact readback mismatch: '+item['path'])
            # Re-read committed bytes against the ORIGINAL manifest; never bless
            # a new hash or rerun an episode to recover delayed Volume visibility.
            time.sleep((2,5)[attempt])
        receipt={'path':item['path'],'bytes':size,'seconds':time.monotonic()-began,'readback_attempts':attempt+1}
        if on_verified:on_verified(receipt)
        return receipt
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(verify,files))


def proxy():
    import traceback

    import modal
    payload=json.loads(sys.stdin.readline())
    if payload['spec'].get('diagnostic_cpu'):payload['lifetime']=int(payload['spec'].get('simulator_lifetime_seconds',300))
    environment=payload.get('environment',ENVIRONMENT)
    app_name=payload.get('app',APP)
    volume_name=payload.get('volume',VOLUME)
    local=Path(payload['directory'])
    remote='/evidence/active/'+local.name+'/'+uuid4().hex
    payload['remote_directory']=remote
    app=modal.App.lookup(app_name,environment_name=environment,create_if_missing=True)
    volume=modal.Volume.from_name(volume_name,environment_name=environment,create_if_missing=True)
    sb=None
    try:
        def phase(name,**details):
            sys.stderr.write(json.dumps({'transport_phase':name,'epoch':time.time(),**details})+'\n');sys.stderr.flush()
        phase('create_start')
        for creation_try in range(6):
            try:
                sb=modal.Sandbox.create('sleep',str(payload['lifetime']),app=app,
                    image=modal.Image.from_id(payload['image']),cpu=4,memory=16384,gpu=None if payload['spec'].get('diagnostic_cpu') else 'L4',timeout=payload['lifetime'],
                    env={'PYTHONHASHSEED':str(payload['config'].get('seed',0)),'LP_NUM_THREADS':'4','MUJOCO_GL':'osmesa' if payload['spec'].get('diagnostic_cpu') else 'egl','PYOPENGL_PLATFORM':'osmesa' if payload['spec'].get('diagnostic_cpu') else 'egl'},volumes={'/evidence':volume},
                    workdir='/app',block_network=True)
                break
            except modal.exception.ResourceExhaustedError as exc:
                if 'creation rate limit' not in str(exc).lower() or creation_try==5:raise
                time.sleep(min(30,2**(creation_try+1)))

        phase('created')
        receipt={'sandbox_id':sb.object_id,'app_id':app.app_id,'image':payload['image'],
            'environment':environment,'volume':volume_name,'remote_directory':remote,'cpu':4,'memory_mb':16384,'gpu':None if payload['spec'].get('diagnostic_cpu') else 'L4'}
        (local/'modal-session.json').write_text(json.dumps(receipt,indent=2))
        phase('worker_exec_start')
        if payload['spec'].get('capture_region_mismatch'):
            diagnostic_source = """import os,json
from pathlib import Path
import pairing.reset_region_order as regions
original=regions.reorder_regions
def audited(actual,expected,**kwargs):
    try:return original(actual,expected,**kwargs)
    except ValueError:
        Path(os.environ['REGION_DIAG_FILE']).write_text(json.dumps({'actual':regions.plain(actual),'expected':expected},indent=2))
        raise
regions.reorder_regions=audited
from pairing.modal_episode import remote_worker
remote_worker()
"""
            process=sb.exec('python','-u','-c',diagnostic_source,env={'REGION_DIAG_FILE':remote+'/region-mismatch.json'},bufsize=1,timeout=payload['lifetime'])
        else:
            process=sb.exec('python','-u','-m','pairing.modal_episode','worker',bufsize=1,timeout=payload['lifetime'])
        phase('worker_exec_ready')
        def logs():
            for line in process.stderr:sys.stderr.write(line);sys.stderr.flush()
        threading.Thread(target=logs,daemon=True).start()
        phase('worker_init_send_start')
        process.stdin.write(json.dumps(payload)+'\n');process.stdin.drain()
        phase('worker_init_send_complete')
        responses=iter(process.stdout)
        first=json.loads(next(responses))
        phase('worker_initial_response')
        print(json.dumps(first),flush=True)
        if 'error' in first:return
        for line in sys.stdin:
            request=json.loads(line)
            process.stdin.write(line);process.stdin.drain()
            reply=json.loads(next(responses))
            if request['operation']=='close' and 'ok' in reply:
                process.stdin.write_eof(); process.stdin.drain(); process.wait()
                if payload.get('artifact_transfer') == 'remote-video':
                    # Keep large videos on the campaign volume. Publish a hash manifest,
                    # terminate this owned sandbox, then verify persisted bytes via Volume.
                    import hashlib
                    listing=sb.exec('python','-c',
                        'import pathlib,json,hashlib,sys; r=pathlib.Path(sys.argv[1]); files=[]; '
                        '\nfor p in r.rglob("*"):\n if p.is_file():\n  h=hashlib.sha256(); size=0\n  with p.open("rb") as f:\n   for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk); size+=len(chunk)\n  files.append({"path":str(p.relative_to(r)),"bytes":size,"sha256":h.hexdigest()})'
                        '\n(r/"remote-artifacts.json").write_text(json.dumps({"files":files}))\nprint(json.dumps(files))',remote)
                    files=json.loads(listing.stdout.read());listing.wait()
                    for item in files:
                        name=item['path']
                        if name.endswith('.mp4') and not payload['spec'].get('retain_local_videos',False):continue
                        target=local/name
                        if target.exists():raise RuntimeError('Refusing to overwrite local artifact '+name)
                        target.parent.mkdir(parents=True,exist_ok=True)
                        sb.filesystem.copy_to_local(remote+'/'+name,str(target))
                        if hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('Downloaded artifact hash mismatch')
                    phase('sandbox_commit_start')
                    sb.terminate(wait=True);sb=None
                    phase('sandbox_commit_complete')
                    phase('artifact_readback_start',files=len(files),bytes=sum(item['bytes'] for item in files))
                    readbacks=verify_volume_files(volume,remote.removeprefix('/evidence'),files,on_verified=lambda receipt:phase('artifact_file_verified',**receipt))
                    phase('artifact_readback_complete')
                    artifact_receipt={'environment':environment,'volume':volume_name,'directory':remote.removeprefix('/evidence'),'files':files,'persisted_readback_verified':True,'readback_concurrency':4,'readback_timings':readbacks}
                    (local/'remote-artifacts.json').write_text(json.dumps(artifact_receipt,indent=2))
                    reply['ok']['remote_artifacts']=artifact_receipt
                    reply['ok']['recording_dir']=str(local/Path(reply['ok']['recording_dir']).relative_to(remote))
                    print(json.dumps(reply),flush=True)
                    break
                # Read only this job's simulator artifacts; never overwrite host transcripts.
                listing=sb.exec('python','-c',
                    'import pathlib,json,sys; r=pathlib.Path(sys.argv[1]); print(json.dumps([str(p.relative_to(r)) for p in r.rglob("*") if p.is_file()]))',remote)
                names=json.loads(listing.stdout.read());listing.wait()
                for name in names:
                    target=local/name
                    if target.exists():raise RuntimeError('Refusing to overwrite local artifact '+name)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    sb.filesystem.copy_to_local(remote+'/'+name, str(target))
                reply['ok']['recording_dir']=str(local/Path(reply['ok']['recording_dir']).relative_to(remote))
                print(json.dumps(reply),flush=True)
                break
            print(json.dumps(reply),flush=True)
    except BaseException:  # noqa: BLE001 -- Return transport failure and always terminate this owned sandbox.
        print(json.dumps({'error':traceback.format_exc()}),flush=True)
    finally:
        if sb:sb.terminate(wait=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('build','proxy','worker'))
    args=parser.parse_args()
    {'build':build,'proxy':proxy,'worker':remote_worker}[args.mode]()


if __name__=='__main__':main()
