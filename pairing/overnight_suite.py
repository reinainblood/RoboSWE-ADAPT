"""Time-bounded, inspectable overnight ICL queue. Solver code stays in Docker."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path

from roboswe.campaign import reference_lease
from roboswe.jobs import create,save,now
from roboswe.storage import require_disk_headroom
from roboswe.library.serial import SerialRuntime,serial_spec
from pairing.run_toaster_learning import FreshEpisode,run_series
from pairing.simulation_wait import passive_wait

ROOT=Path('artifacts/overnight-20261005')
PLAN=[
 ('ToastBagel','GR1','two-arm composite'),
 ('BlendIngredients','PandaOmron','multi-stage blending'),
 ('TurnOnSinkFaucet','PandaOmron','simpler fixture control'),
 ('CloseMicrowave','PandaOmron','simpler door closure'),
 ('TurnOnBlender','GR1','simpler two-arm button task'),
 ('OpenMicrowave','GR1','two-arm articulated fixture'),
 ('OpenCabinet','PandaOmron','door manipulation'),
 ('StartCoffeeMachine','PandaOmron','appliance control'),
 ('CloseCabinet','GR1','two-arm door closure'),
 ('OpenDrawer','PandaOmron','drawer manipulation'),
]


def emit_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


class SuiteRuntime(SerialRuntime):
    def __init__(self,*args,**kwargs):
        SerialRuntime.__init__(self,*args,**kwargs);self.last_action=None
    def bridge(self,message):
        if time.time()>=self.spec.get('absolute_deadline',float('inf')):raise TimeoutError('Overnight execution window ended')
        previous=self.last_action
        if message['operation']=='step':self.last_action=list(message['action'])
        try:return SerialRuntime.bridge(self,message)
        except BaseException:self.last_action=previous;raise


def fixture_diagnostics(raw):
    """Observer-only getters; never feed these privileged states to the solver."""
    result={}
    for name,method,needs_env in (
        ('blender','get_state',False),
        ('sink','get_handle_state',True),
        ('fxtr','get_door_state',True),
        ('drawer','get_door_state',True),
        ('microwave','get_door_state',True),
        ('toaster_oven','get_state',True),
    ):
        fixture=getattr(raw,name,None)
        getter=getattr(fixture,method,None)
        if getter is None and method=='get_door_state' and hasattr(fixture,'get_joint_state') and hasattr(fixture,'door_joint_names'):
            getter=lambda env:fixture.get_joint_state(env,fixture.door_joint_names)
        if getter is None:continue
        try:
            value=getter(raw) if needs_env else getter()
            # Detach any mutable state and normalize NumPy scalar values.
            result[name]=json.loads(json.dumps(value,default=lambda x:x.tolist(),allow_nan=False))
        except Exception as exc:
            # A diagnostic incompatibility must not abort or alter a task trial.
            result[name]={'diagnostic_unavailable':type(exc).__name__}
    return result


def telemetry(session,initial):
    raw=session.raw;objects=raw.objects
    primary=next((k for k in ('bagel','obj','fruit') if k in objects),None)
    if primary is None:
        primary=next((k for k in objects if 'container' not in k),None)
    position=raw.sim.data.body_xpos[raw.obj_body_id[primary]].tolist() if primary else [0,0,0]
    names=set(objects[primary].contact_geoms) if primary else set()
    touching=[]
    for c in raw.sim.data.contact[:raw.sim.data.ncon]:
        a,b=raw.sim.model.geom_id2name(c.geom1),raw.sim.model.geom_id2name(c.geom2)
        if a in names:touching.append(b)
        elif b in names:touching.append(a)
    hands=[g for g in touching if g and g.startswith('gripper')]
    gain=position[2]-initial.get(primary,position)[2]
    plate=any('plate' in (g or '') or 'container' in (g or '') for g in touching)
    result={'step':session.steps,'primary_object':primary,'object_position':position,'height_gain_m':gain,
        'hand_contacts':hands,'plate_contact':plate,'thumb_and_finger_contact':any('thumb' in g for g in hands) and any('thumb' not in g for g in hands),
        'supported_lift_candidate':bool(hands and not plate and gain>=.025),'upstream_success':bool(session.info.get('success')),
        'objects':{k:raw.sim.data.body_xpos[raw.obj_body_id[k]].tolist() for k in objects}}
    if hasattr(raw,'toaster'):result['toaster_state']=raw.toaster.get_state(raw)
    result['fixture_diagnostics']=fixture_diagnostics(raw)
    result['fixture_diagnostics_scope']='Privileged human audit only; not solver feedback or a replacement success predicate'
    return result


def suite_worker(pipe,config,spec,directory,number,previous):
    import numpy as np
    from robot_session.session import RobotSession
    from pairing.run_gr1_trial import GR1Session,RetrievalRuntime
    from pairing.gr1_compat import install_stock_gr1_placement
    class FullGR1Runtime(SuiteRuntime,RetrievalRuntime):
        # Reuse documented robot controls, but retain the OFFICIAL task predicate.
        start_task=SerialRuntime.start_task
        finish_task=SerialRuntime.finish_task
        success=SerialRuntime.success
        def observe(self):
            result = super().observe()
            if self.spec.get('cohort_protocol'):
                solver = self.session.raw.robots[0].composite_controller.joint_action_policy
                transform = solver.transform_pose(np.eye(4), src_frame='robot0_base', dst_frame='world')
                result['whole_body_ik'].update(
                    command_frame_position_world=transform[:3, 3].tolist(),
                    command_frame_rotation_world=transform[:3, :3].tolist(),
                    world_to_command_formula='p_command = R.T @ (p_world - origin); R_command = R.T @ R_world. R and origin are the command_frame fields here. Per-arm base_frame fields in action_schema describe downstream joint controllers, NOT the WholeBodyIK command frame.')
            if self.spec.get('hand_action_contract') == 'native-raw-position-v2':
                for hand in result['hand_interfaces'].values():
                    hand['units']='Raw joint-position targets in radians, passed through unchanged. Use the native action_schema low/high bounds. Upstream native grasp commands and full actuator ranges are accepted; physical actuator saturation remains unchanged. Part-controller normalized input_min/input_max are not the raw hand-command bounds when use_action_scaling is false.'
            return result
    directory=Path(directory);session=evidence=metrics=None
    try:
        if config['robot']=='GR1':install_stock_gr1_placement()
        session_class=GR1Session if config['robot']=='GR1' else RobotSession
        if spec.get('capture_counter_regions') or spec.get('counter_region_reference') is not None:
            from pairing.reset_region_order import CounterRegionOrder,guarded_factory
            with CounterRegionOrder(spec.get('counter_region_reference'),spec.get('counter_region_anchors',[]),atol=spec.get('counter_region_match_atol',0.)) as guard:
                session=session_class(config,factory=guarded_factory(guard))
            (directory/'counter-region-order.json').write_text(json.dumps(guard.payload(),indent=2))
        else:
            session=session_class(config)
        if spec.get('hand_action_contract') == 'native-raw-position-v2':
            session.env.enable_native_hand_ranges()
        if spec.get('transport_optimizations') == 'sensor-serialization-cache-v1':
            session._encode_step_observation = False
            session._cache_public_observation = True
        if spec.get('sensor_documentation') == 'nested-rgb-v1':
            session._nested_sensor_documentation = True
        if spec.get('cohort_protocol'):
            (directory/'initial-snapshot.zip').write_bytes(session.snapshot())
        session.recording_root=directory/'recordings';session.start_recording({'recording_id':'episode','width':640,'height':480})
        evidence=(directory/'actions.jsonl').open('x');metrics=(directory/'stage-telemetry.jsonl').open('x')
        runtime=(FullGR1Runtime if config['robot']=='GR1' else SuiteRuntime)(session,spec,timeout=spec['task_timeout_seconds'],max_steps=spec.get("max_control_steps_per_task",6000),evidence=evidence)
        initial={k:session.raw.sim.data.body_xpos[session.raw.obj_body_id[k]].tolist() for k in session.raw.objects}
        def capture():
            if not np.isfinite(session.raw.sim.data.qpos).all():raise RuntimeError('Non-finite simulator state')
            metrics.write(json.dumps(telemetry(session,initial))+'\n');metrics.flush()
        capture();pipe.send({'ok':runtime.metadata(),'fingerprint':session.initial})
        while True:
            request=pipe.recv();op=request['operation']
            try:
                if op=='start_task':
                    result=runtime.start_task(0)
                    result.update(trial_number=number,total_trials=spec.get('learning_trials',2),previous_attempts=previous,
                        reset_notice=f'Attempt {number} starts from the same initial scene. The robot body, hands, appliances and objects have reset. Your model conversation (including any compaction), Python library and NOTES.md persist from the previous attempt. This is the official whole task, not the earlier GR1 bread-retrieval diagnostic.')
                elif op=='observe':result=runtime.observe()
                elif op=='bridge':
                    result=runtime.bridge(request['message'])
                    if request['message']['operation']=='step':capture()
                elif op=='wait':result=passive_wait(runtime,runtime.last_action,request['steps'],on_step=capture)
                elif op=='digest':result=runtime.action_digest.hexdigest()
                elif op=='finish_task':
                    result=runtime.finish_task(code_errors=request['code_errors'])
                    result.update(passive_wait_control_steps=getattr(runtime,'passive_wait_steps',0),passive_wait_wall_seconds=getattr(runtime,'passive_wait_wall_seconds',0),budget_policy='passive_wait_v1')
                elif op=='close':
                    session.stop_recording({'recording_id':'episode'});(directory/'final-snapshot.zip').write_bytes(session.snapshot())
                    pipe.send({'ok':{'recording_dir':str(session.recording.directory)}});break
                else:raise ValueError('Unknown trusted coordinator operation')
                pipe.send({'ok':result})
            except Exception as exc:pipe.send({'error':traceback.format_exc(),'error_type':type(exc).__name__})
    except BaseException:
        try:pipe.send({'error':traceback.format_exc()})
        except (BrokenPipeError,EOFError):pass
    finally:
        if evidence:evidence.close()
        if metrics:metrics.close()
        if session:
            if session.recording and session.recording.active:session.stop_recording({'recording_id':'episode'})
            session.close()
        pipe.close()


def config_for(task,robot,seed=0):
    return {'robot':robot,'task':'robocasa/'+task,'split':None,'seed':seed,'env_kwargs':{'layout_ids':2,'style_ids':1,'camera_widths':640,'camera_heights':480,'camera_depths':True}}


def cohort(args):
    config=config_for(args.task,args.robot,args.seed)
    probe=ROOT/f'preflight-{args.index:02d}';probe.mkdir(exist_ok=False)
    spec=serial_spec(args.task,timeout=180,max_steps=6000);spec['absolute_deadline']=args.deadline
    runtime=FreshEpisode(config,spec,probe,0,[],worker=suite_worker)
    try:
        obs=runtime.start_task(0)['observation']
        if args.robot=='GR1':a=obs['whole_body_ik']['current_pose_hold_action']
        else:a=[0.]*obs['action_schema']['shape'][0]
        begin=time.monotonic()
        for _ in range(50):runtime.bridge({'operation':'step','action':a})
        seconds=time.monotonic()-begin
        emit_json(probe/'receipt.json',{'50_steps_seconds':seconds,'initial_fingerprint':runtime.initial['fingerprint'],'config':config})
        if seconds>45:raise RuntimeError('Recording throughput leaves too little margin for a 50-step/60-second execution; reduce concurrency')
    finally:runtime.close()
    result=run_series(args.task,args.code_image,args.parent,args.photo,timeout=1800,trials=2,config_override=config,worker=suite_worker,absolute_deadline=args.deadline)
    emit_json(ROOT/f'cohort-{args.index:02d}-result.json',result)


def report_status(directory,record,queue,active,completed,deadline):
    report={'deadline_utc':datetime.fromtimestamp(deadline,timezone.utc).isoformat(),'queued':queue,
            'active':[{'index':k,**{a:v[a] for a in ['task','robot','pid','started_at']}} for k,v in active.items()],
            'completed':completed,'benchmark_score':None,'assessment_scope':'Descriptive paired ICL with retained context, code and notes. No claim of isolated causal effect or independent replay certification.'}
    emit_json(directory/'report.json',report);record['artifacts']['report']='report.json';save(directory,record)


def supervisor(args):
    ROOT.mkdir(exist_ok=True)
    directory,record=create('jobs','Astra overnight · paired ICL task variety · 8 hours','overnight-campaign')
    record['status']='waiting_for_current_run';save(directory,record)
    emit_json(ROOT/'campaign.json',{'id':record['id'],'directory':str(directory),'deadline':args.deadline})
    queue=[{'index':i,'task':t,'robot':r,'purpose':purpose,'seed':0} for i,(t,r,purpose) in enumerate(PLAN)]
    emit_json(directory/'plan.json',{'queue':queue,'attempts_per_task':2,'active_seconds_per_attempt':1800,'max_concurrency':2,'context_request':1000000,'compact_at':800000,'deadline_epoch':args.deadline,'deadline_utc':datetime.fromtimestamp(args.deadline,timezone.utc).isoformat(),'code_image':args.code_image})
    record['artifacts']['plan']='plan.json';save(directory,record)
    (directory/'source.py').write_bytes(Path(__file__).read_bytes());record['artifacts']['source']='source.py'
    active={};completed=[];lease=None;handled_requests=set();next_retry=100;replication=0;next_replica=1000;stopping=False
    try:
        while time.time()<args.deadline:
            candidate=reference_lease({'jobs_dir':'jobs','execution':{'memory_mb':6144,'cpu':4,'min_free_disk_gb':10}})
            try:candidate.__enter__();lease=candidate;break
            except ValueError as exc:
                if 'Another reference campaign or verification is active' not in str(exc):raise
                time.sleep(10)
        if lease is None:record['status']='window_expired';return
        record['status']='running';save(directory,record)
        while time.time()<args.deadline:
            control=json.loads((ROOT/'control.json').read_text()) if (ROOT/'control.json').exists() else {}
            if control.get('stop'):
                stopping=True
                break
            for request_file in sorted((ROOT/'retry-requests').glob('*.json')):
                if request_file.name in handled_requests:continue
                request=json.loads(request_file.read_text())
                if request.get('task') not in {x[0] for x in PLAN} or request.get('robot') not in {'GR1','PandaOmron'}:
                    raise ValueError('Invalid monitor retry request')
                queue.append({'index':next_retry,'task':request['task'],'robot':request['robot'],'purpose':'Documented infrastructure repair retry','retry_of':request.get('retry_of'),'reason':request.get('reason')})
                next_retry+=1;handled_requests.add(request_file.name)
            if not queue and not active and args.deadline-time.time()>=4980:
                replication+=1
                queue.extend({'index':next_replica+i,'task':t,'robot':r,'purpose':'Independent paired replication with a new seed','seed':replication} for i,(t,r,_) in enumerate(PLAN))
                next_replica+=len(PLAN)
            limit=max(1,min(2,int(control.get('concurrency',2))))
            require_disk_headroom('jobs',10)
            while queue and len(active)<limit and not control.get('pause_launches'):
                # Start a pair only with room for two full work budgets plus reset/wait overhead.
                if args.deadline-time.time()<2*(1800+600)+180:break
                item=queue.pop(0)
                if item['task'] in control.get('disabled_tasks',[]):
                    completed.append({**item,'status':'quarantined_by_monitor'});continue
                log=ROOT/f"cohort-{item['index']:02d}.log";stream=log.open('x')
                command=[sys.executable,'-u','-m','pairing.overnight_suite','cohort','--task',item['task'],'--robot',item['robot'],'--index',str(item['index']),'--parent',record['id'],'--seed',str(item.get('seed',0)),'--deadline',str(args.deadline),'--code-image',args.code_image,'--photo',args.photo]
                proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYNPUT_BACKEND':'dummy','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
                active[item['index']]={**item,'proc':proc,'pid':proc.pid,'stream':stream,'started_at':now()}
                print(json.dumps({'launched':item,'pid':proc.pid}),flush=True)
            for idx,item in list(active.items()):
                code=item['proc'].poll()
                if code is not None:
                    item['stream'].close();completed.append({k:v for k,v in item.items() if k not in ['proc','stream']}|{'exit_code':code,'finished_at':now()})
                    if code:
                        tail=(ROOT/f'cohort-{idx:02d}.log').read_text(errors='replace')[-12000:]
                        if 'Recording throughput leaves too little margin' in tail and not item.get('throughput_retry'):
                            updated=json.loads((ROOT/'control.json').read_text());updated['concurrency']=1;emit_json(ROOT/'control.json',updated)
                            queue.insert(0,{'index':next_retry,'task':item['task'],'robot':item['robot'],'seed':item.get('seed',0),'purpose':'Repeat preflight at concurrency one','throughput_retry':True})
                            next_retry+=1
                        elif any(k in tail.lower() for k in ['usage limit','rate limit','insufficient_quota','account differs','account changed']):
                            updated=json.loads((ROOT/'control.json').read_text());updated.update(pause_launches=True,pause_reason='Account/quota issue; monitor review required');emit_json(ROOT/'control.json',updated)
                    del active[idx]
            report_status(directory,record,queue,active,completed,args.deadline)
            if not active and args.deadline-time.time()<4980:
                record['status']='review_window';save(directory,record)
            time.sleep(10)
        # In-flight requests have the same absolute deadline. Allow artifact cleanup.
        for item in active.values():
            if stopping and item['proc'].poll() is None:os.kill(item['pid'],signal.SIGTERM)
            try:item['proc'].wait(timeout=60 if stopping else max(1,args.deadline-time.time())+60)
            except subprocess.TimeoutExpired:
                os.kill(item['pid'],signal.SIGTERM)
                try:item['proc'].wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(item['pid'],signal.SIGKILL)
            item['stream'].close()
            completed.append({k:v for k,v in item.items() if k not in ['proc','stream']}|{'exit_code':item['proc'].returncode,'finished_at':now()})
        active.clear()
        record['status']='stopped_by_monitor' if stopping else ('window_complete' if time.time()>=args.deadline else 'queue_finished')
        report_status(directory,record,queue,active,completed,args.deadline)
    except BaseException:
        record['status']='infrastructure_error';(directory/'stderr.txt').write_text(traceback.format_exc());record['artifacts']['stderr']='stderr.txt';raise
    finally:
        for item in active.values():
            if item['proc'].poll() is None:
                os.kill(item['pid'],signal.SIGTERM)
                try:item['proc'].wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(item['pid'],signal.SIGKILL)
            if not item['stream'].closed:item['stream'].close()
        if lease:lease.__exit__(None,None,None)
        record['finished_at']=now();save(directory,record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['supervisor','cohort']);parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--code-image',required=True);parser.add_argument('--photo',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--task');parser.add_argument('--robot');parser.add_argument('--index',type=int);parser.add_argument('--parent')
    args=parser.parse_args()
    if args.mode=='cohort':
        def stop_cohort(*_):raise KeyboardInterrupt('Overnight supervisor requested cleanup')
        signal.signal(signal.SIGTERM,stop_cohort)
    (supervisor if args.mode=='supervisor' else cohort)(args)
if __name__=='__main__':main()
