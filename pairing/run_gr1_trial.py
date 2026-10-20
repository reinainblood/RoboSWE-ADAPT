"""Stock GR1 + Fourier hands, one explicitly scoped toast-retrieval diagnostic."""
import json,time,traceback
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scipy.spatial.transform import Rotation
from robot_session.session import RobotSession
from roboswe.library.serial import SerialRuntime,serial_spec
from roboswe.library.attempts import Journal
from roboswe.execution.sandbox import CodeSandbox
from roboswe.campaign import drive_task,publish_video
from roboswe.jobs import create,save,now
from pairing.gr1_compat import install_stock_gr1_placement
from pairing.preview_gr1 import hold
from pairing.run_hand_microwave import trial_slot
from pairing.run_toaster_learning import LearningModel,instructions,write_human_outputs,EXPECTED_ACCOUNT
from pairing.simulation_wait import WaitingAttemptRunner,passive_wait

ROOT=Path('artifacts/gr1-toast-trial')
GOAL='Start the toaster, wait for its cycle to finish and the bread to return, then retrieve the bread and hold it clear of the toaster with one or both hands. This diagnostic ends at retrieval: delivery to the distant plate is excluded because stock GR1 has no actuated mobile base. No built-in grasp routines are supplied.'


class GR1Session(RobotSession):
    def step(self, action):
        result = super().step(action)
        if not np.isfinite(self.raw.sim.data.qpos).all() or not np.isfinite(self.raw.sim.data.qvel).all():
            self.phase = 'faulted'
            raise RuntimeError('GR1 physics became non-finite; diagnostic stopped')
        return result

    def passive_hold_action(self, previous):
        c=self.raw.robots[0].composite_controller
        solver=c.joint_action_policy
        if type(c).__name__!='WholeBodyIK' or solver.input_action_repr!='absolute' or solver.input_ref_frame!='base' or solver.input_rotation_repr!='axis_angle':
            raise ValueError('GR1 hold requires the validated stock absolute/base/axis-angle controller')
        a=hold(self)
        for name,bounds in c.get_action_info_dict().items():
            if name=='Action Dimension':continue
            start,end=bounds
            if name.endswith('_gripper'):a[start:end]=previous[start:end]
        return a


class RetrievalRuntime(SerialRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.last_action=None;self.passive_wait_steps=0;self.retrieval_streak=0;self.ever_on=False
        self.bread_id=self.session.raw.obj_body_id['obj']
        self.bread_start_z=float(self.session.raw.sim.data.body_xpos[self.bread_id][2])
        self.diagnostic=(ROOT/'stage-telemetry.jsonl').open('w')
    def start_task(self,index):
        result=super().start_task(index)
        result.update(instruction=GOAL,total_trials=1,trial_number=1,success_criteria={'source':'diagnostic only','requires':['toaster cycle activated and completed','bread at least 4 cm above its initial raised center height','hand contact and no toaster contact sustained for 20 control ticks']})
        return result
    def success(self):return self.retrieval_streak>=20
    def observe(self):
        result=super().observe();robot=self.session.raw.robots[0];c=robot.composite_controller
        hands={}
        for arm,g in robot.gripper.items():
            part=arm+'_gripper';pc=robot.part_controllers[part]
            hands[arm]={'model':type(g).__name__,'action_slice':list(c.get_action_info_dict()[part]),
                'channels':['pinky curl','ring curl','middle curl','index curl','thumb bend','thumb opposition'],
                'format':'Six commands expanded to eleven joint position actuators; two joints per finger and thumb bend, one thumb opposition.',
                'use_action_scaling':pc.use_action_scaling,
                'units':'Raw joint-position targets in radians. Stock native action bounds are [-1,1]; negative targets are below these actuators\' zero minimum. 1 is one radian, NOT necessarily the full joint range.',
                'joint_observation_names':g.joints,
                'actuator_target_ranges':np.stack([pc.actuator_min,pc.actuator_max],axis=1).tolist(),
                'sensing':'Joint positions/velocities and a wrist force/torque sensor. No fingertip tactile sensors in this stock XML.'}
        result['hand_interfaces']=hands
        arm_hold = hold(self.session)
        if self.last_action is not None:
            for arm in robot.arms:
                start,end=c.get_action_info_dict()[arm+'_gripper'];arm_hold[start:end]=self.last_action[start:end]
        result['whole_body_ik']={'input_type':'absolute','reference_frame':'robot0_base','rotation':'axis-angle rotation vector, radians','order':'right position xyz + rotation vector; left position xyz + rotation vector; right hand 6; left hand 6',
            'note':'The arm part-controller joint parameters are downstream of IK. Do not send joint angles in the 6D arm pose slices. Preserve the other hand pose while moving one. The stock base is fixed; both arms, torso, head and legs participate in the stock solver.',
            'current_pose_hold_action':arm_hold}
        return result
    def bridge(self,message):
        previous=self.last_action
        if message['operation']=='step':self.last_action=list(message['action'])
        try:result=super().bridge(message)
        except BaseException:
            self.last_action=previous
            raise
        if message['operation']=='step':
            raw=self.session.raw;state=raw.toaster.get_state(raw)
            on=any(v['turned_on'] for v in state.values());self.ever_on|=on
            obj=set(raw.objects['obj'].contact_geoms);touch=[]
            for contact in raw.sim.data.contact[:raw.sim.data.ncon]:
                a=raw.sim.model.geom_id2name(contact.geom1);b=raw.sim.model.geom_id2name(contact.geom2)
                if a in obj:touch.append(b)
                elif b in obj:touch.append(a)
            height=float(raw.sim.data.body_xpos[self.bread_id][2])-self.bread_start_z
            hand=any(x.startswith('gripper') for x in touch);toaster=any(x.startswith(raw.toaster.naming_prefix) for x in touch)
            candidate=self.ever_on and not on and height>=.04 and hand and not toaster
            self.retrieval_streak=self.retrieval_streak+1 if candidate else 0
            self.diagnostic.write(json.dumps({'step':self.session.steps,'toaster':state,'height_gain':height,'touching':touch,'retrieval_streak':self.retrieval_streak})+'\n');self.diagnostic.flush()
            result['task_success']=self.success()
        return result
    def wait(self,steps):return passive_wait(self,self.last_action,steps)
    def finish_task(self,*,code_errors=0):
        report=super().finish_task(code_errors=code_errors)
        report.update(instruction=GOAL,diagnostic_only=True,official_full_task_evaluated=False,
            verdict={'source':'retrieval diagnostic predicate, not official whole-task success','success':self.success(),'sustained_steps':self.retrieval_streak},
            passive_wait_control_steps=self.passive_wait_steps,passive_wait_wall_seconds=getattr(self,'passive_wait_wall_seconds',0))
        return report


def main():
    ROOT.mkdir(exist_ok=True)
    install_stock_gr1_placement()
    config={'robot':'GR1','task':'robocasa/GetToastedBread','split':None,'seed':0,'env_kwargs':{'layout_ids':2,'style_ids':1,'camera_widths':640,'camera_heights':480,'camera_depths':True}}
    image=Path('artifacts/toaster-learning-code-image.txt').read_text().strip()
    prompt=instructions('GetToastedBread',timeout=1800,trials=1)
    prompt += '\n'+GOAL+'\nUse the actual GR1 action schema and per-hand interface, not Panda/Inspire assumptions. The robot has TWO independent Fourier hands. You may use both, or one, as your own plan requires. No walking/base-translation action is available. This is a retrieval diagnostic, not a full RoboCasa task score. The wait operation holds both current absolute hand poses and preserves their finger targets, rather than sending zero Cartesian targets.\n'
    directory,record=create('jobs','Astra high · GR1 + stock Fourier hands · toast retrieval','gr1-retrieval-diagnostic')
    record.update(status='queued',task='GetToastedBread · retrieval only',benchmark_score=None,verifier_status='diagnostic_only')
    (ROOT/'job.json').write_text(json.dumps({'id':record['id'],'directory':str(directory)}));save(directory,record)
    plan={'config':config,'model':'gpt-6-astra','effort':'high','timeout_seconds':1800,'instructions':prompt,'goal':GOAL,'placement_compatibility':'Local reset-only placement shim for stock unactuated GR1 base; upstream files unchanged','stock_controller_and_hands':True}
    (directory/'plan.json').write_text(json.dumps(plan,indent=2));record['artifacts']['plan']='plan.json';save(directory,record)
    s=runtime=journal=None
    try:
        with trial_slot({'execution':{'memory_mb':6144,'cpu':4,'min_free_disk_gb':10},'jobs_dir':'jobs'},True), CodeSandbox(image,lifetime_seconds=3000) as sandbox, LearningModel(model='gpt-6-astra',effort='high',instructions=prompt,context_window=1000000,compact_at=800000,reasoning_summary='detailed',verbosity='high',persistent=True,diagnostics=directory/'model-events.jsonl',summary_log=directory/'reasoning-summaries.jsonl') as model:
            if model._rpc('account/rateLimits/read',{}).get('accountId')!=EXPECTED_ACCOUNT:raise RuntimeError('Active account is not the selected RoboSWE account')
            (directory/'model.json').write_text(json.dumps(model.metadata,indent=2))
            s=GR1Session(config);s.recording_root=directory/'recordings';s.start_recording({'recording_id':'attempt','width':640,'height':480})
            runtime=RetrievalRuntime(s,serial_spec('GetToastedBread',timeout=1800,max_steps=6000),timeout=1800,max_steps=6000,evidence=(directory/'actions.jsonl').open('x'))
            journal=Journal(directory/'submissions.jsonl');runner=WaitingAttemptRunner(runtime,sandbox,journal)
            record.update(status='running',execution_started_at=now());save(directory,record)
            with (directory/'model-transcript.jsonl').open('x') as transcript:
                report,_=drive_task(runner,model,0,{'max_control_steps':6000,'max_attempt_steps':50,'attempt_timeout_seconds':60},directory,transcript,time.monotonic()+2700)
            report['library']=sandbox.archive(directory/'library.zip')
            (directory/'report.json').write_text(json.dumps(report,indent=2));record['artifacts'].update(report='report.json',library='library.zip')
            write_human_outputs(directory);record['artifacts']['stdout']='stdout.txt'
            record['status']='diagnostic_pass' if report['observed_success'] else 'diagnostic_incomplete'
    except BaseException:
        record['status']='infrastructure_error';(directory/'stderr.txt').write_text(traceback.format_exc());record['artifacts']['stderr']='stderr.txt';raise
    finally:
        if journal:journal.close()
        if runtime:runtime.diagnostic.close();runtime.evidence.close()
        if s:
            if s.recording and s.recording.active:s.stop_recording({'recording_id':'attempt'});publish_video(s.recording.directory,directory,record,phase='original')
            (directory/'final-snapshot.zip').write_bytes(s.snapshot());s.close()
        record['finished_at']=now();save(directory,record)
if __name__=='__main__':main()
