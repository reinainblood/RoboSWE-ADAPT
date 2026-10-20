"""Read the stock GR1 interface, render it, and test a brief neutral hold."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from robot_session.session import RobotSession
from pairing.run_hand_microwave import trial_slot
from roboswe.campaign import publish_video
from roboswe.jobs import create,save,now

def hold(session):
    robot=session.raw.robots[0]
    solver=robot.composite_controller.joint_action_policy
    data=session.raw.sim.data
    base=data.get_body_xpos('robot0_base');B=data.get_body_xmat('robot0_base')
    commands=[]
    for arm in robot.arms:
        site=robot.gripper[arm].important_sites['grip_site']
        p=data.get_site_xpos(site);R=data.get_site_xmat(site)
        commands.extend((B.T@(p-base)).tolist())
        commands.extend(Rotation.from_matrix(B.T@R).as_rotvec().tolist())
    commands.extend([0.]*(session.env.action_space.shape[0]-len(commands)))
    return commands

def main():
    from pairing.gr1_compat import install_stock_gr1_placement
    install_stock_gr1_placement()
    root=Path('artifacts/gr1-default-preview')
    config={'robot':'GR1','task':'robocasa/GetToastedBread','split':None,'seed':0,'env_kwargs':{'layout_ids':2,'style_ids':1,'camera_widths':640,'camera_heights':480,'camera_depths':True}}
    with trial_slot({'execution':{'memory_mb':6144,'cpu':4,'min_free_disk_gb':10},'jobs_dir':'jobs'},True):
        s=RobotSession(config)
        directory,record=create('jobs','GR1 · stock Fourier hands · first look','robot-preview')
        (root/'job.json').write_text(json.dumps({'id':record['id'],'directory':str(directory)}))
        try:
            (root/'status.json').write_text(json.dumps(s.status(),indent=2))
            (root/'model.xml').write_text(s.raw.sim.model.get_xml())
            for k,v in s.observation.items():
                if k.endswith('_image'):Image.fromarray(v).save(root/f'initial-{k}.png')
            s.recording_root=root/'recordings';s.start_recording({'recording_id':'preview','width':640,'height':480})
            robot=s.raw.robots[0];raw=s.raw
            p0=raw.sim.data.get_body_xpos('robot0_base').copy();action=hold(s)
            trace=[]
            for _ in range(60):
                s.step(action)
                p=raw.sim.data.get_body_xpos('robot0_base').copy()
                trace.append({'step':s.steps,'base':p.tolist()})
                if not np.isfinite(raw.sim.data.qpos).all() or np.linalg.norm(p-p0)>.20:break
            for k,v in s.observation.items():
                if k.endswith('_image'):Image.fromarray(v).save(root/f'after-{k}.png')
            report={'config':config,'grippers':{arm:type(g).__name__ for arm,g in robot.gripper.items()},'controller':type(robot.composite_controller).__name__,'action_schema':s.env.action_schema(),'initial_hold_action':action,'base_start':p0.tolist(),'base_end':p.tolist(),'base_displacement_m':float(np.linalg.norm(p-p0)),'steps':s.steps,'trace':trace,'benchmark_score':None}
            (root/'report.json').write_text(json.dumps(report,indent=2));(directory/'report.json').write_text(json.dumps(report,indent=2));record['artifacts']['report']='report.json'
            record['status']='hold_stable' if s.steps==60 and np.linalg.norm(p-p0)<.05 else 'hold_unstable'
        finally:
            if s.recording and s.recording.active:
                s.stop_recording({'recording_id':'preview'});publish_video(s.recording.directory,directory,record,phase='original')
            record['finished_at']=now();save(directory,record);s.close()
if __name__=='__main__':main()
