"""Explicit image provisioning; never part of a plan-only invocation."""
import json
import os
import subprocess
from pathlib import Path
from roboswe.execution.sandbox import build_image as build_code, docker_path


def build(source_root, *, backend, profile=None, environment='FDR', output):
    root=Path(source_root).resolve()
    output=Path(output).resolve()
    if output.exists():raise ValueError('Refusing to overwrite image receipt')
    for path in ('robot_session/Dockerfile','upstream/robocasa/robocasa','upstream/robosuite/robosuite','pairing'):
        if not (root/path).exists():raise ValueError('Source checkout with upstream assets required: missing '+path)
    if backend not in ('docker','modal'):raise ValueError('Unknown environment')
    if backend=='modal' and (not profile or not environment):raise ValueError('Explicit Modal profile and environment required')
    docker=docker_path()
    if backend=='docker':
        subprocess.run([docker,'build','-f',str(root/'robot_session/Dockerfile'),'-t','roboswe-adaptive-simulator:local',str(root)],check=True)
        simulator=subprocess.check_output([docker,'image','inspect','--format','{{.Id}}','roboswe-adaptive-simulator:local'],text=True).strip()
    else:
        previous=os.environ.get('MODAL_PROFILE');os.environ['MODAL_PROFILE']=profile
        try:
            import modal
            app=modal.App.lookup('roboswe-adaptive-images',environment_name=environment,create_if_missing=True)
            with modal.enable_output():
                image=modal.Image.from_dockerfile(root/'robot_session/Dockerfile',context_dir=root,
                    ignore=modal.FilePatternMatcher.from_file(root/'robot_session/Dockerfile.dockerignore')).build(app)
            simulator=image.object_id
        finally:
            if previous is None:os.environ.pop('MODAL_PROFILE',None)
            else:os.environ['MODAL_PROFILE']=previous
    code=build_code()
    receipt={'backend':backend,'simulator_image':simulator,'code_image':code}
    if backend=='modal':receipt.update(modal_profile=profile,modal_environment=environment)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    return output
