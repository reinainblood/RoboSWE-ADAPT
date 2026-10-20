import json
import pytest
from roboswe import adaptive
from roboswe.cli import parser_for_cli, main


def files(tmp_path, backend='docker'):
    dataset=tmp_path/'dataset.json'
    dataset.write_text(json.dumps({'tasks':[{'id':'blend','task':'BlendIngredients','config':{'task':'robocasa/BlendIngredients','robot':'PandaOmron','seed':0,'env_kwargs':{'layout_ids':1,'style_ids':1}}}]}))
    config=tmp_path/'run.json'
    value={'participant':{'provider':'openai','model':'explicit-model','context_window':1000000,'max_output_tokens':131000},'execution':{'backend':backend,'simulator_image':'sha256:'+'a'*64,'code_image':'sha256:'+'b'*64}}
    config.write_text(json.dumps(value));return dataset,config,value


def test_public_plan_has_no_private_cohort_dependency(tmp_path):
    data,config,_=files(tmp_path)
    plan=adaptive.build_plan(data,config)
    assert plan['planned_attempts']==5
    task=plan['tasks'][0]
    assert task['initial_fingerprint'] is None
    assert task['video_judge'] is None
    assert task['context_policy']=='sqlite-sliding-v1'
    assert '5-attempt' in task['prompt']


def test_optional_video_judge_requires_separate_explicit_budget(tmp_path):
    data, config, value = files(tmp_path)
    value['video_judge'] = 'gemini'
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='ledger_path'):
        adaptive.build_plan(data, config)
    value['video_judge'] = {'backend': 'gemini', 'ledger_path': 'judge.json', 'spend_limit_usd': 10}
    config.write_text(json.dumps(value))
    plan = adaptive.build_plan(data, config)
    assert plan['tasks'][0]['video_judge'] == value['video_judge']
    assert not (tmp_path/'judge.json').exists()


def test_cli_dry_run_and_environment_alias(tmp_path,capsys):
    data,config,_=files(tmp_path)
    assert main(['run',str(data),'--config',str(config),'-e','docker','--dry-run'])==0
    plan=json.loads(capsys.readouterr().out)
    assert plan['execution']['backend']=='docker'
    assert parser_for_cli().parse_args(['run','-e','modal']).backend=='modal'


def test_custom_limits_are_explicit(tmp_path):
    data,config,value=files(tmp_path)
    value['limits']={'active_seconds':2400,'control_steps':12000,'per_script_steps':75,'per_script_seconds':90}
    config.write_text(json.dumps(value))
    plan=adaptive.build_plan(data,config)['tasks'][0]
    assert plan['limits']['control_steps']==12000
    assert '2400-second' in plan['prompt']
    assert '75-step, 90-second' in plan['prompt']


def test_reject_plaintext_keys_and_missing_modal_configuration(tmp_path):
    data,config,value=files(tmp_path)
    value['participant']['api_key']='never-serialize-this'
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='API keys'):adaptive.build_plan(data,config)
    del value['participant']['api_key']
    value['execution'].update(backend='modal',simulator_image='im-explicit')
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='modal_environment'):adaptive.build_plan(data,config)


def test_openrouter_refuses_before_transport(tmp_path):
    data,config,value=files(tmp_path)
    value['participant']['provider']='openrouter';config.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='token counter'):adaptive.run(adaptive.build_plan(data,config))


def test_replay_failure_is_never_pass_and_closes_owned_runtime(tmp_path,monkeypatch):
    from pairing import adaptive_run as runner
    from zipfile import ZipFile
    from roboswe.library import attempts
    with ZipFile(tmp_path/'library-start.zip','w') as archive:
        archive.writestr('library-manifest.json','{}')
        archive.writestr('motion.py','print("retained source")')
    writes=[];closed=[]
    class Sandbox:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def restore_checkpoint(self,path):writes.append(('restore_checkpoint',path.name))
    class Runtime:
        def close(self):closed.append(True)
    monkeypatch.setattr(runner.c,'CodeSandbox',Sandbox)
    monkeypatch.setattr(attempts,'replay_journal',lambda *a,**kw:{'reproduced':False,'trials':[{'verified_success':True}]})
    plan={'code_image':'image','limits':{'active_seconds':1800},'cohort':{'config':{}}}
    result=runner.verify_attempt(plan,{},tmp_path,1,[],lambda *a,**kw:Runtime())
    assert not result['verified_success']
    assert writes==[('restore_checkpoint','library-start.zip')] and closed==[True]


def test_reject_mixed_legacy_budget_flags(tmp_path,capsys):
    data,config,_=files(tmp_path)
    assert main(['run',str(data),'--config',str(config),'--timeout','3','--dry-run'])==1
    assert 'belong in --config' in capsys.readouterr().err


def test_images_receipt_supplies_ids_and_records_exact_provenance(tmp_path, capsys):
    import hashlib
    data, config, value = files(tmp_path)
    value.pop('execution')
    config.write_text(json.dumps(value))
    receipt = tmp_path/'images.json'
    receipt.write_text(json.dumps({'backend':'docker','simulator_image':'sha256:'+'c'*64,'code_image':'sha256:'+'d'*64}))
    assert main(['run',str(data),'--config',str(config),'--images',str(receipt),'-e','docker','--dry-run']) == 0
    plan=json.loads(capsys.readouterr().out)
    assert plan['execution']['code_image']=='sha256:'+'d'*64
    assert plan['tasks'][0]['code_image']==plan['execution']['code_image']
    provenance=plan['image_build_receipt']
    assert provenance['sha256']==hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert plan['tasks'][0]['image_build_receipt']==provenance
    assert not (tmp_path/'jobs').exists()


def test_modal_images_receipt_overrides_build_fields_only(tmp_path):
    data, config, value=files(tmp_path, 'modal')
    value['execution'].update(modal_app='my-app',modal_profile='old-profile')
    config.write_text(json.dumps(value))
    receipt=tmp_path/'images.json'
    receipt.write_text(json.dumps({'backend':'modal','simulator_image':'im-ABC123','code_image':'sha256:'+'e'*64,'modal_profile':'explicit-profile','modal_environment':'FDR'}))
    plan=adaptive.build_plan(data,config,images_path=receipt)
    assert plan['execution']['modal_app']=='my-app'
    assert plan['execution']['modal_profile']=='explicit-profile'
    assert plan['execution']['artifact_transfer']=='remote-video'


@pytest.mark.parametrize('patch,match',[
    ({'backend':'modal'},'backend'),
    ({'code_image':'mutable:tag'},'immutable'),
    ({'simulator_image':'mutable:tag'},'immutable'),
    ({'api_key':'do-not-echo'},'Unsupported image receipt'),
    ({'modal_sdk_python':'/untrusted-command'},'Unsupported image receipt'),
    ({'code_image':None},'nonempty strings'),
])
def test_images_receipt_rejects_mismatch_mutable_or_extra_fields(tmp_path,patch,match):
    data,config,_=files(tmp_path)
    receipt=tmp_path/'images.json'
    receipt.write_text(json.dumps({'backend':'docker','simulator_image':'sha256:'+'a'*64,'code_image':'sha256:'+'b'*64,**patch}))
    with pytest.raises(ValueError,match=match):adaptive.build_plan(data,config,images_path=receipt)


def test_modal_receipt_cannot_borrow_missing_profile_from_config(tmp_path):
    data,config,value=files(tmp_path,'modal')
    value['execution'].update(modal_profile='configured',modal_environment='FDR')
    config.write_text(json.dumps(value))
    receipt=tmp_path/'images.json'
    receipt.write_text(json.dumps({'backend':'modal','simulator_image':'im-ABC123','code_image':'sha256:'+'b'*64}))
    with pytest.raises(ValueError,match='explicit modal_profile'):
        adaptive.build_plan(data,config,images_path=receipt)


def test_modal_volume_default_resolves_in_selected_environment(tmp_path):
    data,config,value=files(tmp_path,'modal')
    value['execution'].update(simulator_image='im-ABC123',modal_profile='community',modal_environment='evaluation')
    config.write_text(json.dumps(value))
    execution=adaptive.build_plan(data,config)['execution']
    assert execution['modal_volume']=='roboswe-job-artifacts-v1'
    assert execution['modal_environment']=='evaluation' and execution['modal_profile']=='community'
    value['execution']['modal_volume']='custom-evidence';config.write_text(json.dumps(value))
    assert adaptive.build_plan(data,config)['execution']['modal_volume']=='custom-evidence'
    value['execution']['modal_volume']='';config.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='modal_volume'):adaptive.build_plan(data,config)


def test_native_thinking_display_public_configuration(tmp_path):
    data,config,value=files(tmp_path)
    value['participant'].update(provider='anthropic',model='claude-opus-5-5',thinking_display='summarized')
    config.write_text(json.dumps(value))
    assert adaptive.build_plan(data,config)['tasks'][0]['participant']['thinking_display']=='summarized'
    value['participant']['provider']='openrouter';config.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Anthropic-only'):adaptive.build_plan(data,config)
    value['participant'].update(provider='anthropic',thinking_display='updates');config.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='summarized'):adaptive.build_plan(data,config)
