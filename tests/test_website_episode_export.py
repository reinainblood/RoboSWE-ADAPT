import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('episode_export',Path(__file__).parents[1]/'website/scripts/export-episode.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_nested_private_metadata_and_serialized_json_are_removed():
    raw={'workspace_id':'wrkspc_private','modal_profile':'snorkelai','modal_sdk_python':'/example/person/sdk','headers':{'Authorization':'private'},'trace':json.dumps({'workspace_id':'wrkspc_secret','message':'file /example/person/sdk'})}
    text=json.dumps(module.sanitize(raw))
    for forbidden in ('wrkspc_','snorkelai','/Users/','Authorization','modal_profile','modal_sdk_python'):
        assert forbidden not in text


def test_signature_and_encrypted_blocks_opaque_even_when_short():
    result=module.sanitize({'signature':'short','reasoning_details':[{'type':'reasoning.encrypted','data':'tiny'}],'thinking':'Actual provider-exposed text'})
    assert result['signature']['opaque'] is True
    assert result['reasoning_details'][0]['data']['opaque'] is True
    assert result['thinking']=='Actual provider-exposed text'


def test_credential_pattern_fails_closed():
    with pytest.raises(ValueError,match='credential'):
        module.sanitize({'message':'sk-'+'a'*30})


def test_excluded_series_cannot_be_promoted_by_label():
    review={'classification':'development_context_policy','primary_leaderboard_eligible':False,'reason':'early slides'}
    value=module.publication_metadata({},review,'incomplete-series')
    assert value['primary_leaderboard_eligible'] is False
    assert 'excluded' in value['label']


def test_individual_publication_does_not_claim_series_result():
    value=module.publication_metadata({}, {}, 'incomplete-series')
    assert value['primary_leaderboard_eligible'] is None
    assert 'incomplete' in value['label']
    with pytest.raises(ValueError):module.publication_metadata({}, {}, 'leaderboard winner')


def test_active_trial_refused_before_export(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    source=tmp_path/'jobs/active';source.mkdir(parents=True)
    (source/'job.json').write_text(json.dumps({'status':'running'}))
    monkeypatch.setattr(module,'export',lambda *a:pytest.fail('No active export'))
    with pytest.raises(ValueError,match='finalized'):
        module.publish('active',tmp_path/'public',publication_label='incomplete-series')


def completed_series(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path);jobs=tmp_path/'jobs';parent=jobs/'series';parent.mkdir(parents=True)
    (parent/'job.json').write_text(json.dumps({'id':'series','status':'completed_unscored','finished_at':'now'}))
    (parent/'report.json').write_text(json.dumps({'trials':[{}]*5}))
    selected=None
    for n in range(1,6):
        child=jobs/f'trial-{n}';child.mkdir()
        selected={'id':child.name,'campaign_id':'series','kind':'adaptive-direct-attempt','attempt_number':n,'status':'verified_fail','finished_at':'now'}
        (child/'job.json').write_text(json.dumps(selected))
        (child/'verification.json').write_text(json.dumps({'kind':'independent_code_replay','reproduced':True,'verified_success':False,'mismatches':[],'trials':[{'reproduced':True,'infrastructure_valid':True,'precondition_valid':True,'verified_success':False,'observed_success':False,'replay':{'precondition_valid':True,'observed_success':False}}]}))
    return selected


def test_completed_label_requires_all_five_independent_replays(tmp_path,monkeypatch):
    selected=completed_series(tmp_path,monkeypatch)
    value=module.publication_metadata(selected,{},'completed-series')
    assert value['series_completion']['verified_trials']==5
    assert value['series_completion']['verified_successes']==0
    assert value['primary_leaderboard_eligible'] is None
    (tmp_path/'jobs/trial-3/verification.json').unlink()
    with pytest.raises(FileNotFoundError):module.publication_metadata(selected,{},'completed-series')


def test_incomplete_parent_cannot_claim_completed_even_with_five_children(tmp_path,monkeypatch):
    selected=completed_series(tmp_path,monkeypatch)
    (tmp_path/'jobs/series/job.json').write_text(json.dumps({'id':'series','status':'running'}))
    with pytest.raises(ValueError):module.publication_metadata(selected,{},'completed-series')


def test_completed_excluded_series_remains_excluded(tmp_path,monkeypatch):
    selected=completed_series(tmp_path,monkeypatch)
    value=module.publication_metadata(selected,{'primary_leaderboard_eligible':False,'classification':'development','reason':'original exclusion'},'completed-series')
    assert value['primary_leaderboard_eligible'] is False and 'excluded' in value['label']
