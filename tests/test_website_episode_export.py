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
