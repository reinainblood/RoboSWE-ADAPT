from unittest.mock import patch
import pytest
from roboswe.adaptive_transport import docker_command, episode_factory


def test_docker_is_isolated_simulator(tmp_path):
    with patch('roboswe.adaptive_transport.docker_path', return_value='docker'):
        cmd = docker_command({'seed': 2}, {'docker_simulator_image': 'sha256:' + 'a'*64}, tmp_path, 'owned')
    assert cmd[cmd.index('--network')+1] == 'none'
    assert '--read-only' not in cmd  # Trusted upstream writes generated object XML; disposable layer.
    assert cmd[-2:] == ['pairing.modal_episode', 'worker']
    assert sum(x == '--mount' for x in cmd) == 1
    assert not any('secret' in x.lower() or 'docker.sock' in x for x in cmd)


def test_mutable_docker_image_rejected(tmp_path):
    with pytest.raises(ValueError, match='immutable'):
        docker_command({}, {'docker_simulator_image': 'latest'}, tmp_path, 'owned')


def test_public_modal_does_not_use_private_defaults(tmp_path):
    with patch('roboswe.adaptive_transport.RemoteEpisode') as remote:
        episode_factory({}, {'execution_backend':'modal','modal_image_id':'im-public'}, tmp_path, 1, [])
    spec = remote.call_args.args[1]
    assert spec['modal_environment'] == 'FDR'
    assert spec['modal_profile'] != 'snorkelai'


def test_missing_backend_or_image_fails_before_launch(tmp_path):
    with pytest.raises(ValueError, match='execution_backend'):
        episode_factory({}, {}, tmp_path, 1, [])
    with pytest.raises(ValueError, match='modal_image_id'):
        episode_factory({}, {'execution_backend':'modal'}, tmp_path, 1, [])


def test_public_modal_requires_verified_remote_video_mode(tmp_path):
    with patch('roboswe.adaptive_transport.RemoteEpisode') as remote:
        episode_factory({}, {'execution_backend':'modal','modal_image_id':'im-public'}, tmp_path, 1, [])
    assert remote.call_args.args[1]['artifact_transfer']=='remote-video'
    assert remote.call_args.args[1]['retain_local_videos'] is True
    with pytest.raises(ValueError,match='verified'):
        episode_factory({}, {'execution_backend':'modal','modal_image_id':'im-public','artifact_transfer':'all'},tmp_path,1,[])


def test_finalized_modal_archive_requires_verified_receipt(tmp_path):
    from roboswe.adaptive_transport import archive_finalized_attempt,ArtifactDurabilityError
    from types import SimpleNamespace
    execution={'backend':'modal','modal_profile':'selected','modal_environment':'FDR','modal_volume':'selected-volume'}
    with patch('roboswe.adaptive_transport.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='{}',stderr='')) as run:
        with pytest.raises(ArtifactDurabilityError):archive_finalized_attempt(tmp_path,execution)
    assert run.call_args.kwargs['env']['MODAL_PROFILE']=='selected'
    assert run.call_args.kwargs['env']['MODAL_ENVIRONMENT']=='FDR'


@pytest.mark.parametrize('wrong',['job_id','scope','environment','modal_profile','volume','sha256','bytes','readback'])
def test_public_archive_rejects_wrong_bound_receipt(tmp_path,wrong):
    import json
    from types import SimpleNamespace
    from roboswe.adaptive_transport import archive_finalized_attempt,ArtifactDurabilityError
    execution={'backend':'modal','modal_profile':'community','modal_environment':'evaluation','modal_volume':'custom-volume'}
    receipt={'job_id':tmp_path.name,'scope':'finalized-job','environment':'evaluation','modal_profile':'community','volume':'custom-volume',
             'archive':{'sha256':'a'*64,'bytes':10},'remote_readback_verified':True,'restore_test_passed':True}
    if wrong in ('sha256','bytes'):receipt['archive'][wrong]='invalid'
    elif wrong=='readback':receipt['remote_readback_verified']=False
    else:receipt[wrong]='different'
    with patch('roboswe.adaptive_transport.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(receipt),stderr='')):
        with pytest.raises(ArtifactDurabilityError):archive_finalized_attempt(tmp_path,execution)


def test_public_custom_archive_passes_exact_scope(tmp_path):
    import json
    from types import SimpleNamespace
    from roboswe.adaptive_transport import archive_finalized_attempt
    execution={'backend':'modal','modal_profile':'community','modal_environment':'evaluation','modal_volume':'custom-volume'}
    receipt={'job_id':tmp_path.name,'scope':'finalized-job','environment':'evaluation','modal_profile':'community','volume':'custom-volume',
             'archive':{'sha256':'a'*64,'bytes':10},'remote_readback_verified':True,'restore_test_passed':True}
    with patch('roboswe.adaptive_transport.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(receipt),stderr='')) as run:
        assert archive_finalized_attempt(tmp_path,execution)==receipt
    args=run.call_args.args[0]
    assert args[args.index('--environment')+1]=='evaluation'
    assert args[args.index('--volume')+1]=='custom-volume'
