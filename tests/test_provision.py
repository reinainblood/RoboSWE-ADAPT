import json
from pathlib import Path
from unittest.mock import patch

import pytest

from roboswe.provision import ASSET_FOLDERS, prepare, plan, validate_checkout


def root(tmp_path):
    (tmp_path / 'robot_session').mkdir()
    (tmp_path / 'robot_session/requirements.txt').write_text('')
    return tmp_path


def test_plan_never_mutates(tmp_path):
    directory = root(tmp_path)
    with patch('roboswe.provision.subprocess.run') as call:
        result = prepare(directory)
    assert result['status'] == 'planned'
    assert not call.called
    assert not (directory / 'upstream').exists()
    assert result['commands'][-1][-2:] == ['--type', 'all']
    assert len(result['repositories']) == 2


def test_reject_incomplete_existing_without_mutation(tmp_path):
    directory = root(tmp_path)
    (directory / 'upstream/robocasa').mkdir(parents=True)
    with patch('roboswe.provision.validate_checkout'), pytest.raises(ValueError, match='incomplete'):
        plan(directory)


def test_existing_complete_is_read_only(tmp_path):
    directory = root(tmp_path)
    (directory / 'upstream/robosuite').mkdir(parents=True)
    for folder in ASSET_FOLDERS:
        path = directory / 'upstream/robocasa/robocasa/models/assets' / folder
        path.mkdir(parents=True)
        (path / 'example').write_text('asset')
    with patch('roboswe.provision.validate_checkout'):
        assert plan(directory)['commands'] == []


def test_revision_mismatch():
    with patch('roboswe.provision.subprocess.check_output', return_value='wrong'), pytest.raises(ValueError, match='revision'):
        validate_checkout(Path('x'), 'https://example.org', 'pinned')


def test_failed_command_retains_receipt(tmp_path):
    directory = root(tmp_path)
    receipt = directory / 'receipt.json'
    with patch('roboswe.provision.subprocess.run', side_effect=RuntimeError('failure')), pytest.raises(RuntimeError):
        prepare(directory, run=True, output=receipt)
    data = json.loads(receipt.read_text())
    assert data['status'] == 'failed'
    assert data['completed_commands'] == 0


def test_requires_receipt_before_commands(tmp_path):
    with pytest.raises(ValueError, match='--out'):
        prepare(root(tmp_path), run=True)


def test_declined_download_cannot_claim_prepared(tmp_path):
    directory = root(tmp_path)
    receipt = directory / 'receipt.json'
    with patch('roboswe.provision.subprocess.run'), patch('roboswe.provision.validate_checkout'), pytest.raises(ValueError, match='Asset folders'):
        prepare(directory, run=True, output=receipt)
    assert json.loads(receipt.read_text())['status'] == 'failed'


def test_existing_receipt_never_overwritten(tmp_path):
    directory = root(tmp_path)
    receipt = directory / 'receipt.json'
    receipt.write_text('original')
    with patch('roboswe.provision.subprocess.run') as call, pytest.raises(FileExistsError):
        prepare(directory, run=True, output=receipt)
    assert receipt.read_text() == 'original'
    assert not call.called


def test_cli_prepare_defaults_to_plan():
    from roboswe.cli import parser_for_cli
    args = parser_for_cli().parse_args(['prepare'])
    assert not args.run
    assert args.python == 'python3.11'
