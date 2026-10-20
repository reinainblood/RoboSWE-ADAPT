import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'website/scripts/export-results.py'
spec = importlib.util.spec_from_file_location('public_result_export', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_modal_publication_omits_private_coordinator_configuration():
    result = module.sanitize({'execution': {
        'backend': 'modal', 'simulator_image': 'im-example',
        'modal_sdk_python': '/private/operator/python',
        'modal_profile': 'private-account', 'modal_volume': 'private-volume',
        'modal_environment': 'FDR',
    }})
    assert result == {'execution': {
        'backend': 'modal', 'simulator_image': 'im-example', 'modal_environment': 'FDR',
    }}


def test_unexpected_external_absolute_path_still_refused():
    with pytest.raises(ValueError, match='outside the project'):
        module.sanitize({'evidence': '/private/operator/unexpected.json'})
