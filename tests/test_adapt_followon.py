import pytest
from pairing.adapt_followon import safe_predecessor


@pytest.mark.parametrize('status,trace,allowed',[
 ('completed_unscored','',True),('model_refusal','',True),('running','',False),
 ('infrastructure_error','RuntimeError: OpenRouter HTTP 402; no retry',True),
 ('infrastructure_error','RuntimeError: Independent replay invalid',False),
 ('infrastructure_error','RuntimeError: Reset differs from reference fingerprint',False),
 ('verification_invalid','',False),('invalid_precondition','',False)])
def test_followon_does_not_confuse_provider_failure_with_simulator_failure(status,trace,allowed):
    assert safe_predecessor({'status':status},trace)==allowed
