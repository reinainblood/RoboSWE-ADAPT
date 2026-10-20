import pytest
from pairing.adapt_modal_pair import classify

@pytest.mark.parametrize('status,error,expected',[
 ('completed_unscored','','completed'),('completed','','completed'),('model_refusal','','provider_blocked'),
 ('infrastructure_error','OpenRouter HTTP 402; no retry','provider_blocked'),
 ('infrastructure_error','Anthropic HTTP 429','provider_blocked'),
 ('artifact_durability_error','readback failed','blocked_infrastructure'),
 ('verification_invalid','Independent replay invalid','blocked_infrastructure'),
 ('infrastructure_error','Reset differs from reference fingerprint','blocked_infrastructure'),
 ('stopped_child_result','','blocked_infrastructure')])
def test_provider_stop_vs_infrastructure_stop(status,error,expected):
 assert classify(status,error)==expected
