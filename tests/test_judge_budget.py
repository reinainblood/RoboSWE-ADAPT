import json
from pathlib import Path

import pytest

from pairing.anthropic_model import BudgetLimitReached, SpendLedger
from roboswe.library.judge_budget import budgeted_request


def payload():
    return {'model': 'google/gemini-4-argon', 'provider': {'only': ['google-vertex/global'], 'allow_fallbacks': False}, 'max_tokens': 8192}


def wrapper(tmp_path, callback, limit=5):
    return budgeted_request(ledger_path=tmp_path/'ledger.json', limit_usd=limit, receipts=tmp_path/'receipts', request_fn=callback)


def test_archive_then_settle(tmp_path):
    response = {'usage': {'cost': .05}, 'choices': [{'message': {'content': 'secret-key'}}], 'headers': {'api-key': 'secret-key'}}
    call = wrapper(tmp_path, lambda *a, **kw: response)
    assert call(payload(), api_key='secret-key', timeout=1) is response
    receipt = next((tmp_path/'receipts').glob('*.json')).read_text()
    assert 'secret-key' not in receipt
    assert 'headers' not in receipt
    assert json.loads(receipt)['status'] == 'settled'
    ledger = SpendLedger(tmp_path/'ledger.json', limit_usd=5).snapshot()
    assert ledger['spent_usd'] == .05 and ledger['reserved_usd'] == 0


@pytest.mark.parametrize('cost', [None, -1, '0.1', True])
def test_missing_or_invalid_cost_holds(tmp_path, cost):
    call = wrapper(tmp_path, lambda *a, **kw: {'usage': {'cost': cost}, 'choices': []})
    with pytest.raises(ValueError):
        call(payload(), api_key='key', timeout=1)
    ledger = SpendLedger(tmp_path/'ledger.json', limit_usd=5).snapshot()
    assert ledger['spent_usd'] == 0 and ledger['reserved_usd'] == pytest.approx(2.179072)
    assert 'response' in json.loads(next((tmp_path/'receipts').glob('*.json')).read_text())


def test_refusal_with_valid_cost_is_billed(tmp_path):
    call = wrapper(tmp_path, lambda *a, **kw: {'usage': {'cost': .02}, 'choices': [{'finish_reason': 'refusal'}]})
    call(payload(), api_key='key', timeout=1)
    assert SpendLedger(tmp_path/'ledger.json', limit_usd=5).snapshot()['spent_usd'] == .02


def test_cap_prevents_provider_call(tmp_path):
    def forbidden(*a, **kw):
        pytest.fail('provider invoked')
    call = wrapper(tmp_path, forbidden, limit=1)
    with pytest.raises(BudgetLimitReached):
        call(payload(), api_key='key', timeout=1)


def test_transport_failure_holds(tmp_path):
    def fail(*a, **kw):
        raise TimeoutError()
    call = wrapper(tmp_path, fail)
    with pytest.raises(TimeoutError):
        call(payload(), api_key='key', timeout=1)
    assert SpendLedger(tmp_path/'ledger.json', limit_usd=5).snapshot()['reserved_usd'] > 0


def test_cap_mismatch_rejected(tmp_path):
    wrapper(tmp_path, None)
    with pytest.raises(ValueError, match='differs'):
        wrapper(tmp_path, None, limit=6)


def test_wrong_provider_not_called(tmp_path):
    call = wrapper(tmp_path, lambda *a, **kw: pytest.fail('called'))
    data = payload(); data['provider']['only'] = ['other']
    with pytest.raises(ValueError, match='match'):
        call(data, api_key='key', timeout=1)
