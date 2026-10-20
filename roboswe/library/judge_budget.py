"""Trusted budget wrapper for optional OpenRouter video audits."""
import json
import math
from pathlib import Path
from uuid import uuid4

from pairing.anthropic_model import SpendLedger
from .glimpse_judge import BACKENDS, _request

# Public OpenRouter /api/v1/models metadata checked 2026-10-20.
# Reserve a whole input context plus the complete output allowance, without cache discounts.
MODEL_BOUNDS = {
    'gemini': {'context': 1048576, 'input_per_million': 2.0, 'output_per_million': 10.0},
    'glimpse': {'context': 262144, 'input_per_million': 0.6, 'output_per_million': 3.0},
}


def budgeted_request(*, ledger_path, limit_usd, receipts, judge_backend='gemini', request_fn=None):
    if judge_backend not in MODEL_BOUNDS:
        raise ValueError('Unsupported budgeted judge backend')
    if isinstance(limit_usd, bool) or not math.isfinite(limit_usd) or limit_usd <= 0:
        raise ValueError('Judge spend cap must be a positive finite USD amount')
    ledger = SpendLedger(ledger_path, limit_usd=limit_usd)
    # Existing ledger budgets are immutable; fail at server setup, not after a paid call.
    if ledger.snapshot()['limit_usd'] != limit_usd:
        raise ValueError('Judge spend cap differs from the existing ledger')
    receipts = Path(receipts)
    bounds = MODEL_BOUNDS[judge_backend]
    model, provider, _ = BACKENDS[judge_backend]

    def request(payload, *, api_key, timeout):
        if payload.get('model') != model or payload.get('provider', {}).get('only') != [provider] or payload.get('provider', {}).get('allow_fallbacks') is not False:
            raise ValueError('Judge request does not match the budgeted model and provider')
        output = payload.get('max_tokens')
        if type(output) is not int or not 1 <= output <= bounds['context']:
            raise ValueError('Invalid judge output allowance')
        maximum = (bounds['context'] * bounds['input_per_million'] + output * bounds['output_per_million']) / 1_000_000
        receipts.mkdir(parents=True, exist_ok=True)
        path = receipts / (uuid4().hex + '.json')
        reservation = ledger.reserve(maximum, label=f'visual-audit:{model}')
        record = {'model': model, 'provider': provider, 'reservation': reservation, 'reserved_usd': maximum, 'bounds': bounds, 'max_tokens': output, 'status': 'reserved'}
        def save():
            text = json.dumps(record, indent=2, allow_nan=False)
            # Never persist credentials, even if an unexpected upstream envelope echoes one.
            if api_key:
                text = text.replace(api_key, '[REDACTED]')
            path.write_text(text + '\n')
        save()
        try:
            response = (request_fn or _request)(payload, api_key=api_key, timeout=timeout)
            if isinstance(response, dict):
                record['response'] = {k: response[k] for k in ('id', 'model', 'provider', 'usage', 'choices') if k in response}
            record['status'] = 'response_received_cost_unsettled'
            save()  # Archive the provider output before parsing the judgment or settling cost.
            usage = response.get('usage', {}) if isinstance(response, dict) else {}
            cost = usage.get('cost') if isinstance(usage, dict) else None
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                raise ValueError('Judge response lacks a finite nonnegative provider cost; reservation held')
            ledger.commit(reservation, cost, usage={})
            record.update(status='settled', cost_usd=cost)
            save()
            return response
        except BaseException as exc:
            record['error_type'] = type(exc).__name__
            if record['status'] != 'settled':
                record['status'] = 'uncertain_reservation_held'
            save()
            raise
    return request
