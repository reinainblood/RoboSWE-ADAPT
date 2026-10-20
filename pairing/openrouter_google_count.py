"""Versioned, calibrated Google count-only proxy for OpenRouter Gemini inputs.

The mapping is approximate: OpenRouter's native wire transformation is opaque.
Only counting uses this mapping; original generation messages remain untouched.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from pathlib import Path

from pairing.gemini_token_count import GeminiNativeCounter, GeminiCountRequestError

POLICY = 'sqlite-sliding-or-google-count-v2'


class GoogleMappedCounter:
    def __init__(self, directory, *, native_counter=None, max_output_tokens=32768, effort="high", deadline=None):
        self.directory = Path(directory)
        self.native = native_counter or GeminiNativeCounter('gemini-4-argon')
        self.max_output_tokens = max_output_tokens
        self.effort = effort
        self.deadline = deadline
        self.multiplier = 1.30
        self.last_native_count = None
        self.last_request_sha256 = None

    @staticmethod
    def native_request(body):
        if body.get('model') != 'google/gemini-4-argon':
            raise ValueError('Mapped counter requires exact google/gemini-4-argon model')
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages or messages[0].get('role') != 'system':
            raise ValueError('Exactly one initial system message required')
        if not isinstance(messages[0].get('content'), str):
            raise ValueError('System instruction must be text')
        fmt = body.get('response_format', {})
        if fmt.get('type') != 'json_schema' or not isinstance(fmt.get('json_schema', {}).get('schema'), dict):
            raise ValueError('JSON response schema required')
        contents = []
        for message in messages[1:]:
            role = message.get('role')
            allowed = {'role', 'content'} if role == 'user' else {'role', 'content', 'reasoning', 'reasoning_details'}
            # Older saved OpenRouter history includes explicit null refusal.
            # It carries no content; accept only that exact compatibility case.
            if role == 'assistant' and 'refusal' in message and message['refusal'] is None:
                allowed = allowed | {'refusal'}
            if role not in {'user', 'assistant'} or set(message) - allowed:
                raise ValueError('Unsupported message role or fields; no approximate fallback')
            parts = []
            if role == 'user':
                content = message.get('content')
                if not isinstance(content, list):
                    raise ValueError('Expected typed user content')
                for item in content:
                    if item.get('type') == 'text' and set(item) == {'type', 'text'} and isinstance(item['text'], str):
                        parts.append({'text': item['text']})
                    elif item.get('type') == 'image_url' and set(item) == {'type', 'image_url'}:
                        image = item['image_url']
                        if set(image) - {'url', 'detail'} or image.get('detail') != 'high':
                            raise ValueError('Unsupported image metadata')
                        prefix, data = image.get('url', '').split(',', 1)
                        if prefix not in {'data:image/jpeg;base64', 'data:image/png;base64'}:
                            raise ValueError('Only inline JPEG/PNG supported')
                        base64.b64decode(data, validate=True)
                        parts.append({'inlineData': {'mimeType': prefix[5:].split(';')[0], 'data': data}})
                    else:
                        raise ValueError('Unsupported user content part')
            else:
                if not isinstance(message.get('content'), str):
                    raise ValueError('Assistant content must be text')
                signature = None
                details = message.get('reasoning_details') or []
                if not isinstance(details, list):
                    raise ValueError('Malformed reasoning metadata')
                thought_text = []
                for detail in details:
                    if set(detail) - {'type', 'text', 'data', 'format', 'index'} or detail.get('format') != 'google-gemini-v1':
                        raise ValueError('Unknown reasoning metadata')
                    if detail.get('type') == 'reasoning.text' and isinstance(detail.get('text'), str):
                        parts.append({'text': detail['text'], 'thought': True})
                        thought_text.append(detail['text'])
                    elif detail.get('type') == 'reasoning.encrypted' and isinstance(detail.get('data'), str) and signature is None:
                        signature = detail['data']
                        base64.b64decode(signature, validate=True)
                    else:
                        raise ValueError('Ambiguous or unsupported reasoning part')
                if message.get('reasoning') and not thought_text:
                    raise ValueError('Reasoning without typed parts cannot be mapped safely')
                part = {'text': message['content']}
                if signature is not None:
                    part['thoughtSignature'] = signature
                parts.append(part)
            contents.append({'role': 'model' if role == 'assistant' else 'user', 'parts': parts})
        if not contents:
            raise ValueError('Native content required')
        return {'model': 'models/gemini-4-argon',
                'systemInstruction': {'parts': [{'text': messages[0]['content']}]},
                'contents': contents,
                'generationConfig': {'maxOutputTokens': body['max_tokens'],
                    'responseMimeType': 'application/json',
                    'responseJsonSchema': fmt['json_schema']['schema'],
                    'thinkingConfig': {'thinkingLevel': body['reasoning']['effort'].upper(), 'includeThoughts': True}}}

    def _receipt(self, row):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / 'context-estimates.jsonl').open('a') as stream:
            stream.write(json.dumps({'policy': POLICY, 'exact_native_count': False, **row}) + '\n')

    def count_body(self, body):
        request = self.native_request(body)
        digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        started=time.monotonic()
        deadline=started+180 if self.deadline is None else self.deadline()
        for attempt in (1,2):
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError("Context counting exhausted original deadline")
            timeout=min(90,remaining)
            try:
                count=self.native.count_native_request(request,timeout=timeout)
                break
            except GeminiCountRequestError as exc:
                delay=exc.retry_after
                retry=exc.retryable and attempt<2 and delay<=30 and deadline-time.monotonic()>delay
                self._receipt({'event':'count_transport_failure','transformed_request_sha256':digest,
                    'attempt':attempt,'seconds':time.monotonic()-started,'attempt_timeout_seconds':timeout,
                    'original_deadline_seconds_remaining':max(0,deadline-time.monotonic()),
                    'completion_generated':False,'will_retry':retry,'retry_delay_seconds':delay if retry else None,
                    'error':exc.diagnostic})
                if retry:
                    time.sleep(delay)
                    continue
                if deadline<=time.monotonic():raise TimeoutError('Context counting exhausted original deadline') from exc
                raise
        if type(count) is not int or count < 0:
            raise ValueError('Invalid native count; no fallback')
        estimate = math.ceil(count * self.multiplier) + 8192
        self.last_native_count = count
        self.last_request_sha256 = digest
        self._receipt({'event': 'count', 'transformed_request_sha256': digest,
                       'native_input_tokens': count, 'estimated_input_units': estimate,
                       'multiplier': self.multiplier, 'additive_reserve': 8192,
                       'count_attempts':attempt,'count_seconds':time.monotonic()-started,
                       'count_transport_policy':'at most2 attempts of90s within original episode deadline'})
        return estimate

    def __call__(self, *, messages, pending, overhead):
        from pairing.openrouter_model import OpenRouterLearningModel
        return self.count_body({'model': 'google/gemini-4-argon',
            'messages': [{'role': 'system', 'content': overhead['instructions']}, *messages,
                         {'role': 'user', 'content': OpenRouterLearningModel._content(pending['text'], pending['images'])}],
            # Output and thinking config are count-only metadata; actual caller's
            # final assembled count_body captures exact configured values.
            'max_tokens': self.max_output_tokens, 'reasoning': {'effort': self.effort},
            'response_format': {'type': 'json_schema', 'json_schema': {'schema': overhead['schema']}}})

    def observe(self, actual, *, estimated, call):
        if type(actual) is not int or actual < 0 or self.last_native_count is None:
            raise ValueError('Valid usage and preceding count required')
        if self.last_native_count == 0 and actual:
            raise ValueError('Zero native count contradicts actual prompt usage')
        before = self.multiplier
        if self.last_native_count:
            self.multiplier = max(self.multiplier, 1.1 * actual / self.last_native_count)
        self._receipt({'event': 'calibration', 'call': call,
                       'transformed_request_sha256': self.last_request_sha256,
                       'native_input_tokens': self.last_native_count,
                       'estimated_input_units': estimated, 'actual_prompt_tokens': actual,
                       'multiplier_before': before, 'multiplier_after': self.multiplier,
                       'underestimated': actual > estimated})

    @property
    def metadata(self):
        return {'method': POLICY, 'model': 'google/gemini-4-argon',
                'exact_native_count': False, 'native_translation_verified': False,
                'native_upper_bound_proven': False, 'initial_multiplier': 1.30,
                'additive_reserve': 8192, 'calibration': 'upward only: 1.1 times actual/native',
                'generation_history': 'OpenRouter messages unchanged; mapping is count-only',
                'failure_policy': 'stop; no fallback or inference retry',
                'count_transport_policy':'at most2 transient count-only attempts of90s within original episode deadline'}
