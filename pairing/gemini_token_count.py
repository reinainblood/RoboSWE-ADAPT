"""Native Gemini countTokens, deliberately NOT an OpenRouter-equivalent counter.

Caller must supply the exact native generateContentRequest that will be generated.
No Chat Completions translation or heuristic counting is performed here.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'


class GeminiCountRequestError(RuntimeError):
    def __init__(self, message, *, diagnostic, retryable, retry_after=1.0):
        super().__init__(message)
        self.diagnostic=diagnostic
        self.retryable=retryable
        self.retry_after=retry_after


class GeminiNativeCounter:
    def __init__(self, model, *, api_key_env='GEMINI_API_KEY', receipt_path=None):
        if not re.fullmatch(r'gemini-[A-Za-z0-9._-]+', model):
            raise ValueError('Explicit native Gemini model ID required')
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', api_key_env):
            raise ValueError('api_key_env must name an environment variable')
        if not os.environ.get(api_key_env):
            raise RuntimeError(f'{api_key_env} required for native Google countTokens; no inference started')
        self.model = model
        self.api_key_env = api_key_env
        self.receipt_path = Path(receipt_path) if receipt_path else None

    def count_native_request(self, request, *, timeout=30):
        """Count full native input including system, images, schema and tool config.

        The server determines supported fields. Unsupported requests fail rather
        than falling back to text-only counting or dropping thought signatures.
        """
        if not isinstance(request, dict) or request.get('model') != 'models/' + self.model:
            raise ValueError('Native request model must match counter model')
        if not isinstance(request.get('contents'), list) or not request['contents']:
            raise ValueError('Native request requires nonempty contents')
        if 'messages' in request or 'response_format' in request:
            raise ValueError('OpenRouter/OpenAI request conversion is not verified')
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError('Native token-count credential absent; no request sent')
        raw = json.dumps({'generateContentRequest': request}, allow_nan=False, separators=(',', ':')).encode()
        req = urllib.request.Request(BASE_URL + '/models/' + self.model + ':countTokens', data=raw,
            headers={'x-goog-api-key': key, 'content-type': 'application/json'})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            def clean(value):
                return re.sub(r'(?:sk-[A-Za-z0-9_-]{10,}|AIza[A-Za-z0-9_-]{20,})','[REDACTED]',str(value).replace(key,'[REDACTED]'))[:1000]
            diagnostic={'type':type(exc).__name__,'message':clean(exc)}
            reason=getattr(exc,'reason',None)
            if reason is not None:diagnostic.update(reason_type=type(reason).__name__,reason=clean(reason))
            status=exc.code if isinstance(exc,urllib.error.HTTPError) else None
            if status is not None:diagnostic['http_status']=status
            delay=1.0
            if isinstance(exc,urllib.error.HTTPError) and exc.headers:
                try:delay=max(0.0,float(exc.headers.get('retry-after','1')))
                except ValueError:delay=1.0
            raise GeminiCountRequestError('Gemini count-only request failed; no completion generated',
                diagnostic=diagnostic,retryable=status is None or status in (429,500,502,503,504,529),retry_after=delay) from RuntimeError(json.dumps(diagnostic))
        count = payload.get('totalTokens')
        if type(count) is not int or count < 0:
            raise ValueError('Native Gemini counter omitted valid totalTokens')
        if self.receipt_path:
            self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
            with self.receipt_path.open('a') as stream:
                stream.write(json.dumps({'method':'google-native-countTokens-v1', 'model':self.model,
                    'request_sha256':hashlib.sha256(raw).hexdigest(), 'input_tokens':count,
                    'seconds':time.monotonic()-start, 'openrouter_translation_verified':False})+'\n')
        return count
