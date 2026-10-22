"""Native Gemini generation and countTokens share one request builder.

Exact provider parts (including returned thought signatures) persist in SQLite.
No OpenRouter translation, provider compaction, fallback, or inference retries.
"""
from __future__ import annotations
import base64
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from pairing.context_history import POLICY,initialize,sliding_ask
from pairing.gemini_token_count import GeminiNativeCounter,BASE_URL
from pairing.openai_model import OpenAILearningModel
from pairing.anthropic_model import SpendLedger


def credential() -> str:
    """Resolve the native Google key in the trusted coordinator only."""
    if os.environ.get('GEMINI_API_KEY'):
        return os.environ['GEMINI_API_KEY']
    result = subprocess.run(
        ['/usr/bin/security', 'find-generic-password', '-s', 'zsh_secrets',
         '-a', 'GEMINI_API_KEY', '-w'], capture_output=True, timeout=20)
    value = result.stdout.decode().strip() if result.returncode == 0 else ''
    if not value:
        raise RuntimeError('GEMINI_API_KEY is unavailable in macOS Keychain; value suppressed')
    return value


class GoogleLearningModel(OpenAILearningModel):
    def __init__(self,instructions,directory,*,model,context_window,max_output_tokens,
                 effort='high',token_counter=None,ledger_path=None,spend_limit_usd=125.0,
                 input_price_per_million=2.0,output_price_per_million=10.0):
        if not os.environ.get('GEMINI_API_KEY'):raise RuntimeError('GEMINI_API_KEY required in trusted coordinator')
        if not ledger_path:raise ValueError('Google participant requires explicit shared spending ledger')
        if type(context_window) is not int or type(max_output_tokens) is not int or max_output_tokens<=0 or context_window<=max_output_tokens+4096:raise ValueError('Invalid context/output limits')
        if token_counter is not None:raise ValueError('Native Google participant uses its own exact request counter')
        self.input_price=float(input_price_per_million);self.output_price=float(output_price_per_million)
        if not all(math.isfinite(x) and x>0 for x in (self.input_price,self.output_price,float(spend_limit_usd))):raise ValueError('Positive finite pricing and spending cap required')
        self.provider_label='Google Gemini';self.model=model;self.key=os.environ['GEMINI_API_KEY']
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.instructions=instructions;self.context_window=context_window;self.max_output_tokens=max_output_tokens
        self.effort=effort;self.messages=[];self.calls=0;self.compactions=0;self.usage=[]
        self.thread_id='google-direct-'+str(uuid.uuid4())
        self.ledger=SpendLedger(ledger_path,limit_usd=spend_limit_usd)
        self.counter=GeminiNativeCounter(model,receipt_path=self.directory/'token-counts.jsonl')
        self.metadata={'model':model,'model_provider':'google-direct','thread_id':self.thread_id,
            'context_window':context_window,'max_output_tokens':max_output_tokens,'provider_compaction':False,
            'fallback_models':[],'pricing':{'input_per_million':self.input_price,'output_per_million':self.output_price,
            'source':'https://ai.google.dev/gemini-api/docs/pricing','cache_discount_applied':False},
            'shared_spend_limit_usd':spend_limit_usd,'reasoning_policy':'Returned thought summaries and signatures preserved verbatim; hidden thinking not reconstructed'}
        initialize(self,context_policy=POLICY,context_window=context_window,output_reserve=max_output_tokens,
                   history_attribute='messages',token_counter=self._count_history_tokens)

    @staticmethod
    def _parts(text,images):
        parts=[{'text':text}]
        for image in images:
            p=Path(image)
            parts.append({'inlineData':{'mimeType':'image/jpeg' if p.suffix.lower() in ('.jpg','.jpeg') else 'image/png',
                         'data':base64.b64encode(p.read_bytes()).decode()}})
        return parts

    def _native_request(self,messages,pending,instructions,schema):
        return {'model':'models/'+self.model,'systemInstruction':{'parts':[{'text':instructions}]},
            'contents':[*messages,{'role':'user','parts':self._parts(pending['text'],pending['images'])}],
            'generationConfig':{'maxOutputTokens':self.max_output_tokens,'candidateCount':1,
                'responseMimeType':'application/json','responseJsonSchema':schema,
                'thinkingConfig':{'thinkingLevel':self.effort.upper(),'includeThoughts':True}}}

    def _count_history_tokens(self,*,messages,pending,overhead):
        self.counter.receipt_path=self.directory/'token-counts.jsonl'
        remaining=getattr(self,'_history_deadline',time.monotonic()+30)-time.monotonic()
        if remaining<=0:raise TimeoutError('Google count deadline exhausted')
        return self.counter.count_native_request(self._native_request(messages,pending,overhead['instructions'],overhead['schema']),timeout=min(30,remaining))

    def _generate(self,request,timeout):
        body={k:v for k,v in request.items() if k!='model'}
        req=urllib.request.Request(BASE_URL+'/models/'+self.model+':generateContent',
            data=json.dumps(body,separators=(',',':')).encode(),headers={'x-goog-api-key':self.key,'content-type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
        except urllib.error.HTTPError as exc:
            diagnostic=sanitize_google_error(exc,self.key)
            with (self.directory/'provider-http-errors.jsonl').open('a') as stream:
                stream.write(json.dumps({'status':exc.code,'retry':False,'endpoint':'generateContent','provider_error':diagnostic})+'\n')
            raise RuntimeError(f'Google generation HTTP {exc.code}; no retry; reservation retained') from None
        except urllib.error.URLError:
            raise RuntimeError('Google generation transport failed; no retry; reservation retained') from None

    @sliding_ask
    def ask(self,text,*,images=(),schema=None,timeout=180,as_tool_output=False):
        del as_tool_output
        if schema is None:
            from pairing.cohort50 import SCHEMA
            schema=SCHEMA
        request=self._native_request(self.messages,{'text':text,'images':images},self.instructions,schema)
        self.counter.receipt_path=self.directory/'token-counts.jsonl'
        began=time.monotonic()
        count=self.counter.count_native_request(request,timeout=min(30,timeout))
        if count+self.max_output_tokens+4096>self.context_window:raise RuntimeError('Native assembled input exceeds context budget; generation not sent')
        remaining=timeout-(time.monotonic()-began)
        if remaining<=0:raise TimeoutError('Google counting exhausted generation deadline')
        # Full window reservation protects aggregate cap even if count/generation accounting differs.
        maximum=(self.context_window*self.input_price+self.max_output_tokens*self.output_price)/1e6
        reservation=self.ledger.reserve(maximum,label=f'{self.model}:{self.directory.name}:call{self.calls+1}')
        payload=self._generate(request,remaining);self.calls+=1
        with (self.directory/'provider-responses.jsonl').open('a') as stream:stream.write(json.dumps({'call':self.calls,'response':payload})+'\n')
        usage=payload.get('usageMetadata') or {};self.usage.append(usage)
        with (self.directory/'model-usage.jsonl').open('a') as stream:stream.write(json.dumps({'call':self.calls,'usage':usage,'model_version':payload.get('modelVersion')})+'\n')
        if (payload.get('promptFeedback') or {}).get('blockReason'):
            raise RuntimeError('Google provider blocked response; raw promptFeedback preserved; reservation retained; no action executed')
        inp=usage.get('promptTokenCount');out=usage.get('candidatesTokenCount');thought=usage.get('thoughtsTokenCount',0)
        if any(type(n) is not int or n<0 for n in (inp,out,thought)):
            raise RuntimeError('Google usage missing; reservation retained; no action executed')
        cost=(inp*self.input_price+(out+thought)*self.output_price)/1e6
        self.ledger.commit(reservation,cost,usage={'input_tokens':inp,'output_tokens':out+thought})
        candidates=payload.get('candidates') or []
        if len(candidates)!=1:raise RuntimeError('Expected one Google candidate')
        candidate=candidates[0];content=candidate.get('content') or {}
        self.messages.extend([request['contents'][-1],content])
        parts=content.get('parts') or []
        thinking=[part for part in parts if part.get('thought') or part.get('thoughtSignature')]
        self.last_provider_thinking_trace={'call':self.calls,'parts':thinking,'policy':'verbatim provider parts; signatures opaque'}
        with (self.directory/'provider-thinking.jsonl').open('a') as stream:stream.write(json.dumps(self.last_provider_thinking_trace)+'\n')
        if candidate.get('finishReason')!='STOP':raise RuntimeError('Google candidate incomplete; no action executed')
        text=''.join(p.get('text','') for p in parts if not p.get('thought'))
        decision=json.loads(text)
        if not isinstance(decision,dict) or set(decision)!=set(schema['required']) or not all(isinstance(v,str) for v in decision.values()):raise RuntimeError('Invalid Google structured decision')
        if decision['operation'] not in schema['properties']['operation']['enum']:raise RuntimeError('Unsupported Google operation')
        return decision

    def usage_summary(self):
        return {'calls':self.calls,'compactions':0,'reported_context_windows':[self.context_window],
            'cumulative_token_usage':{k:sum(u.get(k,0) for u in self.usage) for k in ('promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount')}}


def sanitize_google_error(error, key):
    """Keep quota reason/metric/delay, never dump headers, request or arbitrary data."""
    try:payload=json.loads(error.read(65536))
    except Exception:return {'body_available':False}
    value=payload.get('error',{})
    def text(value):
        result=str(value).replace(key,'[REDACTED]') if key else str(value)
        return re.sub(r'AIza[A-Za-z0-9_-]+','[REDACTED]',result)[:2000]
    result={k:text(value[k]) for k in ('code','status','message') if k in value}
    details=[]
    for detail in value.get('details',[])[:10]:
        if not isinstance(detail,dict):continue
        item={k:text(detail[k]) for k in ('@type','reason','domain','retryDelay') if k in detail}
        if isinstance(detail.get('violations'),list):
            item['violations']=[{k:text(v[k]) for k in ('quotaMetric','quotaId','quotaValue','description') if k in v} for v in detail['violations'][:10] if isinstance(v,dict)]
        details.append(item)
    result['details']=details
    return result
