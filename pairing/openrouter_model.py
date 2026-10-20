"""Explicit OpenRouter participant for Gemini guinea-pig runs.

No count-only endpoint exists in inspected OpenRouter OpenAPI. A verified model-
aware counter is mandatory for SQLite sliding; post-hoc usage is not a substitute.
"""
from __future__ import annotations
import base64
import json
import math
import time
import uuid
from pathlib import Path
from pairing.context_history import POLICY,initialize,sliding_ask
from pairing.muse_model import credential
from pairing.openai_model import OpenAILearningModel

API_URL='https://openrouter.ai/api/v1/chat/completions'


class ModelError(RuntimeError):
    status = 'model_error'


class ModelRefusal(ModelError):
    status = 'model_refusal'


class ModelOutputLimit(ModelError):
    status = 'model_output_limit'


class OpenRouterLearningModel(OpenAILearningModel):
    def __init__(self,instructions,directory,*,model,context_window,max_output_tokens,
                 token_counter=None,provider=None,effort='high',context_policy=POLICY,ledger_path=None,spend_limit_usd=150.0,
                 input_price_per_million=2.0,output_price_per_million=10.0):
        if not isinstance(model,str) or not model.strip():raise ValueError('Explicit model required')
        if type(max_output_tokens) is not int or max_output_tokens<=0:raise ValueError('Positive output reserve required')
        self.ledger=None
        self.input_price=float(input_price_per_million)
        self.output_price=float(output_price_per_million)
        if not all(math.isfinite(x) and x>0 for x in (self.input_price,self.output_price,float(spend_limit_usd))):
            raise ValueError('Positive finite pinned prices and spending cap required')
        if ledger_path:
            from pairing.anthropic_model import SpendLedger
            self.ledger=SpendLedger(ledger_path,limit_usd=spend_limit_usd)
        self.provider_label="OpenRouter"
        self.key=credential()
        self.model=model
        self.instructions=instructions
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.context_window=context_window
        self.max_output_tokens=max_output_tokens
        self.effort=effort
        self.provider=provider
        self.calls=0;self.compactions=0;self.messages=[];self.usage=[]
        self.thread_id='openrouter-direct-'+str(uuid.uuid4())
        self.metadata={'model':model,'model_provider':'openrouter','api_url':API_URL,
            'thread_id':self.thread_id,'persistent_thread':True,'max_output_tokens':max_output_tokens,
            'context_window':context_window,'reasoning_effort':effort,'provider_pin':provider,
            'credential_source':'trusted coordinator OpenRouter resolver; never robot workspace',
            'token_count_readiness':'requires verified supplied counter; no OpenRouter count endpoint verified',
            'provider_compaction':False,'fallback_models':[]}
        self.approximate_counter=None
        if context_policy=='sqlite-sliding-or-conservative-v1':
            if token_counter is not None:raise ValueError('Conservative policy cannot accept a replacement counter')
            from pairing.openrouter_budgeting import ConservativeCounter
            self.approximate_counter=ConservativeCounter(directory)
            token_counter=self.approximate_counter
            self.metadata['token_count_readiness']=self.approximate_counter.metadata
        elif context_policy=='sqlite-sliding-or-google-count-v2':
            if token_counter is not None:raise ValueError('Mapped Google count policy cannot accept replacement counter')
            from pairing.openrouter_google_count import GoogleMappedCounter
            self.approximate_counter=GoogleMappedCounter(directory,max_output_tokens=max_output_tokens,effort=effort,
                deadline=lambda:getattr(self,'_history_deadline',time.monotonic()+25))
            token_counter=self.approximate_counter
            self.metadata['token_count_readiness']=self.approximate_counter.metadata
        initialize(self,context_policy=context_policy,context_window=context_window,
                   output_reserve=max_output_tokens,history_attribute='messages',token_counter=token_counter)

    @staticmethod
    def _content(text,images):
        content=[{'type':'text','text':text}]
        for image in images:
            image=Path(image);mime='image/jpeg' if image.suffix.lower() in ('.jpg','.jpeg') else 'image/png'
            content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(image.read_bytes()).decode(),'detail':'high'}})
        return content

    @sliding_ask
    def ask(self,text,*,images=(),schema=None,timeout=180,as_tool_output=False):
        del as_tool_output
        request_started=time.monotonic()
        request_deadline=min(request_started+timeout,getattr(self,'_history_deadline',request_started+timeout))
        if schema is None:
            from pairing.cohort50 import SCHEMA
            schema=SCHEMA
        pending={'role':'user','content':self._content(text,images)}
        provider={'allow_fallbacks':False,'require_parameters':True}
        if self.provider:provider['only']=[self.provider]
        body={'model':self.model,'messages':[{'role':'system','content':self.instructions},*self.messages,pending],
              'max_tokens':self.max_output_tokens,'reasoning':{'effort':self.effort},'provider':provider,
              'plugins':[{'id':'context-compression','enabled':False}],
              'response_format':{'type':'json_schema','json_schema':{'name':'robot_decision','strict':True,'schema':schema}}}
        estimated=None
        if self.approximate_counter:
            self.approximate_counter.directory=Path(self.directory)
            estimated=self.approximate_counter.count_body(body)
            if estimated+self.max_output_tokens+4096>self.context_window:
                raise RuntimeError('Conservative context budget exceeded after input assembly; no generation sent')
        remaining=request_deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('OpenRouter context counting exhausted original request deadline; inference not sent')
        reservation=None
        if self.ledger:
            # Reserve full model-window input cost, not an unproven estimate.
            # This remains safe if the conservative context proxy underestimates.
            maximum=(self.context_window*self.input_price+self.max_output_tokens*self.output_price)/1_000_000
            reservation=self.ledger.reserve(maximum,label=f'{self.model}:{Path(self.directory).name}:call{self.calls+1}')
        started=time.monotonic()
        payload=self._post(API_URL,body,timeout=remaining)
        self.calls+=1
        with (self.directory/'provider-responses.jsonl').open('a') as stream:
            stream.write(json.dumps({'call':self.calls,'seconds':time.monotonic()-started,'response':payload})+'\n')
        # Unknown transport outcomes keep their full reservation. Never rerun.
        if self.ledger:
            receipt=payload.get('usage') or {}
            cost=receipt.get('cost')
            if type(cost) not in (int,float) or not math.isfinite(cost) or cost<0 or type(receipt.get('prompt_tokens')) is not int or type(receipt.get('completion_tokens')) is not int:
                with (self.directory/'unsettled-spend.jsonl').open('a') as stream:
                    stream.write(json.dumps({'call':self.calls,'reservation':reservation,'usage':receipt,'reason':'missing valid cost or token usage; reservation retained'})+'\n')
                raise RuntimeError('OpenRouter cost/usage receipt missing; reservation retained; stopped')
            self.ledger.commit(reservation,float(cost),usage={'input_tokens':receipt['prompt_tokens'],'output_tokens':receipt['completion_tokens']})
        if payload.get('error') or payload.get('model')!=self.model:raise RuntimeError('OpenRouter error or model substitution; no operation executed')
        choices=payload.get('choices') or []
        if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict):raise ModelError('Expected one OpenRouter choice')
        message=choices[0].get('message') or {}
        history={key:message[key] for key in ('role','content','reasoning','reasoning_details','refusal') if key in message}
        if history.get('refusal') is None:
            history.pop('refusal', None)
        history.setdefault('role','assistant')
        self.messages.extend([pending,history])
        self.last_provider_thinking_trace={'call':self.calls,
            **{k:message[k] for k in ('reasoning','reasoning_details') if k in message},
            'policy':'Verbatim provider-exposed fields; encrypted blocks are opaque, not readable thoughts'}
        with (self.directory/'provider-thinking.jsonl').open('a') as stream:
            stream.write(json.dumps({'call':self.calls,**{k:message[k] for k in ('reasoning','reasoning_details') if k in message}})+'\n')
        usage=payload.get('usage') or {};self.usage.append(usage)
        if self.approximate_counter:self.approximate_counter.observe(usage.get('prompt_tokens'),estimated=estimated,call=self.calls)
        with (self.directory/'model-usage.jsonl').open('a') as stream:stream.write(json.dumps({'call':self.calls,'usage':usage,'provider':payload.get('provider')})+'\n')
        choice = choices[0]
        if choice.get('finish_reason') == 'content_filter' or choice.get('native_finish_reason') == 'SAFETY' or message.get('refusal'):
            raise ModelRefusal('OpenRouter model refusal/content filter; no operation executed')
        if choice.get('finish_reason') == 'length' or choice.get('native_finish_reason') == 'MAX_TOKENS':
            raise ModelOutputLimit('OpenRouter output limit; no operation executed')
        if choice.get('finish_reason') != 'stop':
            raise ModelError('OpenRouter incomplete decision; no operation executed')
        try:
            decision=json.loads(message.get('content',''))
        except (ValueError, TypeError):
            raise ModelError('Invalid OpenRouter decision JSON; no operation executed') from None
        if not isinstance(decision,dict) or set(decision)!=set(schema['required']) or not all(isinstance(v,str) for v in decision.values()):
            raise ModelError('Invalid OpenRouter robot decision')
        if decision['operation'] not in schema['properties']['operation']['enum']:raise ModelError('Unsupported robot operation')
        return decision

    def usage_summary(self):
        return {'calls':self.calls,'compactions':0,'reported_context_windows':[self.context_window],
                'cumulative_token_usage':{key:sum(x.get(key,0) for x in self.usage)
                    for key in ('prompt_tokens','completion_tokens','total_tokens')}}
