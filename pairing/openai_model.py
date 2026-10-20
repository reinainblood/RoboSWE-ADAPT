"""Direct, stateless OpenAI Responses participant with exact local history.

Uses official input token counting, never server-side compaction or previous_response_id.
Only provider-returned reasoning summaries and opaque replay items are retained.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from pairing.context_history import POLICY, initialize, sliding_ask

API_URL = 'https://api.openai.com/v1/responses'


class OpenAILearningModel:
    def __init__(self, instructions, directory, *, model, context_window,
                 max_output_tokens, effort='high', reasoning_summary='detailed',
                 token_counter=None):
        if not isinstance(model,str) or not model.strip():
            raise ValueError('An explicit OpenAI model identifier is required')
        if type(context_window) is not int or type(max_output_tokens) is not int or max_output_tokens<=0:
            raise ValueError('Explicit integer context/output limits are required')
        key=os.environ.get('OPENAI_API_KEY','')
        if not key:
            raise RuntimeError('OPENAI_API_KEY is required in trusted coordinator environment')
        self.provider_label='OpenAI'
        self.key=key
        self.model=model
        self.instructions=instructions
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.context_window=context_window
        self.max_output_tokens=max_output_tokens
        self.effort=effort
        self.reasoning_summary=reasoning_summary
        self.messages=[]
        self.calls=0
        self.compactions=0
        self.thread_id='openai-direct-'+str(uuid.uuid4())
        self.usage=[]
        self.metadata={
            'model':model,'model_provider':'openai-direct','api_url':API_URL,
            'thread_id':self.thread_id,'persistent_thread':True,
            'conversation_transport':'local exact Responses API item replay',
            'store':False,'provider_compaction':False,'reasoning_effort':effort,
            'max_output_tokens':max_output_tokens,'context_window':context_window,
            'limits_source':'explicit run configuration; caller must pin verified model capabilities',
            'credential_source':'OPENAI_API_KEY in trusted coordinator environment only',
            'reasoning_policy':'Provider-exposed summaries and opaque encrypted items only; no fabricated hidden thinking',
            'fallback_models':[],
        }
        initialize(self,context_policy=POLICY,context_window=context_window,
                   output_reserve=max_output_tokens,history_attribute='messages',
                   token_counter=token_counter or self._count_history_tokens)

    def __enter__(self):return self

    def __exit__(self,*_):self.close()

    def close(self):self.key=''

    @staticmethod
    def _content(text,images):
        content=[{'type':'input_text','text':text}]
        for image in images:
            image=Path(image)
            mime=mimetypes.guess_type(image.name)[0] or 'image/png'
            content.append({'type':'input_image','detail':'high',
                            'image_url':f'data:{mime};base64,'+base64.b64encode(image.read_bytes()).decode()})
        return content

    def _body(self,messages,pending,instructions,schema):
        reasoning={'effort':self.effort}
        if self.reasoning_summary is not None:reasoning['summary']=self.reasoning_summary
        return {'model':self.model,'instructions':instructions,
                'input':[*messages,{'role':'user','content':self._content(pending['text'],pending['images'])}],
                'max_output_tokens':self.max_output_tokens,'store':False,
                'truncation':'disabled','reasoning':reasoning,
                'include':['reasoning.encrypted_content'],
                'text':{'format':{'type':'json_schema','name':'robot_decision','strict':True,'schema':schema}}}

    def _post(self,url,body,*,timeout):
        if timeout<=0:raise TimeoutError('OpenAI active request deadline expired')
        request=urllib.request.Request(url,data=json.dumps(body,separators=(',',':')).encode(),
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        # Do not silently retry paid requests or substitute another model.
        try:
            with urllib.request.urlopen(request,timeout=timeout) as response:return json.load(response)
        except urllib.error.HTTPError as exc:
            with (self.directory/'provider-http-errors.jsonl').open('a') as stream:
                stream.write(json.dumps({'status':exc.code,'endpoint':url,'epoch':time.time(),'retry':False})+'\n')
            raise RuntimeError(f'{self.provider_label} HTTP {exc.code}; no retry or fallback attempted') from None
        except urllib.error.URLError:
            raise RuntimeError(f'{self.provider_label} transport failed; no retry or fallback attempted') from None

    def _count_history_tokens(self,*,messages,pending,overhead):
        body=self._body(messages,pending,overhead['instructions'],overhead['schema'])
        started=time.monotonic()
        timeout=min(30,getattr(self,'_history_deadline',started+30)-started)
        payload=self._post(API_URL+'/input_tokens',body,timeout=timeout)
        count=payload.get('input_tokens')
        if type(count) is not int or count<0:raise RuntimeError('OpenAI token counter returned invalid input_tokens')
        with (self.directory/'token-counts.jsonl').open('a') as stream:
            stream.write(json.dumps({'input_tokens':count,'seconds':time.monotonic()-started,
                                     'completion_generated':False})+'\n')
        return count

    @sliding_ask
    def ask(self,text,*,images=(),schema=None,timeout=180,as_tool_output=False):
        del as_tool_output  # JSON structured output; feedback remains user input.
        if schema is None:
            from pairing.cohort50 import SCHEMA
            schema=SCHEMA
        pending={'text':text,'images':images}
        body=self._body(self.messages,pending,self.instructions,schema)
        started=time.monotonic()
        payload=self._post(API_URL,body,timeout=timeout)
        self.calls+=1
        with (self.directory/'provider-responses.jsonl').open('a') as stream:
            stream.write(json.dumps({'call':self.calls,'seconds':time.monotonic()-started,'response':payload})+'\n')
        if payload.get('model')!=self.model:
            raise RuntimeError('OpenAI returned a different model identifier; pin the actual snapshot explicitly')
        output=payload.get('output')
        if not isinstance(output,list):raise RuntimeError('OpenAI response output is missing')
        # Archive even incomplete/error response items before refusing robot execution.
        self.messages.extend([body['input'][-1],*output])
        reasoning=[item for item in output if item.get('type')=='reasoning']
        self.last_provider_thinking_trace={'call':self.calls,'blocks':reasoning,
            'readable_summary_available':any(item.get('summary') for item in reasoning),
            'opaque_replay_available':any(item.get('encrypted_content') for item in reasoning),
            'policy':'Only returned summaries are readable; encrypted reasoning is opaque'}
        with (self.directory/'provider-thinking.jsonl').open('a') as stream:
            stream.write(json.dumps({'call':self.calls,'blocks':reasoning,
                                     'policy':'verbatim provider output; encrypted_content is opaque'})+'\n')
        usage=payload.get('usage') or {}
        self.usage.append(usage)
        with (self.directory/'model-usage.jsonl').open('a') as stream:
            stream.write(json.dumps({'call':self.calls,'usage':usage,'model':payload.get('model')})+'\n')
        if payload.get('status')!='completed' or payload.get('error'):
            raise RuntimeError('OpenAI response incomplete or failed; no robot operation executed')
        blocks=[block for item in output if item.get('type')=='message'
                for block in item.get('content',[]) if block.get('type')=='output_text']
        if len(blocks)!=1:raise RuntimeError('Expected exactly one structured decision')
        decision=json.loads(blocks[0]['text'])
        if not isinstance(decision,dict) or set(decision)!=set(schema['required']):
            raise RuntimeError('Invalid OpenAI robot decision fields')
        if not all(isinstance(value,str) for value in decision.values()):
            raise RuntimeError('Invalid OpenAI robot decision value types')
        if decision['operation'] not in schema['properties']['operation']['enum']:
            raise RuntimeError('Unsupported OpenAI robot operation')
        return decision

    def usage_summary(self):
        return {'calls':self.calls,'compactions':0,'reported_context_windows':[self.context_window],
                'cumulative_token_usage':{key:sum(x.get(key,0) for x in self.usage)
                                          for key in ('input_tokens','output_tokens','total_tokens')}}

    def save_context(self,path):
        raw=json.dumps(self.messages,separators=(',',':')).encode()
        with gzip.open(path,'wb') as stream:stream.write(raw)
        self.history_store.export(self.directory)
        return {'file':Path(path).name,'sha256':hashlib.sha256(raw).hexdigest(),'messages':len(self.messages)}
