"""Exact OpenRouter Muse contributor client; validated final operations only."""
import base64
import copy
import gzip
import hashlib
import json
import math
import os
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path
from pairing.context_history import POLICY, initialize, sliding_ask
from pairing.cohort50 import SCHEMA
from pairing.run_qwen_trials import parse_decision, ModelProtocolError

MODEL='meta/muse-spark-1.3-contributor'
BASE='https://openrouter.ai/api/v1'

def credential():
    import os
    value = os.environ.get("OPENROUTER_API_KEY")
    if not value:
        raise RuntimeError("Set OPENROUTER_API_KEY in the trusted coordinator environment")
    return value

class MuseModel:
    def __init__(self,prompt,directory,*,decision_transport="native-tool-results-v2",schema=None,context_policy="legacy",context_window=1_000_000,token_counter=None):
        self.schema=copy.deepcopy(SCHEMA if schema is None else schema)
        if decision_transport not in ("native-tool-results-v2","single-function-auto-validated-v1"):raise ValueError("Unsupported decision transport")
        self.legacy_history=decision_transport=="single-function-auto-validated-v1"
        self.key=credential();self.directory=Path(directory);self.messages=[{'role':'system','content':prompt}];self.calls=0;self.compactions=0;self.pending_call=None
        self.metadata={'model':MODEL,'base_url':BASE,'reasoning_effort':'high','max_completion_tokens':32768,'decision_transport':decision_transport,'context_policy':'Messages and provider-returned reasoning retained, model-generated compaction above 8 MB request history; prior history archived; no cross-task transfer','credential_location':'trusted coordinator only'}
        initialize(self,context_policy=context_policy,context_window=context_window,output_reserve=32768,history_attribute="messages",token_counter=token_counter)
    @sliding_ask
    def ask(self,text,*,images=(),timeout=180,as_tool_output=False):
        began=time.monotonic()
        raw=json.dumps(self.messages).encode()
        if self.history_store is None and len(raw)>8_000_000:
            self.compactions+=1
            checkpoint=self.directory/f'context-before-compaction-{self.compactions}.json.gz'
            with gzip.open(checkpoint,'wb') as out:out.write(raw)
            decision=self._ask('Create a working-context checkpoint for your own continuation. Preserve task and reset history, verified measurements, tested script sequence, file names, failed assumptions, remaining unsolved work and current physical state. Distinguish observations from hypotheses. Do not take a robot action. Call robot_decision with operation finish, empty path and content, and the complete checkpoint in message. This response is used only as a context summary; it does not end the episode.',timeout=timeout)
            summary=decision['message']
            if not summary.strip():raise ModelProtocolError('Empty model-generated context checkpoint')
            self.messages=[self.messages[0],{'role':'user','content':'Continue your same task using this saved context. Your Python library and NOTES.md are still present. Any subsequent reset notice supersedes the physical-state description.\n'+summary}]
            (self.directory/f'compaction-{self.compactions}.json').write_text(json.dumps({'checkpoint':checkpoint.name,'sha256':hashlib.sha256(raw).hexdigest(),'summary':summary},indent=2))
            self.pending_call=None
            timeout-=time.monotonic()-began
            if timeout<=0:raise TimeoutError('No active budget remains after context checkpoint')
            began=time.monotonic()
        try:return self._ask(text,images=images,timeout=timeout)
        except ModelProtocolError as exc:
            with (self.directory/'protocol-recovery.jsonl').open('a') as out:out.write(json.dumps({'error':str(exc),'call':self.calls,'executed':False})+'\n')
            remaining=timeout-(time.monotonic()-began)
            if remaining<=0:raise TimeoutError('No time remains for structured-output correction') from None
            return self._ask('Your previous final content violated the JSON schema. No operation executed and physics did not advance. Call robot_decision exactly once with valid arguments: string fields operation, path, content, message. Prior images and state remain current.',timeout=remaining)

    def _ask(self,text,*,images=(),timeout=180):
        content=[{'type':'text','text':text}]
        for path in images:
            path=Path(path);mime='image/jpeg' if path.suffix=='.jpg' else 'image/png'
            content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode()}})
        pending=[{'role':'user','content':content}]
        if self.pending_call:
            pending=[{'role':'tool','tool_call_id':self.pending_call,'content':text}]
            if images:pending.append({'role':'user','content':[{'type':'text','text':'Current sensor images accompanying the tool result.'}]+content[1:]})
        payload={'model':MODEL,'messages':self.messages+pending,'reasoning':{'effort':'high'},'max_tokens':32768,'tools':[{'type':'function','function':{'name':'robot_decision','description':'Submit exactly one bounded robot-code operation to the harness.','strict':True,'parameters':self.schema}}],'tool_choice':'auto','provider':{'require_parameters':True,'allow_fallbacks':False}}
        request=urllib.request.Request(BASE+'/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        began=time.monotonic()
        for retry in range(4):
            remaining=timeout-(time.monotonic()-began)
            if remaining<=0:raise TimeoutError('Muse inference reached remaining deadline')
            try:
                with urllib.request.urlopen(request,timeout=max(.1,remaining)) as response:result=json.load(response)
                break
            except urllib.error.HTTPError as exc:
                detail=exc.read(8192).decode(errors='replace').replace(self.key,'[REDACTED]')
                with (self.directory/'provider-http-errors.jsonl').open('a') as out:out.write(json.dumps({'epoch':time.time(),'status':exc.code,'retry':retry,'detail':detail})+'\n')
                transient=exc.code in (429,500,502,503,504) or (exc.code==404 and 'model_not_found' in detail)
                if exc.code==402:
                    try:
                        transient=json.loads(detail).get('error',{}).get('metadata',{}).get('reason')=='in_flight_budget_exhausted'
                    except (ValueError,AttributeError):pass
                delay=2**(retry+1)
                try:
                    requested=exc.headers.get('Retry-After') if exc.headers else None
                    if requested is None:requested=json.loads(detail).get('error',{}).get('metadata',{}).get('retry_after_seconds')
                    requested=float(requested)
                    if math.isfinite(requested) and requested>=0:delay=max(delay,requested)
                except (ValueError,TypeError,AttributeError):pass
                if transient and retry<3 and timeout-(time.monotonic()-began)>delay:
                    while delay>0:
                        interval=min(60,delay);time.sleep(interval);delay-=interval
                    continue
                raise RuntimeError(f'OpenRouter Muse HTTP {exc.code}; credentials suppressed; diagnostic saved') from None
            except (TimeoutError,urllib.error.URLError) as exc:
                if isinstance(exc,TimeoutError) or isinstance(getattr(exc,'reason',None),TimeoutError):raise TimeoutError('Muse inference reached remaining deadline') from None
                raise RuntimeError('Muse inference transport failed') from None
        self.calls+=1
        with (self.directory/'provider-responses.jsonl').open('a') as out:out.write(json.dumps({'call':self.calls,'seconds':time.monotonic()-began,'request_bytes':len(request.data),'response':result})+'\n')
        if result.get('error'):raise RuntimeError('OpenRouter returned provider error; see redacted provider receipt')
        message=result['choices'][0]['message']
        calls=message.get('tool_calls') or []
        decision_message=message
        self.pending_call=None
        if calls:
            if len(calls)!=1 or calls[0].get('function',{}).get('name')!='robot_decision':raise ModelProtocolError('Expected one robot_decision function call')
            if not calls[0].get('id'):raise ModelProtocolError('Native tool call is missing its ID')
            self.pending_call=calls[0]['id']
            decision_message={**message,'content':calls[0]['function'].get('arguments')}
        history={k:message[k] for k in ('role','content','tool_calls','reasoning_details','reasoning') if k in message}
        if self.legacy_history:
            history={k:decision_message[k] for k in ('role','content','reasoning_details','reasoning') if k in decision_message}
            self.pending_call=None
        self.messages.extend(pending+[history])
        with (self.directory/'model-usage.jsonl').open('a') as out:out.write(json.dumps({'call':self.calls,'usage':result.get('usage'),'model':result.get('model'),'provider':result.get('provider')})+'\n')
        return parse_decision(decision_message,self.schema)

if __name__=='__main__':
    from PIL import Image
    folder=Path('artifacts/muse-learning-20261006/provider-smoke');folder.mkdir(parents=True,exist_ok=True)
    im=Image.new('RGB',(96,96),'red');im.save(folder/'red.png')
    model=MuseModel('Call robot_decision exactly once with the requested operation. This is an image-input and tool smoke test.',folder)
    result=model.ask('Describe the image color in message. Use operation finish and empty path/content.',images=[folder/'red.png'],timeout=120)
    (folder/'receipt.json').write_text(json.dumps({'metadata':model.metadata,'decision':result},indent=2));print(json.dumps(result))
