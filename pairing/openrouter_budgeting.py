"""Explicit approximate OR context admission; never advertised as exact tokens.

Versioned experimental policy: UTF-8 serialized text bytes + 16,384 units per
image + 8,192 fixed units, calibrated upward only from actual prompt usage.
These are conservative reserves, not a proof of the native tokenizer bound.
"""
import json
import math
from pathlib import Path

POLICY='sqlite-sliding-or-conservative-v1'


class ConservativeCounter:
    def __init__(self, directory):
        self.directory=Path(directory)
        self.multiplier=1.0
        self.last_base=None

    def base(self, messages, schema):
        images=0
        def scrub(value):
            nonlocal images
            if isinstance(value,dict):
                if value.get('type')=='image_url':
                    images+=1
                    return {'type':'image_url','reserved_image':True}
                return {k:scrub(v) for k,v in value.items()}
            if isinstance(value,list):return [scrub(v) for v in value]
            return value
        text=json.dumps({'messages':scrub(messages),'response_schema':schema},ensure_ascii=False,separators=(',',':')).encode()
        return len(text)+images*16384+8192

    def count_body(self, body):
        base=self.base(body['messages'],body['response_format'])
        self.last_base=base
        return math.ceil(base*self.multiplier)

    def __call__(self, *, messages, pending, overhead):
        from pairing.openrouter_model import OpenRouterLearningModel
        all_messages=[{'role':'system','content':overhead['instructions']},*messages,
                      {'role':'user','content':OpenRouterLearningModel._content(pending['text'],pending['images'])}]
        schema={'type':'json_schema','json_schema':{'name':'robot_decision','strict':True,'schema':overhead['schema']}}
        return math.ceil(self.base(all_messages,schema)*self.multiplier)

    def observe(self, actual, *, estimated, call):
        if type(actual) is not int or actual<0:
            raise RuntimeError('OpenRouter omitted prompt usage; cannot calibrate conservative context admission')
        before=self.multiplier
        if self.last_base:
            self.multiplier=max(self.multiplier,actual*1.25/self.last_base)
        self.directory.mkdir(parents=True,exist_ok=True)
        with (self.directory/'context-estimates.jsonl').open('a') as stream:
            stream.write(json.dumps({'policy':POLICY,'call':call,'estimated_input_units':estimated,
                'actual_prompt_tokens':actual,'multiplier_before':before,'multiplier_after':self.multiplier,
                'underestimated':actual>estimated,'exact_native_count':False})+'\n')

    @property
    def metadata(self):
        return {'method':POLICY,'exact_native_count':False,'text_units':'serialized UTF-8 bytes',
                'image_reserve_units':16384,'fixed_reserve_units':8192,
                'usage_calibration':'upward only, actual prompt count times1.25',
                'native_upper_bound_proven':False,'overflow_policy':'stop and retain evidence; no silent retry'}
