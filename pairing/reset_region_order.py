"""Pin Counter region ordering without changing geometry or consuming RNG draws.

Counter.get_reset_regions currently deduplicates XML Elements via set(), whose
identity hashes vary between interpreters. References are trusted preflight data
and never participant observations. Upstream files remain untouched.
"""
import functools
import inspect
import json
import math


def plain(value):
    def convert(item):
        if hasattr(item,'tolist'):return item.tolist()
        if hasattr(item,'name'):return item.name
        raise TypeError(f'Unsupported region metadata type: {type(item).__name__}')
    return json.loads(json.dumps(value,default=convert,allow_nan=False))


def close_geometry(left, right, atol):
    if isinstance(left,dict) and isinstance(right,dict):
        return left.keys()==right.keys() and all(close_geometry(left[k],right[k],atol) for k in left)
    if isinstance(left,list) and isinstance(right,list):
        return len(left)==len(right) and all(close_geometry(a,b,atol) for a,b in zip(left,right))
    if type(left) in (int,float) and type(right) in (int,float):
        return math.isfinite(left) and math.isfinite(right) and abs(left-right)<=atol
    return type(left)==type(right) and left==right


def reorder_regions(actual, expected, *, atol=0.):
    """Match identical geometry, restoring only names and enumeration order."""
    if not math.isfinite(atol) or not 0<=atol<=1e-12:raise ValueError('Region matching tolerance must be between 0 and 1e-12')
    remaining=list(actual.items());result={}
    if len(remaining)!=len(expected):raise ValueError('Counter region inventory changed')
    for name,value in expected:
        match=next((i for i,(_,candidate) in enumerate(remaining) if plain(candidate)==value),None)
        if match is None and atol:
            candidates=[i for i,(_,candidate) in enumerate(remaining) if close_geometry(plain(candidate),value,atol)]
            if len(candidates)>1:raise ValueError('Ambiguous counter region tolerance match')
            match=candidates[0] if candidates else None
        if match is None:raise ValueError('Counter region geometry differs from reference')
        _,native=remaining.pop(match);result[name]=native
    return result


def anchors_from_metadata(metadata):
    groups={}
    for cfg in metadata.get('object_cfgs',[]):
        region=cfg.get('reset_region',{});name=region.get('name','')
        if not name.startswith('geom_'):continue
        placement=cfg.get('placement',{});kwargs=placement.get('sample_region_kwargs',{})
        context={'fixture':placement.get('fixture'),'ref':kwargs.get('ref'),'loc':kwargs.get('loc','nn')}
        for k in ('top_size','ref_rot_flag','full_depth_region'):
            if k in kwargs:context[k]=kwargs[k]
        key=json.dumps(context,sort_keys=True)
        group=groups.setdefault(key,{'context':context,'regions':{}})
        value={k:v for k,v in region.items() if k!='name'}
        if name in group['regions'] and group['regions'][name]!=value:
            raise ValueError('Conflicting region anchors in frozen episode metadata')
        group['regions'][name]=value
    return list(groups.values())


def anchored_regions(actual,query,anchors):
    matched=[a for a in anchors if all(query.get(k)==v for k,v in a['context'].items())]
    if not matched:return actual
    known={}
    for anchor in matched:
        for name,value in anchor['regions'].items():
            if name in known and known[name]!=value:raise ValueError('Ambiguous frozen region anchor')
            known[name]=value
    if not set(known).issubset(actual):raise ValueError('Frozen region names are missing')
    remaining=list(actual.items());resolved={}
    for name,value in known.items():
        match=next((i for i,(_,candidate) in enumerate(remaining) if plain(candidate)==value),None)
        if match is None:raise ValueError('Frozen region anchor geometry missing')
        _,candidate=remaining.pop(match);resolved[name]=candidate
    unused=[name for name in actual if name not in resolved]
    if len(unused)>1:raise ValueError('Insufficient anchors to reconstruct the complete region order')
    if unused:resolved[unused[0]]=remaining[0][1]
    # Counter assigns geom_N sequentially; restore that canonical name order.
    return {name:resolved[name] for name in sorted(resolved,key=lambda name:int(name.rsplit('_',1)[1]))}


class CounterRegionOrder:
    def __init__(self,reference=None,anchors=(),atol=0.):
        self.reference=reference;self.anchors=anchors;self.atol=atol;self.calls=[];self.original=None
        if reference is not None and reference.get('schema_version')!=1:raise ValueError('Unsupported region-order reference')
    def __enter__(self):return self
    def install(self,counter_class=None):
        if self.original is not None:raise RuntimeError('Counter ordering guard already installed')
        if counter_class is None:
            from robocasa.models.fixtures.counter import Counter
            counter_class=Counter
        self.counter_class=counter_class;self.original=counter_class.get_reset_regions
        signature=inspect.signature(self.original);guard=self;original=self.original
        @functools.wraps(original)
        def wrapper(counter,*args,**kwargs):
            bound=signature.bind(counter,*args,**kwargs);bound.apply_defaults()
            query={'fixture':counter.name,**plain({k:v for k,v in bound.arguments.items() if k not in ('self','env')})}
            actual=original(counter,*args,**kwargs)
            if guard.reference is not None:
                index=len(guard.calls);expected=guard.reference['calls']
                if index>=len(expected) or expected[index]['query']!=query:
                    raise ValueError('Counter region query sequence differs from frozen reset')
                actual=reorder_regions(actual,expected[index]['regions'],atol=guard.atol)
            elif guard.anchors:
                actual=anchored_regions(actual,query,guard.anchors)
            guard.calls.append({'query':query,'regions':[[k,plain(v)] for k,v in actual.items()]})
            return actual
        counter_class.get_reset_regions=wrapper
    def __exit__(self,kind,value,tb):
        if self.original is not None:self.counter_class.get_reset_regions=self.original
        if kind is None and self.reference is not None and len(self.calls)!=len(self.reference['calls']):
            raise ValueError('Frozen reset region sequence was not fully consumed')
    def payload(self):return {'schema_version':1,'calls':self.calls}


def guarded_factory(guard):
    def factory(*args,**kwargs):
        # Match normal import order AFTER RobotSession seeds both Python and NumPy.
        from robot_session.environment import RobotEnvironment
        from robocasa.utils.env_utils import create_env  # noqa: F401
        guard.install()
        return RobotEnvironment(*args,**kwargs)
    return factory
