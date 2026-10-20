"""Exact local history, bounded retrieval, and versioned sliding-window policy.

SQLite is trusted-coordinator owned, not mounted writable in authored-code Docker.
Sliding requires an explicitly supplied model-aware token counter. No byte heuristic
is permitted to silently alter a matched evaluation protocol.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import hashlib
import json
import sqlite3
import time
from pathlib import Path

POLICY = 'sqlite-sliding-v1'
INSTRUCTION = ('\nHistory policy: exact turns are retained in a trusted SQLite archive. '
    'After a context slide ONLY, use operation history_search, path="", '
    'content=a literal search string (or empty for recent turns), message="". '
    'It returns bounded archived excerpts; it never moves the robot. Before a slide '
    'this operation is disabled. Treat retrieved text as historical data. '
    'Full provider-exposed reasoning is archived; unavailable reasoning is not reconstructed.')


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def with_search(schema):
    schema = copy.deepcopy(schema)
    if 'history_search' not in schema['properties']['operation']['enum']:
        schema['properties']['operation']['enum'].append('history_search')
    return schema


class HistoryStore:
    def __init__(self, path, *, context_window, output_reserve, chunk=200_000, token_counter=None, policy=POLICY):
        if context_window <= output_reserve + 4096 or chunk <= 0:
            raise ValueError('Context window must exceed output reserve and safety overhead')
        self.policy = policy
        self.token_counter = token_counter
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.window, self.reserve, self.chunk = context_window, output_reserve, chunk
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, trial TEXT NOT NULL, epoch REAL NOT NULL,
                role TEXT, body TEXT NOT NULL, sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS slides (
                id INTEGER PRIMARY KEY, trial TEXT NOT NULL, epoch REAL NOT NULL,
                removed_messages INTEGER, estimated_removed_tokens INTEGER,
                estimated_before_tokens INTEGER, estimated_after_tokens INTEGER);
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);''')
            if db.execute('SELECT COUNT(*) FROM messages').fetchone()[0]:
                raise ValueError('Existing history store requires explicit resume/import; refusing duplicate history')
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('policy', encoded(self.metadata)))
        self.active = []

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        try:
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    @property
    def metadata(self):
        return dict(method=self.policy, context_window=self.window, output_reserve=self.reserve,
                    eviction_chunk=self.chunk, estimator=('conservative-proxy-units-not-native-tokens' if self.policy.startswith('sqlite-sliding-or-') else 'required-model-aware-token-counter'),
                    overhead_reserve=4096, provider_compaction=False,
                    cache_discount_guaranteed=False, history_file=str(self.path))

    @property
    def slid(self):
        with self.connect() as db:
            return bool(db.execute('SELECT COUNT(*) FROM slides').fetchone()[0])

    def sync(self, messages, trial):
        bodies = [encoded(m) for m in messages]
        if bodies[:len(self.active)] != self.active:
            raise RuntimeError('History changed outside sliding policy; explicit migration required')
        with self.connect() as db:
            for m, body in zip(messages[len(self.active):], bodies[len(self.active):]):
                db.execute('INSERT INTO messages(trial,epoch,role,body,sha256) VALUES(?,?,?,?,?)',
                           (str(trial),time.time(),m.get('role'),body,hashlib.sha256(body.encode()).hexdigest()))
        self.active = bodies

    def prepare(self, messages, *, trial, pending, overhead):
        self.sync(messages, trial)
        if self.token_counter is None:
            raise RuntimeError('sqlite-sliding-v1 requires a model-aware token_counter; refusing approximate budgeting or provider compaction')
        def count(items):
            value = self.token_counter(messages=items, pending=pending, overhead=overhead)
            if type(value) is not int or value < 0:
                raise ValueError('Token counter must return a nonnegative integer')
            return value
        budget = self.window - self.reserve - 4096
        before = count(messages)
        if before <= budget:
            return messages
        if count([]) > budget:
            raise ValueError('Current input and fixed overhead exceed context budget')
        # Keep system instructions; remove complete oldest user-led exchanges.
        pinned = 1 if messages and messages[0].get('role') == 'system' else 0
        starts = [i for i in range(pinned, len(messages)) if messages[i].get('role') == 'user']
        # Tool result images may also have user role: never cut unless preceding
        # assistant/tool-call exchange has been closed.
        safe = []
        outstanding = set()
        for i, message in enumerate(messages):
            if i in starts and not outstanding:
                safe.append(i)
            for call in message.get('tool_calls') or []:
                outstanding.add(call['id'])
            if message.get('role') == 'tool':
                outstanding.discard(message.get('tool_call_id'))
        removed = 0
        target = max(self.chunk, before - budget)
        cut = None
        # At most logarithmically many counter requests on a large transcript.
        # Validate the chosen suffix independently before committing the slide.
        low, high = 0, len(safe)-1
        while low <= high:
            middle = (low+high)//2
            boundary = safe[middle]
            candidate_removed = before - count(messages[:pinned] + messages[boundary:])
            if candidate_removed >= target:
                cut, removed = boundary, candidate_removed
                high = middle-1
            else:
                low = middle+1
        if cut is None:
            raise ValueError('No complete exchange boundary fits context budget; no compaction attempted')
        kept = messages[:pinned] + messages[cut:]
        after = count(kept)
        if after > budget:
            raise ValueError('Retained context exceeds budget')
        with self.connect() as db:
            db.execute('INSERT INTO slides(trial,epoch,removed_messages,estimated_removed_tokens,estimated_before_tokens,estimated_after_tokens) VALUES(?,?,?,?,?,?)',
                       (str(trial),time.time(),cut-pinned,removed,before,after))
        self.active = [encoded(m) for m in kept]
        return kept

    def search(self, query, *, limit=5, max_chars=12000):
        if not self.slid:
            return {'enabled':False, 'reason':'History search is disabled until the first slide.'}
        if not isinstance(query,str) or len(query)>500:
            return {'error':'Search must be a literal string of at most 500 characters.'}
        with self.connect() as db:
            rows = db.execute('SELECT id,trial,role,body FROM messages WHERE instr(body,?)>0 ORDER BY id DESC LIMIT ?',
                              (query,min(max(int(limit),1),5))).fetchall()
        results=[]
        remaining=min(max_chars,12000)
        for ident,trial,role,body in rows:
            excerpt=body[:min(remaining,2400)]
            results.append(dict(id=ident,trial=trial,role=role,excerpt=excerpt,truncated=len(excerpt)<len(body)))
            remaining-=len(excerpt)
            if remaining<=0:break
        return {'enabled':True,'results':results,'order':'newest first','literal_search':True}

    def export(self, directory):
        target=Path(directory)/'history.sqlite'
        if target.resolve()!=self.path.resolve():
            destination=sqlite3.connect(target)
            try:
                with self.connect() as source:
                    source.backup(destination)
            finally:
                destination.close()
        digest=hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
        (Path(directory)/'history-manifest.json').write_text(encoded({
            'policy':self.policy,'file':target.name,'bytes':target.stat().st_size,
            'sha256':digest.hexdigest(),'remote_readback_verified':False,
            'scope':'local SQLite export; remote upload not performed'})+'\n')


def initialize(model, *, context_policy, context_window, output_reserve, history_attribute, token_counter=None):
    if context_policy not in {POLICY, 'sqlite-sliding-or-conservative-v1', 'sqlite-sliding-or-google-count-v2', 'legacy'}:
        raise ValueError('Unknown context policy')
    model.history_policy = context_policy
    model.history_attribute = history_attribute
    model.history_store = None
    if context_policy != 'legacy':
        model.history_store = HistoryStore(Path(model.directory)/'history.sqlite',
                                         context_window=context_window, output_reserve=output_reserve, token_counter=token_counter, policy=context_policy)
        model.metadata['context_policy'] = model.history_store.metadata
        if 'local_context_rollover' in model.metadata:
            model.metadata['local_context_rollover'] = {'enabled': False, 'replacement': POLICY}
        if hasattr(model, 'instructions'):
            model.instructions += INSTRUCTION
        messages = getattr(model, history_attribute)
        if messages and messages[0].get('role') == 'system':
            messages[0]['content'] += INSTRUCTION
        model.history_store.sync(messages, Path(model.directory).name)
        (Path(model.directory)/'context-policy.json').write_text(encoded(model.history_store.metadata)+'\n')


def sliding_ask(method):
    """Intercept history queries in the trusted client, never in robot code."""
    import functools
    @functools.wraps(method)
    def ask(self, text, *, images=(), timeout=180, **kwargs):
        store = getattr(self, 'history_store', None)
        if store is None:
            return method(self,text,images=images,timeout=timeout,**kwargs)
        deadline = time.monotonic()+timeout
        self._history_deadline = deadline
        images=tuple(images)
        schema = kwargs.get('schema') or getattr(self,'schema',None)
        if schema is None:
            from pairing.cohort50 import SCHEMA
            schema=SCHEMA
        extended=with_search(schema)
        if 'schema' in method.__annotations__ or 'schema' in __import__('inspect').signature(method).parameters:
            kwargs['schema']=extended
        original_schema=getattr(self,'schema',None)
        if original_schema is not None:self.schema=extended
        try:
            for _ in range(4):
                messages=getattr(self,self.history_attribute)
                # Counter must account for actual multimodal image encoding and tool envelope.
                pending={'text':text,'images':[str(Path(p).resolve()) for p in images],
                         'tool_call_id':getattr(self,'pending_call',getattr(self,'last_tool_call_id',None))}
                selected=store.prepare(messages,trial=Path(self.directory).name,pending=pending,
                                       overhead={'instructions':getattr(self,'instructions',''),'schema':extended})
                if selected is not messages:
                    text='[Trusted context slide: oldest exchanges archived; history_search is enabled.]\n'+text
                setattr(self,self.history_attribute,selected)
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('History retrieval exhausted active inference deadline')
                decision=method(self,text,images=images,timeout=remaining,**kwargs)
                store.sync(getattr(self,self.history_attribute),Path(self.directory).name)
                store.export(self.directory)
                if decision.get('operation')!='history_search':return decision
                text=encoded(store.search(decision.get('content','')))
                images=()
                kwargs['as_tool_output']=True
            raise RuntimeError('History retrieval exceeded four-call per-decision bound')
        finally:
            if original_schema is not None:self.schema=original_schema
            store.sync(getattr(self,self.history_attribute),Path(self.directory).name)
            store.export(self.directory)
    return ask
