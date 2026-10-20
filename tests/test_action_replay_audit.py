import hashlib
import json
from pathlib import Path

import pytest

from experiments.action_replay_audit import audit, inspect_source, UnsupportedActionAudit, REQUIRED_CAPABILITIES
from roboswe.library.attempts import Journal


def put(path, data):
    path.write_text(json.dumps(data))


def fixture(tmp_path, actions=(0.1, 0.2), wait=False):
    source=tmp_path/'original';source.mkdir()
    spec={'mode':'serial','tasks':[{'task':'Example'}],'settle_steps':2,'max_control_steps_per_task':10,'sha256':'spec','docker_simulator_image':'sha256:'+'b'*64}
    initial={'config':{'seed':0},'initial_fingerprint':'initial','spec_sha256':'spec','spec':spec,'bindings':{},'code_image':'sha256:'+'a'*64}
    rows=[];digest=hashlib.sha256()
    for n,value in enumerate(actions,1):
        vector=[value];digest.update(json.dumps(vector,separators=(',',':')).encode()+b'\n')
        rows.append({'control_step':n,'action':vector,'upstream_success':True,'sustained_steps':n})
    report={'control_steps':len(rows),'observed_success':len(rows)>=2,'precondition_valid':True,'start_fingerprint':'initial','end_fingerprint':'final','cumulative_action_sha256':digest.hexdigest(),'passive_wait_control_steps':1 if wait else 0,'verdict':{'success':len(rows)>=2,'sustained_steps':len(rows),'final_predicate':True}}
    put(source/'job.json',{'finished_at':'now','status':'verification_invalid'})
    put(source/'verification.json',{'reproduced':False,'reason':'stochastic authored code diverged'})
    put(source/'report.json',report)
    (source/'actions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    journal=Journal(source/'submissions.jsonl');journal.append('initial',initial);journal.append('start_task_request',{'index':0})
    if wait:
        journal.append('operation_request',{'decision':{'operation':'wait'},'policy':'passive_wait_v1'})
        journal.append('operation_result',{'operation':'wait','result':{'passive_wait':{'requested_steps':1,'advanced_steps':1}}})
    journal.append('finish_task',report);journal.close()
    return source,initial,report


class Runtime:
    action_audit_capabilities = REQUIRED_CAPABILITIES.copy()
    def __init__(self,initial,report,predicate=True):
        self.initial=initial;self.report=report;self.steps=0;self.closed=False;self.actual_predicate=predicate;self.actions=[]
    def metadata(self):return self.initial
    def observe(self):return {'control_step':self.steps,'sensors':{'position':sum(a[0] for a in self.actions)},'action_schema':{'format':'native_vector','shape':[1],'low':[-1],'high':[1]}}
    def start_task(self,index):assert index==0;return {'precondition_valid':True,'observation':self.observe()}
    def cached_predicate(self):return self.actual_predicate if self.steps else False
    def bridge(self,message):
        assert message['operation']=='step';self.steps+=1;self.actions.append(message['action']);return self.observe()
    def finish_task(self,code_errors):
        result=dict(self.report);result['observed_success']=self.actual_predicate and self.steps>=2
        result['verdict']={'success':result['observed_success'],'sustained_steps':self.steps if self.actual_predicate else 0,'final_predicate':self.actual_predicate}
        return result
    def close(self):self.closed=True


def test_stochastic_actor_is_not_rerun_and_official_grade_is_untouched(tmp_path):
    samples=iter([0.1,0.8]);actor_calls=[]
    def stochastic_actor():actor_calls.append(1);return next(samples)
    original=stochastic_actor();source,initial,report=fixture(tmp_path,(original,original))
    before=(source/'verification.json').read_bytes();runtime=Runtime(initial,report)
    result=audit(source,lambda meta,output:runtime,tmp_path/'audit')
    assert result['status']=='consistent' and result['recomputed_success'] is True
    assert actor_calls==[1] and runtime.actions==[[0.1],[0.1]]
    assert (source/'verification.json').read_bytes()==before
    assert result['changes_official_score'] is False and result['actor_code_executed'] is False and runtime.closed


def test_predicates_recomputed_never_copied_from_log(tmp_path):
    source,initial,report=fixture(tmp_path);runtime=Runtime(initial,report,predicate=False)
    result=audit(source,lambda *a:runtime,tmp_path/'audit')
    assert result['status']=='diverged' and result['recomputed_success'] is False
    assert any(x['reason']=='predicate_trajectory_diverged' for x in result['mismatches'])


@pytest.mark.parametrize('bad',['gap','nan','budget','digest','unfinalized','unknown_wait'])
def test_invalid_source_rejected_before_factory(tmp_path,bad):
    source,initial,report=fixture(tmp_path)
    if bad=='unfinalized':put(source/'job.json',{'status':'running'})
    elif bad in {'budget','digest'}:
        report['control_steps' if bad=='budget' else 'cumulative_action_sha256']=999 if bad=='budget' else 'wrong';put(source/'report.json',report)
    elif bad=='unknown_wait':report['passive_wait_control_steps']=1;put(source/'report.json',report)
    else:
        rows=[json.loads(l) for l in (source/'actions.jsonl').read_text().splitlines()]
        if bad=='gap':rows[1]['control_step']=3
        else:rows[0]['action']=[float('nan')]
        (source/'actions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(UnsupportedActionAudit):audit(source,lambda *a:pytest.fail('Invalid source must not start runtime'),tmp_path/'audit')


def test_bounded_vectors_checked_before_motion(tmp_path):
    source,initial,report=fixture(tmp_path,(2.0,2.0));runtime=Runtime(initial,report)
    result=audit(source,lambda *a:runtime,tmp_path/'audit')
    assert result['status']=='unsupported_or_incomplete' and runtime.steps==0 and runtime.closed


def test_wait_ticks_replayed_once_no_extra_settle_steps(tmp_path):
    source,initial,report=fixture(tmp_path,wait=True);runtime=Runtime(initial,report)
    result=audit(source,lambda *a:runtime,tmp_path/'audit')
    assert result['status']=='consistent' and runtime.steps==2 and result['source_passive_wait_ticks']==1


def test_observation_divergence_is_explicit(tmp_path):
    source,initial,report=fixture(tmp_path)
    (source/'model-transcript.jsonl').write_text(json.dumps({'event':'request','prompt':json.dumps({'observation':{'control_step':1,'sensors':{'position':999}}})})+'\n')
    result=audit(source,lambda *a:Runtime(initial,report),tmp_path/'audit')
    assert result['status']=='diverged' and any(x['reason']=='observation_checkpoint_diverged' for x in result['mismatches'])


def test_wrong_image_and_missing_predicate_fail_closed(tmp_path):
    source,initial,report=fixture(tmp_path);runtime=Runtime({**initial,'config':{'seed':1}},report)
    result=audit(source,lambda *a:runtime,tmp_path/'audit')
    assert result['status']=='unsupported_or_incomplete' and runtime.steps==0
    other=Runtime(initial,report);other.cached_predicate=None
    result=audit(source,lambda *a:other,tmp_path/'other-audit')
    assert result['status']=='unsupported_or_incomplete' and other.closed


def test_audit_output_never_overwrites_original(tmp_path):
    source,_,_=fixture(tmp_path)
    with pytest.raises(ValueError):audit(source,None,source/'audit')


def test_uncertified_mutating_predicate_rejected_before_inspection_or_actions(tmp_path):
    source,initial,report=fixture(tmp_path)
    class Mutating(Runtime):
        action_audit_capabilities = None
        def metadata(self):pytest.fail('Cannot inspect uncertified runtime')
        def cached_predicate(self):self.steps+=1;return True
        def predicate(self):pytest.fail('Never call upstream predicate accessor')
    runtime=Mutating(initial,report)
    result=audit(source,lambda *a:runtime,tmp_path/'audit')
    assert result['status']=='unsupported_or_incomplete' and runtime.steps==0 and runtime.closed
    assert result['runtime_replay_performed'] is False


@pytest.mark.parametrize('capability',list(REQUIRED_CAPABILITIES))
def test_each_nonmutating_contract_requirement_is_enforced(tmp_path,capability):
    source,initial,report=fixture(tmp_path);runtime=Runtime(initial,report)
    runtime.action_audit_capabilities={**REQUIRED_CAPABILITIES,capability:False}
    result=audit(source,lambda *a:runtime,tmp_path/'audit')
    assert result['status']=='unsupported_or_incomplete' and runtime.steps==0


def test_certified_runtime_uses_cache_never_additional_upstream_check(tmp_path):
    source,initial,report=fixture(tmp_path)
    class CachedRuntime(Runtime):
        upstream_checks=0
        def predicate(self):pytest.fail('Extra upstream predicate query changes state')
        def bridge(self,message):
            # Preserve the actual native reward + environment success cadence:
            # two upstream evaluations per tick, no additional audit query.
            self.upstream_checks+=2
            return super().bridge(message)
    runtime=CachedRuntime(initial,report)
    result=audit(source,lambda *a:runtime,tmp_path/'audit')
    assert result['status']=='consistent'
    assert len(runtime.actions)==2 and runtime.upstream_checks==4
    assert result['runtime_capability_certification']['observation_calls_do_not_evaluate_predicate'] is True
