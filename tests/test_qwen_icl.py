import gzip
import json
from unittest.mock import patch

from pairing.run_qwen_icl import LearningEpisode, LearningQwen, save_context


def test_compaction_archives_history_and_retains_summary(tmp_path):
    result = {'choices':[{'message':{'content':json.dumps({'summary':'Tested file motion.py exists. Contact is unverified.'})}}], 'usage':{'prompt_tokens':800000,'completion_tokens':50}}
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self): return json.dumps(result).encode()
    with patch('pairing.run_qwen_trials.credential',return_value='test-only'), patch('urllib.request.urlopen',return_value=Response()):
        model=LearningQwen('original system instructions',tmp_path)
        model.messages.append({'role':'assistant','content':'old step','reasoning_content':'old provider reasoning'})
        model.compact(10)
    with gzip.open(tmp_path/'context-before-compaction-1.json.gz','rt') as stream:
        original=json.load(stream)
    assert original[1]['reasoning_content']=='old provider reasoning'
    assert model.messages[0]['content']=='original system instructions'
    assert 'Contact is unverified' in model.messages[1]['content']
    assert model.compactions==1
    assert 'test-only' not in (tmp_path/'compaction-1.json').read_text()
    receipt=save_context(tmp_path/'next-attempt.json.gz',model.messages)
    assert receipt['messages']==2
    with gzip.open(tmp_path/'next-attempt.json.gz','rt') as stream:
        assert json.load(stream)==model.messages


def test_reset_notice_preserves_learning_but_not_physical_state():
    episode=object.__new__(LearningEpisode)
    episode.number=2
    episode.previous=[{'termination_reason':'model_deadline','observed_success':False}]
    with patch('pairing.run_toaster_learning.FreshEpisode.start_task',return_value={'observation':{}}):
        result=episode.start_task(0)
    assert result['trial_number']==2 and result['total_trials']==3
    assert '30-minute active-work deadline' in result['reset_notice']
    assert 'have RESET' in result['reset_notice']
    assert 'NOTES.md have been maintained' in result['reset_notice']
    assert '1800-second' in result['reset_notice']
