"""Offline DOM-contract tests for readable, text-only trajectory rendering."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

VIEWER=Path(__file__).parents[1]/'roboswe/viewer/review.js'


def render(turn):
    if not shutil.which('node'):pytest.skip('Node required for viewer contract tests')
    source=VIEWER.read_text()
    helpers=source[source.index('function reasoningView('):source.index('function render(){')]
    script='''class Element { constructor(tag){this.tag=tag;this.children=[];this.textContent=''} append(...items){this.children.push(...items)} }
const document={createElement:tag=>new Element(tag)};
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
const pretty=x=>JSON.stringify(x,null,2),pre=x=>el('pre',typeof x==='string'?x:pretty(x));
'''+helpers+'\nconst turn='+json.dumps(turn)+';const before=JSON.stringify(turn);console.log(JSON.stringify({view:renderTurn(turn,0),reasoning:reasoningView(turn.reasoning),unchanged:before===JSON.stringify(turn)}));'
    return json.loads(subprocess.check_output(['node','-e',script],text=True))


def nodes(node):
    yield node
    for child in node.get('children',[]):
        if isinstance(child,dict):yield from nodes(child)


def test_actual_newlines_code_and_stdout_are_not_json_escaped():
    source='import robot\nprint("hello")\n';thought='First observation.\n\nNext hypothesis.'
    value=render({'turn':0,'decision':{'operation':'write_execute','path':'move.py','content':source,'message':'A public rationale.'},'result':{'returncode':0,'output':'hello\nsecond line\n','accepted_control_steps':5},'reasoning':[{'reasoning':thought,'reasoning_details':[{'type':'reasoning.text','text':thought}]}]})
    assert value['unchanged']
    assert value['reasoning']['texts']==[thought]
    rendered=list(nodes(value['view']))
    assert any(n['tag']=='pre' and n['textContent']==source for n in rendered)
    assert any(n['tag']=='pre' and n['textContent']=='hello\nsecond line\n' for n in rendered)
    assert any(n.get('className')=='reasoning-text' and n['textContent']==thought for n in rendered)
    assert value['view']['open'] is True


def test_combined_reasoning_details_deduplicated_display_only():
    value=render({'turn':1,'reasoning':[{'reasoning':'One\n\nTwo','reasoning_details':[{'type':'reasoning.text','text':'One'},{'type':'reasoning.text','text':'Two'},{'type':'reasoning.encrypted','data':'opaque'}]}]})
    assert value['reasoning']=={'texts':['One\n\nTwo'],'opaque':1}
    assert value['unchanged']


def test_opaque_and_empty_thinking_are_not_invented():
    value=render({'turn':2,'reasoning':[{'blocks':[{'block':{'type':'thinking','thinking':'','signature':'opaque-signature'}},{'block':{'type':'redacted_thinking','data':'opaque'}}]}]})
    assert value['reasoning']['texts']==[] and value['reasoning']['opaque']==1
    text=' '.join(n['textContent'] for n in nodes(value['view']))
    assert 'No readable provider reasoning was returned' in text


def test_model_html_is_text_not_markup():
    value=render({'turn':3,'decision':{'message':'<img src=x onerror=alert(1)>'},'reasoning':[{'blocks':[{'block':{'type':'thinking','thinking':'<script>alert(1)</script>'}}]}]})
    assert value['reasoning']['texts']==['<script>alert(1)</script>']
    assert not any(n['tag'] in {'img','script'} for n in nodes(value['view']))
