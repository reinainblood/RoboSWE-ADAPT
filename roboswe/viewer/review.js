/* Render all evidence as text: model and tool content is untrusted. */
const $=id=>document.getElementById(id);let records=[],selected=null,data=null,staticMode=false,tab='video';
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
const pretty=x=>JSON.stringify(x,null,2), pre=x=>el('pre',typeof x==='string'?x:pretty(x));
async function json(url){const r=await fetch(url);if(!r.ok)throw Error(`${r.status}: ${await r.text()}`);return r.json()}
function notice(s,error=false){$('status').textContent=s;$('status').className=error?'error':'muted'}
async function load(){try{if(window.ROBOSWE_SNAPSHOT){staticMode=true;records=window.ROBOSWE_SNAPSHOT.records;renderList();notice('Shared snapshot · paid judge disabled');return}if(!records.length){try{records=await json('./api/review-jobs')}catch{staticMode=true;records=await json('./review-jobs.json')}}else records=await json(staticMode?'./review-jobs.json':'./api/review-jobs');renderList();notice(staticMode?'Shared snapshot · paid judge disabled':'Local evidence · refreshes every 15 seconds');}catch(e){notice('Could not load jobs: '+e.message,true)}}
function renderList(){const q=$('search').value.toLowerCase(),filter=$('filter').value;const list=records.filter(j=>(j.name+' '+j.id).toLowerCase().includes(q)&&(!filter||(filter==='failed'?/fail|error|interrupt/.test(j.status):filter==='running'?/running|starting/.test(j.status):/completed|success|passed/.test(j.status))));$('count').textContent=`${list.length} records`;$('jobs').replaceChildren();for(const j of list){const b=el('button',undefined,'job');b.append(el('strong',j.name),el('small',`${j.status} · ${j.id}`));b.setAttribute('aria-current',j.id===selected);b.onclick=()=>select(j.id);$('jobs').append(b)}}
async function select(id){selected=id;location.hash=id;try{data=window.ROBOSWE_SNAPSHOT?.episodes[id]||await json(staticMode?`./jobs/${id}/episode.json`:`./api/episode/${id}`);$('intro').hidden=true;$('detail').hidden=false;$('title').textContent=data.job.name;$('subtitle').textContent=`${data.job.status} · ${id}`;$('download').href=staticMode?`./jobs/${id}/episode.json`:`./api/episode/${id}`;$('atif').href=staticMode?`./jobs/${id}/atif.json`:`./api/episode/${id}/atif`;$('audit').disabled=staticMode||!data.job.finished_at||!!data.audit_running;$('audit').textContent=data.audit_running?'Visual audit running…':'Run LLMaJ on task video output';const r=data.report;$('facts').replaceChildren();for(const [label,value] of [['Mechanical',r.observed_success===true?'Pass':r.observed_success===false?'Fail':'Unrecorded'],['Replay verified',r.verified_success===true?'Yes':r.verified_success===false?'No':'Unrecorded'],['Control steps',r.control_steps??'—'],['Active seconds',r.elapsed_seconds?Math.round(r.elapsed_seconds):'—'],['Attempt',r.attempt_number??'—']]){const f=el('div');f.append(el('b',String(value)),el('span',label,'muted'));$('facts').append(f)}renderList();render();if(data.audit_error)notice(data.audit_error,true)}catch(e){notice(e.message,true)}}
function reasoningView(traces){
 const texts=[],seen=new Set();let opaque=0;
 const add=text=>{if(typeof text!=='string'||!text.trim())return;const key=text.trim();if(!seen.has(key)){seen.add(key);texts.push(text)}};
 for(const trace of traces||[]){
  const details=trace.reasoning_details||[];
  const detailTexts=details.filter(b=>b.type==='reasoning.text'&&typeof b.text==='string').map(b=>b.text);
  const primary=typeof trace.reasoning==='string'?trace.reasoning:null;
  add(primary);
  const comparable=text=>text.replace(/\s+/g,' ').trim();
  if(!primary||comparable(primary)!==comparable(detailTexts.join(' ')))detailTexts.forEach(add);
  opaque+=details.filter(b=>b.type==='reasoning.encrypted').length;
  for(const entry of trace.blocks||[]){const b=entry.block||entry;if(b.type==='thinking')add(b.thinking);else if(b.type==='redacted_thinking')opaque++}
  // Older direct-provider records can store content blocks directly.
  for(const b of trace.content||[]){if(b.type==='thinking')add(b.thinking);else if(b.type==='redacted_thinking')opaque++}
 }
 return {texts,opaque};
}
function rawDetails(label,value){const d=el('details',undefined,'raw-detail');d.append(el('summary',label),pre(value));return d}
function renderTurn(t,index){
 const d=el('details',undefined,'trajectory-turn');d.open=index===0;
 const before=t.control_steps_before??t.input?.observation?.control_step??'—',after=t.control_steps_after;
 d.append(el('summary',`Turn ${t.turn} · robot step ${before}${after!==undefined?' → '+after:''} · ${t.decision?.operation||'input'}`));
 const cols=el('div',undefined,'columns'),left=el('section',undefined,'turn-action'),right=el('section',undefined,'turn-reasoning');
 left.append(el('h3','Action & feedback'));
 const decision=t.decision||{};
 if(decision.message){left.append(el('h4','Public action rationale'),el('p',decision.message,'readable-text'))}
 left.append(el('div',`${decision.operation||'No operation'}${decision.path?' · '+decision.path:''}`,'operation-label'));
 if(decision.content){left.append(el('h4',decision.operation?.includes('write')?'Submitted source':'Operation content'),pre(decision.content))}
 const feedback=t.result;
 if(feedback&&typeof feedback==='object'){
  const summary=[];if(feedback.returncode!==undefined)summary.push(`Exit ${feedback.returncode}`);if(feedback.accepted_control_steps!==undefined)summary.push(`${feedback.accepted_control_steps} control steps`);if(feedback.status)summary.push(feedback.status.replaceAll('_',' '));
  left.append(el('h4','Execution feedback'));if(summary.length)left.append(el('p',summary.join(' · '),'feedback-summary'));
  if(feedback.output)left.append(el('h4','Captured stdout / stderr'),pre(feedback.output));
  if(feedback.error)left.append(el('p',String(feedback.error),'error readable-text'));
  if(feedback.message)left.append(el('p',feedback.message,'muted readable-text'));
  if(!feedback.output&&!feedback.error&&!feedback.message&&!summary.length)left.append(el('p','Structured feedback is available in the raw record below.','muted'));
 }else left.append(el('p','No execution feedback recorded for this turn.','muted'));
 left.append(rawDetails('Raw decision and result',{decision:t.decision,result_after:t.result}));
 left.append(rawDetails('Observation, prior feedback and native actions',{observation_before:t.input?.observation,feedback_before:t.input?.feedback,control_steps_before:t.control_steps_before,control_steps_after:t.control_steps_after,robot_actions:t.robot_actions}));
 const providerSummary=(t.reasoning||[]).some(trace=>trace?.representation==='provider_generated_summary');
 right.append(el('h3',providerSummary?'Provider-generated thinking summary':'Provider-exposed reasoning'));
 if(providerSummary)right.append(el('p',"This is the provider's readable summary. Original returned blocks and signatures are retained in the source run; public exports represent opaque signatures by hashes.",'muted readable-text'));
 const reasoning=reasoningView(t.reasoning);
 for(const text of reasoning.texts)right.append(el('div',text,'reasoning-text'));
 if(!reasoning.texts.length)right.append(el('p','No readable provider reasoning was returned for this turn. The public action rationale is shown separately.','muted readable-text'));
 if(reasoning.opaque)right.append(el('p',`${reasoning.opaque} opaque or redacted provider block${reasoning.opaque===1?'':'s'} preserved in the episode record; no readable thoughts are inferred.`,'opaque-notice'));
 right.append(rawDetails('Raw provider fields',t.reasoning||[]));cols.append(left,right);d.append(cols);return d;
}
function render(){const c=$('content');c.replaceChildren();if(tab==='raw'){c.append(pre({report:data.report,source_hashes:data.source_hashes,events:data.events}));return}if(tab==='trajectory'){c.append(el('p',data.reasoning_notice,'notice'));data.turns.forEach((t,index)=>c.append(renderTurn(t,index)));const d=el('details');d.append(el('summary','Unmapped provider summaries and execution journal'),pre(data.events));c.append(d);return}const grid=el('div',undefined,'videos');for(const v of data.videos){const f=el('figure'),video=el('video');video.controls=true;video.preload='metadata';video.src=`./jobs/${selected}/${v.name}`;f.append(video,el('figcaption',v.label));grid.append(f)}c.append(grid);if(!data.videos.length)c.append(el('p','No local video available for this record.','notice'));c.append(el('h3','Mechanical verdict'),pre(data.report.verdict||{}),el('h3','Visual predicate audits'));if(!data.audits.length)c.append(el('p','Not audited. A mechanical pass does not imply visual review.','notice'));for(const a of [...data.audits].reverse()){c.append(el('h3',(a.comparison_status||a.review_status||a.status||'unknown').replaceAll('_',' ')));if(a.error)c.append(el('p',a.error,'error'));c.append(el('p',`Visual support: ${a.visual_support_score===null?'Not fully observable':a.visual_support_score??'—'} · observable fraction: ${a.observable_fraction??'—'} · mechanical score unchanged`));for(const check of a.answer?.checks||[])c.append(el('div',`${check.id}: ${check.satisfied===null?'Unknown':check.satisfied?'Supported':'Contradicted'} — ${check.reason}`,'check'));const d=el('details');d.append(el('summary','Predicate source, timestamps, costs & provenance'),pre(a));c.append(d)}}
$('audit').onclick=async()=>{if(staticMode)return;$('audit').disabled=true;try{const r=await fetch(`./api/audit/${selected}`,{method:'POST',headers:{'X-RoboSWE-Review':'1'}});if(!r.ok)throw Error(await r.text());notice('Audit started; the result will appear here.');await select(selected)}catch(e){notice('Audit not started: '+e.message,true);$('audit').disabled=false}};
$('search').oninput=renderList;$('filter').onchange=renderList;$('refresh').onclick=async()=>{await load();if(selected)await select(selected)};for(const b of document.querySelectorAll('[data-tab]'))b.onclick=()=>{tab=b.dataset.tab;if(data)render()};load().then(()=>{const id=location.hash.slice(1);if(records.some(r=>r.id===id))select(id)});setInterval(async()=>{if(!staticMode){await load();if(data?.audit_running)await select(selected)}},15000);
