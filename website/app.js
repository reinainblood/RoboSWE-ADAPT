const $=s=>document.querySelector(s);let evidence,allTasks=[],expanded=false;const demo=$('#demo');
fetch('assets/blender-evidence.json').then(r=>r.json()).then(d=>{evidence=d;syncTrace()}).catch(()=>{$('#code').textContent='Evidence could not load. Reload this page to try again.'});
function syncTrace(){if(!evidence)return;let event=evidence.events[0];for(const e of evidence.events){if(e.seconds<=demo.currentTime)event=e;else break}$('#step').textContent=`STEP ${Math.min(evidence.steps,Math.floor(demo.currentTime*20))} / ${evidence.steps}`;$('#operation').textContent=event.operation+(event.path?' · '+event.path:'');$('#turn').textContent='TURN '+event.turn;$('#rationale').textContent=event.rationale||'No action rationale recorded for this turn.';$('#code').textContent=event.code||event.operation}
demo.addEventListener('timeupdate',syncTrace);demo.addEventListener('seeked',syncTrace);
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{const t=demo.currentTime,playing=!demo.paused;demo.src=`assets/blender-${b.dataset.view}.mp4`;demo.addEventListener('loadedmetadata',()=>{demo.currentTime=t;if(playing)demo.play().catch(()=>{})},{once:true});document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x===b))}));
const hv=$('.hero-visual video');$('#hero-play').addEventListener('click',()=>{if(hv.paused){hv.play().then(()=>$('#hero-play').textContent='Pause preview').catch(()=>$('#hero-play').textContent='Try playback again')}else{hv.pause();$('#hero-play').textContent='Play preview'}});
const stages=[['Trial 1: task instruction and an empty codebase.','The agent receives the task instruction, camera observations, robot sensors, and the native control interface. It can inspect state, write or revise Python, and execute programs within the configured time and control-step limits.'],...Array.from({length:4},(_,i)=>[`Trial ${i+2}: the same initial physical state, with retained history.`,'The agent retains the codebase, notes, and conversation from preceding trials. It has the same observation tools, native control interface, model parameters, and budgets. Exploration, code revision, and recovery remain available; the protocol does not prescribe a stage-specific strategy.'])];
document.querySelectorAll('[data-trial]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-trial]').forEach(x=>x.setAttribute('aria-selected',x===b));const s=stages[+b.dataset.trial-1];$('#trial-description').replaceChildren(Object.assign(document.createElement('strong'),{textContent:s[0]}),Object.assign(document.createElement('p'),{textContent:s[1]}))}));
fetch('assets/tasks.json').then(r=>r.json()).then(d=>{allTasks=d;renderTasks()}).catch(()=>$('#task-count').textContent='Catalogue unavailable');
function renderTasks(){const q=$('#search').value.toLowerCase();const filtered=allTasks.filter(n=>n.toLowerCase().includes(q));$('#task-count').textContent=`${filtered.length} selections`;$('#task-list').replaceChildren(...filtered.slice(0,expanded||q?999:12).map(n=>{const a=document.createElement('a');a.className='task';a.href='https://robocasa.ai/docs/build/html/tasks/composite_tasks.html';a.textContent=n.replace(/([a-z])([A-Z])/g,'$1 $2');const tag=document.createElement('span');tag.textContent=n.includes('+')?'Proposed merge':'↗';a.append(tag);return a}));$('#more').hidden=!!q||filtered.length<=12;$('#more').textContent=expanded?'Show fewer tasks':'Show all selected tasks'}$('#search').addEventListener('input',renderTasks);$('#more').addEventListener('click',()=>{expanded=!expanded;renderTasks()});

$('#demo-play').addEventListener('click',()=>{if(demo.paused)demo.play().catch(()=>{});else demo.pause()});

// Public results are a curated, static export. Never infer a score from job status.
function resultElement(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e}
function resultLabel(a){if(a.official_success===true)return 'Verified success';if(a.official_success===false)return 'Verified failure';return ({not_started:'Not started',incomplete:'Incomplete',awaiting_verification:'Awaiting replay',verification_invalid:'Replay invalid',provider_model_failure:'Provider / model failure',infrastructure_failure:'Infrastructure failure'})[a.category]||'Unverified'}
function displayNumber(value,suffix=''){return typeof value==='number'&&Number.isFinite(value)?`${value.toLocaleString(undefined,{maximumFractionDigits:1})}${suffix}`:'—'}
function serviceTimingCell(seconds,coverage){
 const cell=resultElement('td',displayNumber(seconds));
 const known=coverage?.timed_completed_logical_calls??coverage?.timed_completed_calls;
 const label=coverage?.status==='complete_for_recorded_transmissions'?'recorded requests covered':'partial / unknown coverage';
 cell.append(resultElement('small',typeof known==='number'?`${known} completed call${known===1?'':'s'} · ${label}`:'Service timing unavailable','timing-coverage'));
 return cell;
}
function operationalTimingTable(attempts){
 const table=resultElement('table');
 table.append(resultElement('caption','Operational stages in seconds, measured from recorded phase boundaries. Setup and verification are separate from the original active trial. Readback columns cover simulator artifact transfers, not the complete evidence-archive workflow. Missing or incomplete intervals remain unavailable (—). These are elapsed intervals, not pure compute or billed resource time; do not add them to the nested service timings above.'));
 const thead=resultElement('thead'),head=resultElement('tr');
 for(const label of ['Trial','Simulator setup','Original commit','Original readback','Verifier setup','Replay execution','Replay commit','Replay readback']){
  const th=resultElement('th',label);th.scope='col';head.append(th);
 }
 thead.append(head);table.append(thead);const body=resultElement('tbody');
 for(const attempt of attempts){
  const t=attempt.operational_timing||{},original=t.original_transport||{},replay=t.replay_transport||{},row=resultElement('tr');
  for(const value of [attempt.attempt_number,original.setup_seconds,original.commit_seconds,original.artifact_readback_seconds,replay.setup_seconds,t.replay_execution_seconds,replay.commit_seconds,replay.artifact_readback_seconds])row.append(resultElement('td',displayNumber(value)));
  body.append(row);
 }
 table.append(body);const region=resultElement('div',undefined,'result-table-scroll');
 region.tabIndex=0;region.setAttribute('role','region');region.setAttribute('aria-label','Operational timing table');region.append(table);return region;
}
function renderResults(data){
 const time=new Date(data.generated_at);$('#results-asof').textContent=Number.isNaN(time.getTime())?'Snapshot time unavailable':`Snapshot · ${time.toISOString().replace('T',' ').replace(/\.\d+Z$/,' UTC')}`;
 const list=$('#results-list');list.replaceChildren();
 if(!Array.isArray(data.series)||!data.series.length){list.append(resultElement('p','No five-trial series has been published yet.'));return}
 for(const row of data.series){
  const article=resultElement('article',undefined,'result-series');
  const heading=resultElement('div',undefined,'result-heading');heading.append(resultElement('h3',row.task||'Task'),resultElement('span',row.complete===true?'Complete · five verified trials':`Recorded state: ${row.recorded_status||'unknown'}`,'result-state'));article.append(heading);
  const p=row.participant||{};article.append(resultElement('p',`${p.provider||'Provider unrecorded'} · ${p.model||'Model unrecorded'}`,'result-model'),resultElement('p',`Context policy: ${(typeof row.context_policy==='object'?row.context_policy?.method:row.context_policy)||'Unrecorded'}`,'result-context'));
  if(p.thinking_display==='summarized')article.append(resultElement('p','Provider thinking summaries requested; this setting does not imply that full hidden reasoning is available.','small'));
  if(row.protocol_review&&Object.keys(row.protocol_review).length){const review=row.protocol_review;const note=resultElement('aside',undefined,'protocol-review');note.append(resultElement('strong',review.primary_leaderboard_eligible===false?'Development series · excluded from the primary leaderboard':'Protocol review'));if(review.classification)note.append(resultElement('span',review.classification.replaceAll('_',' '),'small'));if(review.reason)note.append(resultElement('p',review.reason));if(review.mechanical_result_preserved)note.append(resultElement('p',typeof review.mechanical_result_preserved==='string'?review.mechanical_result_preserved:'Original mechanical result preserved.','small'));article.append(note)}
  if(row.context_audit){const audit=row.context_audit;const line=resultElement('p',`Context slides recorded: ${displayNumber(audit.slide_count)}`,'small');article.append(line)}
  const episodeLinks={'20261020T172747Z-ab32071f77':['hotdog-opus-trial-01','20261020T172749Z-d693660793'],'20261020T123141Z-f2476610f1':['beverage-argon-trial-01','20261020T123142Z-3749649221'],'20261020T083619Z-f0b03b7329':['development-argon-01','20261020T083619Z-460161983a'],'20261020T105702Z-9ef43916e5':['toast-argon-trial-01','20261020T105702Z-ba1714496d'],'20261020T105702Z-a362574b5c':['toast-opus-trial-01','20261020T105702Z-a9081835ff']};
  if(episodeLinks[row.series_id]){const [folder,job]=episodeLinks[row.series_id];const link=resultElement('a','Inspect trial 1: four cameras, trajectory & reasoning ↗','episode-link');link.href=`episodes/${folder}/#${job}`;article.append(link)}
  if(row.series_id==='20261020T123141Z-f2476610f1'){
   for(const [folder,job,label] of [['beverage-argon-trial-02','20261020T131409Z-24893adde2','Trial 2: verified failure ↗'],['beverage-argon-trial-03','20261020T135653Z-abbb79f2ad','Trial 3: verified success ↗']]){const link=resultElement('a',label,'episode-link');link.href=`episodes/${folder}/#${job}`;article.append(link)}
  }
  if(row.series_id==='20261020T172747Z-ab32071f77'){const link=resultElement('a','Trial 2: verified failure ↗','episode-link');link.href='episodes/hotdog-opus-trial-02/#20261020T181702Z-79d88e1c95';article.append(link)}
  if(row.series_id==='20261020T172747Z-ab32071f77'){const link=resultElement('a','Trial 3: verified failure ↗','episode-link');link.href='episodes/hotdog-opus-trial-03/#20261020T190741Z-06e26cff6f';article.append(link)}
  if(row.series_id==='20261020T130317Z-20ccc4226b'){
   for(const [folder,job,label] of [['beverage-opus-trial-01','20261020T130317Z-d06e1d70d3','Trial 1: verified failure ↗'],['beverage-opus-trial-02','20261020T135735Z-7882496ac8','Trial 2: verified failure ↗'],['beverage-opus-trial-03','20261020T150039Z-2411916e03','Trial 3: verified failure ↗'],['beverage-opus-trial-04','20261020T153422Z-360afcffad','Trial 4: verified failure ↗'],['beverage-opus-trial-05','20261020T160340Z-7227003a7c','Trial 5: verified failure ↗']]){const link=resultElement('a',label,'episode-link');link.href=`episodes/${folder}/#${job}`;article.append(link)}
  }
  const stats=resultElement('dl',undefined,'result-stats');for(const [label,value] of [['Planned trials',row.planned_trials],['Checked outcomes',row.verified_trials],['Incomplete / unverified',row.incomplete_trials],['Verified successes',row.descriptive_metrics?.verified_success_count]]){const wrap=resultElement('div');wrap.append(resultElement('dt',label),resultElement('dd',displayNumber(value)));stats.append(wrap)}article.append(stats);
  const trials=resultElement('ol',undefined,'result-trials');for(const a of row.attempts||[]){const li=resultElement('li',undefined,'result-trial');li.dataset.outcome=a.official_success===true?'pass':a.official_success===false?'fail':'pending';li.append(resultElement('span',`TRIAL ${a.attempt_number}`,'trial-number'),resultElement('strong',resultLabel(a)));if(a.official_success!==null&&typeof a.official_success==='boolean')li.append(resultElement('span',`${displayNumber(a.control_steps)} control steps`));else li.append(resultElement('span','Score pending'));if(a.budget_censored)li.append(resultElement('span','Budget censored','small'));trials.append(li)}article.append(trials);
  const details=resultElement('details',undefined,'result-details');details.append(resultElement('summary','Timing and evidence details'));
  details.append(resultElement('p',`Series receipt: ${row.series_id}`,'small'));
  const table=resultElement('table');const caption=resultElement('caption','Original attempt timing in seconds. Model wrapper includes count-service, inference, and orchestration. Service columns are components of the wrapper, not additional time. Count-service totals may include retry/backoff. Pure model compute, transport-only, and code-only time are unknown. — means unavailable; replay and setup are excluded.');table.append(caption);
  const head=resultElement('tr');for(const label of ['Trial','Active','Model wrapper','Count service','Inference request','Operation roundtrip','Passive wait']){const th=resultElement('th',label);th.scope='col';head.append(th)}
  const thead=resultElement('thead');thead.append(head);table.append(thead);const body=resultElement('tbody');
  for(const a of row.attempts||[]){const tr=resultElement('tr');const t=a.timing||{};
   for(const value of [a.attempt_number,t.active_seconds,t.model_roundtrip_seconds])tr.append(resultElement('td',displayNumber(value)));
   tr.append(serviceTimingCell(t.count_service_completed_roundtrip_seconds,t.count_service_coverage),serviceTimingCell(t.inference_provider_completed_roundtrip_seconds,t.inference_provider_coverage));
   for(const value of [t.operation_roundtrip_seconds,t.passive_wait_seconds])tr.append(resultElement('td',displayNumber(value)));body.append(tr)}
  table.append(body);const scroll=resultElement('div',undefined,'result-table-scroll');scroll.tabIndex=0;scroll.setAttribute('role','region');scroll.setAttribute('aria-label','Trial timing table');scroll.append(table);details.append(scroll);
  if((row.attempts||[]).some(a=>a.operational_timing))details.append(operationalTimingTable(row.attempts));
  article.append(details);list.append(article);
 }
}
fetch('assets/adaptive-results.json').then(r=>{if(!r.ok)throw new Error('Results unavailable');return r.json()}).then(renderResults).catch(()=>{$('#results-asof').textContent='Snapshot unavailable';$('#results-list').replaceChildren(resultElement('p','The results snapshot could not be loaded. No scores are inferred from missing evidence.'))});
