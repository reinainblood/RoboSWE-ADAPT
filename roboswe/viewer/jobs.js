const $=s=>document.querySelector(s);
let jobs=[],selected=null,tab='video',dispose=null,generation=0;
function path(name){return '/jobs/'+encodeURIComponent(selected.id)+'/'+encodeURIComponent(name);}
function text(tag,value){const el=document.createElement(tag);el.textContent=value;return el;}
async function response(url){const r=await fetch(url);if(!r.ok)throw new Error(`Could not read saved artifact (${r.status})`);return r;}
async function showTab(name){
  tab=name;const token=++generation;dispose?.();dispose=null;$('#content').replaceChildren();
  for(const b of document.querySelectorAll('[data-tab]'))b.setAttribute('aria-selected',String(b.dataset.tab===name));
  if(name==='video'){const {showVideo}=await import('./video.js');if(token!==generation)return;const videoJob=selected;const container=document.createElement('div');$('#content').replaceChildren(container);const artifactUrl=file=>'/jobs/'+encodeURIComponent(videoJob.id)+'/'+encodeURIComponent(file);const cleanup=await showVideo(container,videoJob,artifactUrl);if(token!==generation)cleanup();else dispose=cleanup;return;}
  const file=selected.artifacts[name];if(!file){$('#content').append(text('p','No artifact was recorded for this view.'));return;}
  $('#content').append(text('p','Loading saved output…'));
  try{
    if(name==='before'||name==='after'){
      const [{mount},r]=await Promise.all([import('./scene.js'),response(path(file))]);
      const data=new Uint8Array(await r.arrayBuffer());if(token!==generation)return;
      const container=document.createElement('div');$('#content').replaceChildren(container);
      const cleanup=await mount(container,data);if(token!==generation)cleanup();else dispose=cleanup;
    }else{
      const value=await (await response(path(file))).text();if(token!==generation)return;
      $('#content').replaceChildren(text('pre',value||'(empty)'));
    }
  }catch(e){if(token===generation)$('#content').replaceChildren(text('pre',e.message));}
}
function select(job){
  selected=job;$('#detail').hidden=false;$('#welcome').hidden=true;
  $('#name').textContent=job.name;$('#status').textContent=({snapshot:'Snapshot','benchmark-task':'Benchmark task','benchmark-campaign':'Campaign','independent-verification':'Verification','verification-video':'Replay video','learning-control':'Cold control','infrastructure-control':'Infrastructure control'}[job.kind]||'Execution')+' · '+job.status;
  $('#status').className='badge '+job.status;
  const scene=job.scenes.after||job.scenes.before||{},session=job.session||{};
  const values=[['Task',job.task||scene.task||session.config?.task||'—'],['Captured step',scene.step??'—'],['Created',new Date(job.created_at).toLocaleString()],['Exit code',job.exit_code??'not an execution'],['Verification',job.verifier_status||'—'],['Video',job.video_provenance||(job.video?'original recording':'not recorded in this record')],['Visual audit',job.visual_audit_status||'not requested']];
  $('#facts').replaceChildren(...values.map(([k,v])=>{const d=text('div',k+' · ');d.className='fact';d.append(text('b',String(v)));return d;}));
  $('#warnings').replaceChildren();
  for(const error of job.artifact_errors||[]){const w=text('div',`${error.stage} artifact unavailable: ${error.error}`);w.className='warning';$('#warnings').append(w);}
  if(job.status==='running'){$('#warnings').append(text('p','This record has not been finalized. It may still be running, or its host process may have stopped.'));}
  for(const b of document.querySelectorAll('[data-tab]'))b.disabled=b.dataset.tab==='video'?!job.video?.views?.some(v=>job.artifacts[v.wall]||job.artifacts[v.simulation]):!job.artifacts[b.dataset.tab];
  $('#files').replaceChildren();
  for(const [label,file] of Object.entries({...job.artifacts,manifest:'job.json'})){
    const a=text('a','Download '+label);a.href=path(file);a.download=file;$('#files').append(a);
  }
  for(const b of document.querySelectorAll('.job'))b.setAttribute('aria-current',String(b.dataset.id===job.id));
  const hasVideo=job.video?.views?.some(v=>job.artifacts[v.wall]||job.artifacts[v.simulation]);
  const first=hasVideo?'video':job.artifacts[tab]?tab:['report','after','before','source','stdout','stderr'].find(k=>job.artifacts[k]);
  if(first)showTab(first);else{$('#content').replaceChildren(text('p','No artifacts have been saved yet.'));}
}
async function refresh(){
  try{
    jobs=await (await response('/api/jobs')).json();$('#error').textContent='';$('#count').textContent=`${jobs.length} saved record${jobs.length===1?'':'s'}`;
    $('#jobs').replaceChildren();
    for(const job of jobs){const b=document.createElement('button');b.className='job';b.dataset.id=job.id;b.setAttribute('aria-current',String(selected?.id===job.id));b.append(text('strong',job.name));const badge=text('span',job.kind+' · '+job.status);badge.className='badge '+job.status;b.append(badge,text('small',new Date(job.created_at).toLocaleString()));b.onclick=()=>select(job);$('#jobs').append(b);}
    if(!selected&&jobs.length)select(jobs[0]);
    else if(selected){const update=jobs.find(j=>j.id===selected.id);if(update)select(update);}
  }catch(e){$('#error').textContent=e.message;}
}
for(const b of document.querySelectorAll('[data-tab]'))b.onclick=()=>showTab(b.dataset.tab);
$('#refresh').onclick=refresh;
refresh();

const tabs=[...document.querySelectorAll('[data-tab]')];
for(const b of tabs)b.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const enabled=tabs.filter(t=>!t.disabled);const i=enabled.indexOf(b);const next=e.key==='Home'?enabled[0]:e.key==='End'?enabled.at(-1):enabled[(i+(e.key==='ArrowRight'?1:-1)+enabled.length)%enabled.length];next.focus();next.click();});
