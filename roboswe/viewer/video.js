export async function showVideo(el, job, url) {
  const runs=Object.keys(job.video_runs||{}).length?job.video_runs:{recording:job.video};
  let manifest=runs[job.default_video_run]||Object.values(runs)[0];
  let available=(manifest?.views||[]).filter(v=>job.artifacts[v.wall]||job.artifacts[v.simulation]);
  if(!available.length){el.textContent='No video was recorded for this job. A 3D snapshot is not a recording of its movement.';return ()=>{};}
  el.innerHTML=`<style>
  .video-tools{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin-bottom:14px}
  .video-tools label{display:flex;flex-direction:column;gap:5px;font-size:13px;color:#a7bec9}
  .video-tools select{font:inherit;background:#223540;color:#e6eef0;border:1px solid #3e5561;border-radius:6px;padding:8px}
  .job-video{width:100%;max-height:70vh;background:#000;border-radius:9px;display:block}
  .video-readout{font:13px ui-monospace,monospace;margin:12px 0;line-height:1.6}
  </style><div class="video-tools">
  <label>Recording<select aria-label="Recording source"></select></label>
  <label>View<select aria-label="Video view"></select></label>
  <label>Timing<select aria-label="Video timing"><option value="wall">Wall clock · includes waits</option><option value="simulation">Simulation · control steps</option></select></label>
  <label>Playback speed<select aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
  <button class="back-frame">Previous recorded frame</button><button class="next-frame">Next recorded frame</button>
  </div><video class="job-video" aria-label="Robot attempt recording" controls playsinline preload="metadata"></video>
  <div class="video-readout" aria-live="off"></div>
  <p class="muted">Wall clock preserves actual elapsed time, including renderer/encoder work and waits. Simulation timing shows each captured control step at the configured control frequency. Neither view interpolates a path between endpoints.</p>
  <p class="video-warning warning" hidden></p>`;
  const [recordings,views,timing,speed]=el.querySelectorAll('select'), video=el.querySelector('video'),readout=el.querySelector('.video-readout');
  for(const [key,value] of Object.entries(runs)){
    if(value?.views?.length)recordings.add(new Option(({original:'Original attempt',replay:'Independent replay'})[key]||job.video_provenance||'Original recording',key));
  }
  recordings.value=job.default_video_run||recordings.options[0].value;
  let events=[],lastStep=null,loadGeneration=0,disposed=false;
  async function chooseRecording(){
    const generation=++loadGeneration;
    video.pause();video.removeAttribute('src');video.load();
    manifest=runs[recordings.value];
    available=(manifest?.views||[]).filter(v=>job.artifacts[v.wall]||job.artifacts[v.simulation]);
    const previousView=views.value;
    views.replaceChildren();
    for(const view of available)views.add(new Option(`${view.label} · ${view.width}×${view.height}`,view.id));
    if(available.some(v=>v.id===previousView))views.value=previousView;
    else views.value=(available.find(v=>/agentview_left/.test(v.observation_key||v.label))||available[0])?.id||'';
    events=[];lastStep=null;
    const warning=el.querySelector('.video-warning');warning.hidden=true;
    try{
      const response=await fetch(url(manifest.timeline||'video-timeline.jsonl'));
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const loaded=(await response.text()).trim().split('\n').filter(Boolean).map(line=>JSON.parse(line));
      if(disposed||generation!==loadGeneration)return;
      events=loaded;
    }catch(e){
      if(disposed||generation!==loadGeneration)return;
      warning.hidden=false;warning.textContent='Timeline unavailable: '+e.message;
    }
    if(!disposed&&generation===loadGeneration)load();
  }
  function timestamp(event){return timing.value==='wall'?event.wall_seconds:event.simulation_video_seconds;}
  function eventAt(t){let current=events[0];for(const event of events){if(timing.value==='simulation'&&['waiting','stop'].includes(event.event))continue;if(timestamp(event)>t+.002)break;current=event;}return current;}
  function update(){
    const event=eventAt(video.currentTime);
    if(!event){readout.textContent='No timing metadata';return;}
    lastStep=event.step;
    readout.textContent=`Step ${event.step} · simulation ${event.sim_seconds.toFixed(2)} s · wall ${event.wall_seconds.toFixed(2)} s${event.event==='waiting'?' · waiting between control steps':''}`;
  }
  function load(){
    const view=available.find(v=>v.id===views.value);
    if(!view)return;
    for(const option of timing.options)option.disabled=!job.artifacts[view[option.value]];
    if(!job.artifacts[view[timing.value]])timing.value=job.artifacts[view.wall]?'wall':'simulation';
    const event=events.find(e=>e.step===lastStep&&e.event==='step');
    video.src=url(view[timing.value]);video.load();video.playbackRate=Number(speed.value);
    video.addEventListener('loadedmetadata',()=>{if(event)video.currentTime=timestamp(event);update();},{once:true});
  }
  function frame(direction){
    video.pause();const times=[...new Set(events.filter(e=>timing.value==='wall'||e.event!=='waiting').map(timestamp))];
    const next=direction>0?times.find(t=>t>video.currentTime+.002):times.findLast(t=>t<video.currentTime-.002);
    if(next!==undefined)video.currentTime=Math.min(next,Math.max(0,video.duration-.001));
  }
  recordings.onchange=chooseRecording;views.onchange=load;timing.onchange=load;speed.onchange=()=>{video.playbackRate=Number(speed.value);};
  el.querySelector('.back-frame').onclick=()=>frame(-1);el.querySelector('.next-frame').onclick=()=>frame(1);
  video.ontimeupdate=update;
  video.onerror=()=>{el.querySelector('.video-warning').hidden=false;el.querySelector('.video-warning').textContent='This recording could not be decoded. Other views and the downloadable files remain available.';};
  await chooseRecording();
  return ()=>{disposed=true;++loadGeneration;video.pause();video.removeAttribute('src');video.load();};
}
