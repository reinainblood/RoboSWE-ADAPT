/* Excerpts come only from recorded turns, aligned by original control-step boundaries. */
(()=>{
 const controls=document.createElement('label');controls.className='overlay-toggle';
 const toggle=document.createElement('input');toggle.type='checkbox';toggle.id='show-recorded-overlay';
 controls.append(toggle,document.createTextNode(' Show recorded reasoning / tool overlay (simulation time)'));
 document.querySelector('.tabs').before(controls);
 const bound=new WeakSet();
 function bind(){
  for(const video of document.querySelectorAll('#content video')){
   if(bound.has(video))continue;bound.add(video);
   const figure=video.closest('figure');figure.classList.add('recorded-video');
   const box=document.createElement('div');box.className='recorded-overlay';box.hidden=true;figure.append(box);
   function update(){
    box.hidden=!toggle.checked;if(box.hidden||!data)return;
    const fps=data.job.video?.control_fps;
    if(typeof fps!=='number'||fps<=0){box.textContent='Simulation-time alignment unavailable';return}
    const step=Math.floor(video.currentTime*fps);
    const eligible=data.turns.filter(t=>Number.isInteger(t.control_steps_before)&&t.control_steps_before<=step);
    const turn=eligible[eligible.length-1];if(!turn){box.textContent='No turn boundary recorded';return}
    let excerpt='';let kind='Public action rationale';
    for(const trace of turn.reasoning||[]){
     if(typeof trace.reasoning==='string'&&trace.reasoning.trim()){excerpt=trace.reasoning;kind='Provider-exposed reasoning';break}
     for(const entry of trace.blocks||[]){const block=entry.block||entry;if(typeof block.thinking==='string'&&block.thinking.trim()){excerpt=block.thinking;kind='Provider-exposed reasoning';break}}
     if(excerpt)break;
    }
    excerpt=excerpt||turn.decision?.message||'No readable excerpt recorded';
    const clean=excerpt.replace(/\s+/g,' ').trim();
    box.textContent=`Simulation step ${step} · turn ${turn.turn} · ${turn.decision?.operation||'input'}\n${kind}: ${clean.slice(0,220)}${clean.length>220?'…':''}`;
   }
   video.addEventListener('timeupdate',update);video.addEventListener('seeked',update);toggle.addEventListener('change',update);update();
  }
 }
 new MutationObserver(bind).observe(document.querySelector('#content'),{childList:true});bind();
})();
