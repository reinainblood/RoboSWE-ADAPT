import * as THREE from './vendor/three.module.js';
import {OrbitControls} from './vendor/OrbitControls.js';

function bytes(s) { return Uint8Array.from(atob(s), c=>c.charCodeAt(0)); }
function floats(s) { return new Float32Array(bytes(s).buffer); }

export async function mount(el, encoded) {
  el.innerHTML = `<style>
    .robot-view{font:14px system-ui;color:#dbe5ef;background:#14202b;border-radius:12px;overflow:hidden}
    .robot-view header{padding:12px 16px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
    .robot-view button{padding:7px 11px;border:1px solid #758796;border-radius:6px;color:inherit;background:#263847;cursor:pointer}
    .robot-view .stage{height:580px;min-height:350px;position:relative;touch-action:none}
    .robot-view canvas{display:block;width:100%;height:100%}
    .robot-view footer{padding:9px 16px;font-size:12px;color:#a8bdcc}
    .robot-view .label{flex:1;min-width:200px}
  </style><section class="robot-view"><header>
    <span class="label">Loading simulator snapshot…</span>
    <button data-view="room">Room</button><button data-view="robot">Around robot</button>
    <button data-view="top">Overhead</button>
    <label><input type="checkbox" class="walls"> Walls</label>
    <button class="fullscreen">Full screen</button>
  </header><div class="stage"></div><footer>Drag to orbit · scroll/pinch to zoom · right-drag to pan · inspection only, no robot controls</footer></section>`;
  const stage = el.querySelector('.stage');
  let payload;
  try {
    const stream = new Blob([typeof encoded === "string" ? bytes(encoded) : encoded]).stream().pipeThrough(new DecompressionStream('gzip'));
    payload = JSON.parse(await new Response(stream).text());
  } catch (e) { el.querySelector('.label').textContent = 'Snapshot decode failed: '+e.message; return ()=>{}; }
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#202f3c');
  const camera = new THREE.PerspectiveCamera(45, 1, 0.02, 100);
  camera.up.set(0,0,1);
  const renderer = new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
  stage.append(renderer.domElement);
  const controls = new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true; controls.minDistance=.15; controls.maxDistance=25;
  scene.add(new THREE.HemisphereLight(0xffffff,0x647080,3));
  const key = new THREE.DirectionalLight(0xffffff,3);key.position.set(0,-4,7);scene.add(key);
  const fill = new THREE.DirectionalLight(0xe0edff,1.5);fill.position.set(6,1,4);scene.add(fill);
  const meshes = new Map(), textures = new Map(), materials = new Map(), objects=[];
  for (const [id,m] of Object.entries(payload.meshes)) {
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(floats(m.positions),3));
    if(m.indices)g.setIndex(new THREE.BufferAttribute(new Uint32Array(bytes(m.indices).buffer),1));
    if(m.uv)g.setAttribute('uv',new THREE.BufferAttribute(floats(m.uv),2));
    if(m.normals)g.setAttribute('normal',new THREE.BufferAttribute(floats(m.normals),3));else g.computeVertexNormals();
    meshes.set(Number(id),g);
  }
  const loader=new THREE.TextureLoader();
  await Promise.all(Object.entries(payload.textures).map(async([id,png])=>{
    const t=await loader.loadAsync('data:image/png;base64,'+png);t.colorSpace=THREE.SRGBColorSpace;
    t.flipY=false; // MuJoCo tex_data is already bottom-up, unlike browser images.
    t.wrapS=t.wrapT=THREE.RepeatWrapping;textures.set(Number(id),t);
  }));
  for (const g of payload.geoms) {
    let geometry;
    switch(g.kind){
      case 7: geometry=meshes.get(g.mesh);break;
      case 6: geometry=new THREE.BoxGeometry(...g.size.map(x=>2*x));break;
      case 2: geometry=new THREE.SphereGeometry(g.size[0],20,12);break;
      case 4: geometry=new THREE.SphereGeometry(1,20,12);geometry.scale(...g.size);break;
      case 5: geometry=new THREE.CylinderGeometry(g.size[0],g.size[0],2*g.size[1],24);geometry.rotateX(Math.PI/2);break;
      case 3: geometry=new THREE.CapsuleGeometry(g.size[0],2*g.size[1],8,16);geometry.rotateX(Math.PI/2);break;
      case 0: geometry=new THREE.PlaneGeometry(2*(g.size[0]||8),2*(g.size[1]||8));break;
      default: continue;
    }
    if(!geometry)continue;
    // MuJoCo's generated primitive UVs do not match Three's appliance-box UVs.
    // Use textures for exported mesh UVs and room surfaces; keep other primitives solid.
    const canTexture=!!geometry.getAttribute('uv')&&(g.kind===7||/^(floor|wall)/.test(g.name));
    const materialKey=JSON.stringify([g.rgba,g.texture,g.repeat,canTexture]);
    if(!materials.has(materialKey)){
      const color=new THREE.Color().setRGB(...g.rgba.slice(0,3));
      const opts={color,roughness:.8,metalness:.05,side:THREE.DoubleSide,opacity:g.rgba[3],transparent:g.rgba[3]<1};
      if(textures.has(g.texture)&&canTexture){opts.map=textures.get(g.texture).clone();opts.map.repeat.set(...g.repeat);opts.map.needsUpdate=true;}
      materials.set(materialKey,new THREE.MeshStandardMaterial(opts));
    }
    const obj=new THREE.Mesh(geometry,materials.get(materialKey));obj.name=g.name;
    const r=g.rotation,p=g.position;
    obj.matrix.set(r[0],r[1],r[2],p[0],r[3],r[4],r[5],p[1],r[6],r[7],r[8],p[2],0,0,0,1);obj.matrixAutoUpdate=false;
    obj.userData.shell=g.name.startsWith('wall_')||g.name.includes('ceiling');
    obj.visible=!obj.userData.shell;scene.add(obj);objects.push(obj);
  }
  scene.updateMatrixWorld(true);
  const robotBox=new THREE.Box3();
  for(const o of objects)if(/^(robot0|mobilebase0|gripper0)/.test(o.name))robotBox.expandByObject(o);
  const center=robotBox.isEmpty()?new THREE.Vector3(3.1,-1.4,1):robotBox.getCenter(new THREE.Vector3());
  function view(name){
    if(name==='robot'){controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(1.8,-2.5,1.4));}
    else if(name==='top'){controls.target.set(2.75,-1.3,.6);camera.position.set(2.75,-1.31,8);}
    else{controls.target.set(2.75,-1.1,1);camera.position.set(6.5,-7,4.8);}
    controls.update();
  }
  for(const b of el.querySelectorAll('[data-view]'))b.onclick=()=>view(b.dataset.view);
  el.querySelector('.walls').onchange=e=>{for(const o of objects)if(o.userData.shell)o.visible=e.target.checked;};
  el.querySelector('.fullscreen').onclick=async()=>{try{await el.querySelector('.robot-view').requestFullscreen();}catch(e){el.querySelector('footer').textContent='Full screen unavailable here. Open the saved standalone viewer for more room.';}};
  el.querySelector('.label').textContent=`After · step ${payload.step} · ${payload.task.replace('robocasa/','')}`;
  el.querySelector('.robot-view').dataset.loaded='true';
  function resize(){const w=stage.clientWidth,h=stage.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
  const observer=new ResizeObserver(resize);observer.observe(stage);view('room');resize();
  renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
  return ()=>{observer.disconnect();renderer.setAnimationLoop(null);controls.dispose();renderer.dispose();for(const g of meshes.values())g.dispose();for(const m of materials.values()){m.map?.dispose();m.dispose();}for(const t of textures.values())t.dispose();};
}

export default {async render({model,el}){return await mount(el,model.get('scene_data'));}};
