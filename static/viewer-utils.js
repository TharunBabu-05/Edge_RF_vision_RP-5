/** Shared viewer lifecycle, safe text rendering and frame-rate independent IMU motion. */
export const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
export function escapeHTML(value) {
  const node = document.createElement('span'); node.textContent = String(value); return node.innerHTML;
}
export function disposeGroup(group) {
  const geometries = new Set(), materials = new Set(), textures = new Set();
  group.traverse(object => {
    if (object.geometry) geometries.add(object.geometry);
    for (const material of (Array.isArray(object.material) ? object.material : [object.material])) {
      if (!material) continue;
      materials.add(material);
      for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
    }
  });
  group.clear(); textures.forEach(x => x.dispose()); materials.forEach(x => x.dispose()); geometries.forEach(x => x.dispose());
}

export class OrientationSync {
  constructor(THREE, camera, controls) {
    this.THREE=THREE; this.camera=camera; this.controls=controls; this.active=false;
    this.reference=null; this.target=new THREE.Quaternion(); this.current=new THREE.Quaternion();
    this.timer=null; this.generation=0; this.lastFrame=performance.now();
    this.status=document.createElement('div'); this.status.className='orientation-status'; this.status.hidden=true;
    this.status.setAttribute('role','status'); document.body.append(this.status);
  }
  setEnabled(enabled) {
    this.active=enabled; this.generation++; clearTimeout(this.timer); this.status.hidden=!enabled;
    this.controls.enabled=!enabled;
    if(enabled){this.recenter(); this.poll(this.generation);}
    else {this.camera.up.set(0,1,0); this.controls.update();}
  }
  recenter() {
    this.reference=null; this.target.identity(); this.current.identity();
    this.offset=this.camera.position.clone().sub(this.controls.target); this.up=this.camera.up.clone();
    this.lastFrame=performance.now(); this.status.textContent='Waiting for fresh IMU readings…';
  }
  async poll(generation) {
    if(!this.active || generation!==this.generation) return;
    try {
      if(document.hidden) return;
      const response=await fetch('/api/imu/orientation',{signal:AbortSignal.timeout(3000)});
      const data=await response.json();
      if(generation!==this.generation) return;
      if(!response.ok || !data.ok){this.status.textContent='IMU unavailable · camera orientation held'; return;}
      const {THREE}=this, radians=Math.PI/180;
      const q=new THREE.Quaternion().setFromEuler(new THREE.Euler(data.pitch*radians,data.yaw*radians,-data.roll*radians,'YXZ'));
      if(!this.reference) this.reference=q.clone().invert();
      this.target.copy(q).multiply(this.reference);
      this.status.textContent=`${data.simulated?'DEMO IMU':'IMU live'} · yaw ${data.yaw.toFixed(0)}° · pitch ${data.pitch.toFixed(0)}° · roll ${data.roll.toFixed(0)}° · R to recenter`;
    } catch {if(generation===this.generation)this.status.textContent='IMU connection lost · camera orientation held';}
    finally {if(this.active && generation===this.generation)this.timer=setTimeout(()=>this.poll(generation),100);}
  }
  update() {
    const now=performance.now(), dt=Math.min(.1,(now-this.lastFrame)/1000); this.lastFrame=now;
    if(!this.active) return;
    this.current.slerp(this.target,1-Math.exp(-10*dt));
    this.camera.position.copy(this.offset).applyQuaternion(this.current).add(this.controls.target);
    this.camera.up.copy(this.up).applyQuaternion(this.current); this.camera.lookAt(this.controls.target);
  }
}

export function saveViewerPNG(renderer, scene, camera) {
  renderer.render(scene,camera);
  renderer.domElement.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='edgerf-3d-view.png';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
}

export function installViewerTools(getScene, getSync) {
  const note=document.createElement('div');note.className='viewer-data-note';note.textContent='Illustrative room mapping · positions rescaled to layout · not calibrated to metres';document.body.append(note);
  const toolbar=document.createElement('div');toolbar.className='viewer-tools';
  const recenter=document.createElement('button');recenter.textContent='Recenter IMU';recenter.title='Use current sensor orientation as neutral';
  recenter.onclick=()=>getSync()?.recenter();
  const capture=document.createElement('button');capture.textContent='Save view ↓';
  capture.onclick=()=>{const {renderer,scene,camera}=getScene();saveViewerPNG(renderer,scene,camera);};
  toolbar.append(recenter,capture);document.body.append(toolbar);
  document.addEventListener('keydown',event=>{if(event.key.toLowerCase()==='r' && !event.ctrlKey && !event.metaKey && !['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))getSync()?.recenter();});
  window.addEventListener('pagehide',()=>getSync()?.setEnabled(false),{once:true});
}
