'use strict';
const $ = id => document.getElementById(id);
let state, samples = [], history = [], lastScan = 0, lastRevision = -1, rotating = false, angle = -.55;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let toastTimer;
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 5000); }
async function api(url, body) {
  const response = await fetch(url, {method: body === undefined ? 'GET' : 'POST', headers: body === undefined ? {} : {'Content-Type':'application/json'}, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(url.includes('/scan/') ? 130000 : 15000)});
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function band(f) { return f >= 5925 ? '6' : f >= 4900 ? '5' : '2.4'; }
function quality(dbm) { return dbm >= -60 ? ['good','Strong','#6ee7bd'] : dbm >= -75 ? ['fair','Moderate','#f3be65'] : ['weak','Weak','#f08791']; }
function metric(id, value, unit='') { $(id).replaceChildren(document.createTextNode(value)); if (unit) { const em = document.createElement('em'); em.textContent = unit; $(id).append(em); } }
function renderNetworks() {
  if (!state) return;
  const query = $('search').value.toLowerCase();
  const networks = state.scan.networks.filter(n => `${n.ssid} ${n.bssid}`.toLowerCase().includes(query) && (!$('band').value || band(n.freq) === $('band').value)).sort((a,b) => b.dbm-a.dbm);
  const fragment = document.createDocumentFragment();
  for (const n of networks) {
    const row = document.createElement('tr');
    for (const value of [n.ssid || '<hidden>', n.bssid, `${band(n.freq)} GHz`]) { const cell = row.insertCell(); cell.textContent = value; }
    const [type,label,color] = quality(n.dbm), signal = row.insertCell(), wrap = document.createElement('div'); wrap.className = 'strength';
    const track = document.createElement('span'); track.className = 'strength-track'; const fill = document.createElement('i'); fill.style.width = `${Math.min(100,Math.max(0,(n.dbm+100)/70*100))}%`; fill.style.background = color; track.append(fill); wrap.append(track,document.createTextNode(`${n.dbm.toFixed(0)} dBm`)); signal.append(wrap);
    const chip = document.createElement('span'); chip.className = `quality ${type}`; chip.textContent = label; row.insertCell().append(chip); fragment.append(row);
  }
  if (!networks.length) { const row = document.createElement('tr'), cell = row.insertCell(); cell.colSpan = 5; cell.className = 'no-results'; cell.textContent = state.scan.stale ? 'No fresh scan data. Check the scanner diagnostics above.' : 'No networks match your filters.'; fragment.append(row); }
  $('networks').replaceChildren(fragment);
}
function render(data) {
  state = data;
  const nets = [...data.scan.networks].sort((a,b) => b.dbm-a.dbm), strongest = nets[0];
  $('connection').textContent = data.healthy ? 'Pipeline online' : 'Needs attention'; $('connection').classList.toggle('warning', !data.healthy);
  $('mode').textContent = data.mode === 'demo' ? 'DEMO · simulated' : 'Live hardware'; $('mode').classList.toggle('warning',data.mode === 'demo');
  metric('signal', strongest ? strongest.dbm.toFixed(0) : '—','dBm'); metric('net-count',nets.length); metric('samples',data.database.total_samples.toLocaleString()); metric('fps',data.fps.toFixed(1),'fps');
  $('strongest').textContent = (data.scan.stale ? 'Stale · ' : '') + (strongest?.ssid || 'No networks');
  $('storage').textContent = `${data.storage} · ${data.database.ssid_count} networks retained`;
  $('pose-status').textContent = `${data.pose.status} · ${Math.round(data.pose.confidence*100)}% confidence`;
  $('camera-health').textContent = data.camera_age_seconds !== null && data.camera_age_seconds < 3 ? 'Receiving frames' : 'Unavailable';
  $('scan-health').textContent = data.scan.stale ? 'Stale / unavailable' : 'Fresh data';
  $('scan-age').textContent = data.scan.age_seconds === null ? 'No successful scan' : `${Math.floor(data.scan.age_seconds)} seconds ago`;
  $('camera-state').textContent = data.camera_backend; $('units').textContent = data.coordinate_units;
  $('record-state').textContent = data.recording ? 'Recording' : 'Paused'; $('record').textContent = data.recording ? 'Ⅱ Pause recording' : '▶ Resume recording'; $('record').disabled = false;
  $('footer-status').textContent = `${data.mode === 'demo' ? 'Simulated data' : 'Edge device'} · ${Math.floor(data.uptime_seconds/60)}m uptime · ${data.coordinate_units}`;
  const notices = [];
  if (data.mode === 'demo') notices.push('Demo mode: camera, networks and motion are synthetic. Demo samples are stored separately.');
  if (data.requires_new_survey) notices.push('Your previous survey was recovered. Start a new survey before recording with a fresh camera origin.');
  if (data.scan.last_error) notices.push(data.scan.last_error);
  if (data.error) notices.push(data.error);
  if (!Object.values(data.workers).every(Boolean)) notices.push('A video pipeline worker is not running. Check the server log.');
  $('notice').textContent = notices.join(' '); $('notice').hidden = !notices.length;
  if (data.scan.last_scan_epoch !== lastScan && strongest && !data.scan.stale) {
    lastScan = data.scan.last_scan_epoch; history.push(strongest.dbm); history = history.slice(-40);
    $('trend').querySelector('path').setAttribute('d', history.map((v,i) => `${i?'L':'M'}${i/Math.max(1,history.length-1)*180},${32-Math.max(0,Math.min(1,(v+100)/70))*29}`).join(' '));
  }
  const selected = $('ssid').value;
  $('ssid').replaceChildren(new Option('All networks',''),...data.database.ssids.map(s => new Option(s,s)));
  if (data.database.ssids.includes(selected)) $('ssid').value = selected;
  renderNetworks();
}
async function poll() {
  try {
    if (!document.hidden) {
      const data = await api('/api/overview'); render(data);
      if (data.revision !== lastRevision) { const result = await api('/api/rf/samples?max=3000'); samples = result.samples; lastRevision = data.revision; }
    }
  } catch (error) { $('connection').textContent = 'Reconnecting…'; $('connection').classList.add('warning'); $('notice').textContent = `Connection interrupted: ${error.message}. Retrying automatically.`; $('notice').hidden = false; }
  finally { setTimeout(poll,2000); }
}
async function action(button, task) { button.disabled = true; try { await task(); } catch(error) { toast(error.message); } finally { button.disabled = false; } }
$('record').addEventListener('click',() => action($('record'),async () => { const data = await api('/api/recording',{enabled:!state.recording}); state.recording = data.recording; render(await api('/api/overview')); }));
$('scan').addEventListener('click',() => action($('scan'),async () => { $('scan').textContent = 'Scanning…'; try { await api('/api/scan/now',{}); render(await api('/api/overview')); toast('Scan complete'); } finally { $('scan').textContent = '↻ Scan now'; } }));
$('new-survey').addEventListener('click',() => $('new-dialog').showModal());
$('archives').addEventListener('click',() => action($('archives'),async () => {const data=await api('/api/sessions');$('archive-list').replaceChildren();for(const item of data.sessions){const link=document.createElement('a');link.href='/api/sessions/'+encodeURIComponent(item.name);link.download=item.name;link.textContent=new Date(item.modified*1000).toLocaleString()+' · '+Math.ceil(item.size_bytes/1024)+' KB ↓';$('archive-list').append(link);}if(!data.sessions.length)$('archive-list').textContent='No archived surveys yet.';$('archives-dialog').showModal();}));
$('new-dialog').addEventListener('close',() => { if ($('new-dialog').returnValue === 'confirm') action($('new-survey'),async () => { await api('/api/session/new',{}); samples=[]; lastRevision=-1; render(await api('/api/overview')); toast('New survey started. Previous samples archived.'); }); });
for (const id of ['search','band']) $(id).addEventListener('input',renderNetworks);
$('rotate').addEventListener('click',() => { rotating=!rotating; $('rotate').setAttribute('aria-pressed',String(rotating)); });
document.addEventListener('visibilitychange',() => { if (document.hidden) $('video').removeAttribute('src'); else $('video').src='/video'; });
const canvas = $('map'), ctx = canvas.getContext('2d'); let width=1,height=1,lastFrame=0;
new ResizeObserver(entries => { const box = entries[0].contentRect; width=box.width; height=box.height; const dpr=Math.min(2,devicePixelRatio||1); canvas.width=Math.round(width*dpr); canvas.height=Math.round(height*dpr); ctx.setTransform(dpr,0,0,dpr,0,0); }).observe(canvas);
function draw(now) {
  requestAnimationFrame(draw); if(document.hidden || now-lastFrame<33) return;
  const dt=Math.min(.1,(now-lastFrame)/1000); lastFrame=now;
  if(rotating && !reducedMotion.matches) angle+=dt*.12;
  ctx.clearRect(0,0,width,height);
  const points = samples.filter(s => !$('ssid').value || s.ssid === $('ssid').value);
  $('empty-map').hidden = points.length>0;
  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity,zmin=Infinity,zmax=-Infinity;
  for(const p of points){xmin=Math.min(xmin,p.x);xmax=Math.max(xmax,p.x);ymin=Math.min(ymin,p.y);ymax=Math.max(ymax,p.y);zmin=Math.min(zmin,p.z);zmax=Math.max(zmax,p.z);}
  const span=points.length?Math.max(xmax-xmin,ymax-ymin,zmax-zmin,1):6,scale=Math.min(width*.55,height*.78)/span;
  const cx=points.length?(xmin+xmax)/2:0,cy=points.length?(ymin+ymax)/2:0;
  function project(x,y,z=0){const a=(x-cx)*Math.cos(angle)-(y-cy)*Math.sin(angle),b=(x-cx)*Math.sin(angle)+(y-cy)*Math.cos(angle);return [width*.5+a*scale,height*.58+b*scale*.45-z*scale*.65];}
  ctx.strokeStyle='#223949';ctx.lineWidth=.7;
  for(let i=-4;i<=4;i++){for(const swap of [false,true]){const a=project(cx+(swap?-span/2:i*span/8),cy+(swap?i*span/8:-span/2)),b=project(cx+(swap?span/2:i*span/8),cy+(swap?i*span/8:span/2));ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}}
  // Height uses measured Z; signal strength changes colour, never geometry.
  const strongestAtPosition = new Map();
  for(const p of points){const key=`${p.x},${p.y},${p.z}`;if(!strongestAtPosition.has(key)||p.dbm>strongestAtPosition.get(key).dbm)strongestAtPosition.set(key,p);}
  for(const p of strongestAtPosition.values()){const z=(p.z-zmin),base=project(p.x,p.y),pos=project(p.x,p.y,z),color=quality(p.dbm)[2];ctx.globalAlpha=.2;ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(...base);ctx.lineTo(...pos);ctx.stroke();ctx.globalAlpha=.9;ctx.fillStyle=color;ctx.beginPath();ctx.arc(...pos,3,0,Math.PI*2);ctx.fill();}
  ctx.globalAlpha=1;
}
requestAnimationFrame(draw);poll();
