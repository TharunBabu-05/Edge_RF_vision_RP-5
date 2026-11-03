
// ── Three.js imports ────────────────────────────────────────────────
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Globals ────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let layers = { walls: new THREE.Group(), furniture: new THREE.Group(),
               rf: new THREE.Group(), path: new THREE.Group(),
               cameras: new THREE.Group(), grid: new THREE.Group() };
let currentGeom = null;
let ssidFilter = '';
let frameCount = 0, lastFPSTime = performance.now(), fps = 0;

const DBM_MIN = -90, DBM_MAX = -30;

// ── Signal → colour (matches Ekahau convention) ────────────────────
function signalColor(dbm) {
  const t = Math.max(0, Math.min(1, (dbm - DBM_MIN) / (DBM_MAX - DBM_MIN)));
  // Blue (weak) → Cyan → Green → Yellow → Red (strong)
  const stops = [
    [0.00, 0x1a237e], [0.25, 0x0288d1], [0.50, 0x00897b],
    [0.75, 0xfdd835], [1.00, 0xf44336],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0);
      const r = ((c0 >> 16 & 0xff) * (1 - f) + (c1 >> 16 & 0xff) * f) | 0;
      const g = ((c0 >>  8 & 0xff) * (1 - f) + (c1 >>  8 & 0xff) * f) | 0;
      const b = ((c0       & 0xff) * (1 - f) + (c1       & 0xff) * f) | 0;
      return (r << 16) | (g << 8) | b;
    }
  }
  return 0x1a237e;
}

// ── Init Three.js ──────────────────────────────────────────────────
function initThree() {
  const canvas = document.getElementById('three-canvas');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0c10);
  scene.fog = new THREE.Fog(0x0a0c10, 20, 60);

  const vp = document.getElementById('viewport');
  const w = vp.clientWidth, h = vp.clientHeight;
  camera = new THREE.PerspectiveCamera(55, w / h, 0.05, 100);
  camera.position.set(6, 5, 8);

  renderer.setSize(w, h);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 0.5;
  controls.maxDistance = 40;
  controls.maxPolarAngle = Math.PI * 0.9;

  // Add all layer groups to scene
  for (const g of Object.values(layers)) scene.add(g);

  // Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambient);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(5, 10, 5);
  dirLight.castShadow = true;
  scene.add(dirLight);
  const fillLight = new THREE.DirectionalLight(0x8ecae6, 0.3);
  fillLight.position.set(-5, 3, -5);
  scene.add(fillLight);

  window.addEventListener('resize', onResize);
  animate();
}

function onResize() {
  const vp = document.getElementById('viewport');
  const w = vp.clientWidth, h = vp.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

let gyroSyncEnabled = false;
let targetYawRad = 0;
let currentYawRad = 0;

window.toggleGyroSync = function() {
  gyroSyncEnabled = !gyroSyncEnabled;
  const label = gyroSyncEnabled ? '🧭 Gyro Sync: ON' : '🧭 Gyro Sync: OFF';
  const btn = document.getElementById('btn-scene-gyro');
  const hudVal = document.getElementById('hud-gyro-val');
  if (btn) btn.textContent = label;
  if (hudVal) {
    hudVal.textContent = gyroSyncEnabled ? 'ON' : 'OFF';
    hudVal.style.color = gyroSyncEnabled ? 'var(--green)' : 'var(--text)';
  }
};

function pollIMU() {
  if (!gyroSyncEnabled) return;
  fetch('/api/imu/orientation')
    .then(r => r.json())
    .then(data => {
      if (data.ok) {
        targetYawRad = data.yaw_rad;
      }
    })
    .catch(() => {});
}
setInterval(pollIMU, 80);

function animate() {
  requestAnimationFrame(animate);

  if (gyroSyncEnabled) {
    // Smooth lerp toward target yaw
    let diff = targetYawRad - currentYawRad;
    while (diff < -Math.PI) diff += Math.PI * 2;
    while (diff > Math.PI)  diff -= Math.PI * 2;
    currentYawRad += diff * 0.12;

    // Reposition camera using THREE.Spherical — the correct way
    // to programmatically set OrbitControls azimuth angle
    const offset = camera.position.clone().sub(controls.target);
    const spherical = new THREE.Spherical().setFromVector3(offset);
    spherical.theta = currentYawRad;
    offset.setFromSpherical(spherical);
    camera.position.copy(controls.target).add(offset);
    camera.lookAt(controls.target);
  }

  controls.update();
  renderer.render(scene, camera);

  frameCount++;
  const now = performance.now();
  if (now - lastFPSTime >= 1000) {
    fps = Math.round(frameCount * 1000 / (now - lastFPSTime));
    frameCount = 0; lastFPSTime = now;
    document.getElementById('hud-fps').textContent = fps;
  }
}

// ── Build scene from geometry JSON ────────────────────────────────
function buildScene(geom) {
  currentGeom = geom;
  // Clear all layers
  for (const g of Object.values(layers)) { while (g.children.length) g.remove(g.children[0]); }

  buildGrid(geom);
  buildWalls(geom);
  buildFurniture(geom);
  buildRFSamples(geom);
  buildWalkPath(geom);
  buildCameraMarkers(geom);
  updateStats(geom);
  updateSSIDFilter(geom);
  updateFurnitureLegend(geom);

  // Auto-fit camera
  const rw = geom.room_width || 5, rd = geom.room_depth || 5, rh = geom.room_height || 2.5;
  const centre = new THREE.Vector3(rw / 2, rh / 2, rd / 2);
  controls.target.copy(centre);
  const dist = Math.max(rw, rd) * 1.3;
  camera.position.set(centre.x + dist * 0.7, centre.y + dist * 0.8, centre.z + dist * 1.1);
  camera.lookAt(centre);
  controls.update();
}

// ── Grid floor ─────────────────────────────────────────────────────
function buildGrid(geom) {
  const rw = geom.room_width || 5, rd = geom.room_depth || 5;
  const size = Math.max(rw, rd) * 1.5;
  const grid = new THREE.GridHelper(size, Math.ceil(size / 0.5), 0x1a2030, 0x1a2030);
  grid.position.set(rw / 2, -0.001, rd / 2);
  layers.grid.add(grid);
}

// ── Room walls (real geometry from RANSAC planes) ──────────────────
function buildWalls(geom) {
  const walls = geom.walls || [];
  const rw = geom.room_width || 5, rd = geom.room_depth || 5, rh = geom.room_height || 2.5;

  // Always create the main outer bounding cuboid box with cyan wireframe & direction labels
  buildCuboidRoom(rw, rd, rh);

  // Render any secondary/partition walls if present
  walls.forEach(wall => {
    if (wall.role === 'floor' || wall.role === 'ceiling') return; // Handled by cuboid

    const corners = wall.corners;
    if (!corners || corners.length < 3) return;

    const color = parseInt((wall.color || '#3A4F6B').replace('#', ''), 16);
    const mat = new THREE.MeshPhongMaterial({
      color, transparent: true, opacity: 0.35,
      side: THREE.DoubleSide, shininess: 10,
    });

    const verts = corners.map(c => new THREE.Vector3(c[0], c[1], c[2]));

    const geo = new THREE.BufferGeometry();
    const positions = [];
    for (let i = 1; i < verts.length - 1; i++) {
      positions.push(...verts[0].toArray(), ...verts[i].toArray(), ...verts[i + 1].toArray());
    }
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, mat);
    layers.walls.add(mesh);

    const edgeGeo = new THREE.BufferGeometry().setFromPoints([...verts, verts[0]]);
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x00d4ff, opacity: 0.6, transparent: true });
    layers.walls.add(new THREE.Line(edgeGeo, edgeMat));
  });
}

function buildCuboidRoom(w, d, h) {
  // ── Floor (solid, wood-like) ───────────────────────────────────────
  const floorGeo = new THREE.PlaneGeometry(w, d);
  const floorMat = new THREE.MeshLambertMaterial({ color: 0xC4A875, side: THREE.DoubleSide });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(w / 2, 0, d / 2);
  floor.receiveShadow = true;
  layers.walls.add(floor);

  // ── 4 Semi-transparent walls (each direction subtly different tint) ──
  const wallDefs = [
    { label: 'N', color: 0x1e3a5f, verts: [[0,0,0],[w,0,0],[w,h,0],[0,h,0]] },
    { label: 'S', color: 0x1e3a5f, verts: [[0,0,d],[w,0,d],[w,h,d],[0,h,d]] },
    { label: 'W', color: 0x2a3f5f, verts: [[0,0,0],[0,0,d],[0,h,d],[0,h,0]] },
    { label: 'E', color: 0x2a3f5f, verts: [[w,0,0],[w,0,d],[w,h,d],[w,h,0]] },
  ];

  wallDefs.forEach(({ label, color, verts }) => {
    const v = verts.map(p => new THREE.Vector3(...p));
    const geo = new THREE.BufferGeometry();
    const pos = [
      ...v[0].toArray(), ...v[1].toArray(), ...v[2].toArray(),
      ...v[0].toArray(), ...v[2].toArray(), ...v[3].toArray(),
    ];
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.computeVertexNormals();
    const mat = new THREE.MeshPhongMaterial({
      color, transparent: true, opacity: 0.18,
      side: THREE.DoubleSide, shininess: 0,
    });
    layers.walls.add(new THREE.Mesh(geo, mat));

    // Wall label in the centre of each wall
    const cx = (v[0].x + v[1].x + v[2].x + v[3].x) / 4;
    const cy = h * 0.55;
    const cz = (v[0].z + v[1].z + v[2].z + v[3].z) / 4;
    const sp = makeTextSprite(label, { fontSize: 64, color: 'rgba(100,200,255,0.55)' });
    sp.position.set(cx, cy, cz);
    sp.scale.set(1.2, 1.2, 1);
    layers.walls.add(sp);
  });

  // ── Glowing wireframe edges (the main cuboid outline) ────────────
  const boxGeo = new THREE.BoxGeometry(w, h, d);
  const edgesGeo = new THREE.EdgesGeometry(boxGeo);
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x00d4ff, linewidth: 2 });
  const wireBox = new THREE.LineSegments(edgesGeo, edgeMat);
  wireBox.position.set(w / 2, h / 2, d / 2);
  layers.walls.add(wireBox);

  // ── Ceiling (very faint, just edges) ─────────────────────────────
  const ceilGeo = new THREE.PlaneGeometry(w, d);
  const ceilMat = new THREE.MeshLambertMaterial({
    color: 0xffffff, transparent: true, opacity: 0.06, side: THREE.DoubleSide,
  });
  const ceil = new THREE.Mesh(ceilGeo, ceilMat);
  ceil.rotation.x = -Math.PI / 2;
  ceil.position.set(w / 2, h, d / 2);
  layers.walls.add(ceil);

  // ── Corner pillars (accent lines) ────────────────────────────────
  const corners = [[0,0],[w,0],[w,d],[0,d]];
  corners.forEach(([cx, cz]) => {
    const pts = [new THREE.Vector3(cx, 0, cz), new THREE.Vector3(cx, h, cz)];
    const pillarGeo = new THREE.BufferGeometry().setFromPoints(pts);
    const pillarMat = new THREE.LineBasicMaterial({ color: 0x00d4ff, opacity: 0.6, transparent: true });
    layers.walls.add(new THREE.Line(pillarGeo, pillarMat));
  });
}

// ── Furniture 3D boxes ─────────────────────────────────────────────
function buildFurniture(geom) {
  const items = geom.furniture || [];
  items.forEach(f => {
    const color = parseInt((f.color || '#95A5A6').replace('#', ''), 16);

    const bw = Math.max(f.width      || 0.5, 0.15);
    const bh = Math.max(f.height     || 0.5, 0.15);
    const bd = Math.max(f.depth_size || 0.4, 0.15);

    // Solid box
    const geo = new THREE.BoxGeometry(bw, bh, bd);
    const mat = new THREE.MeshPhongMaterial({ color, transparent: true, opacity: 0.7, shininess: 40 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(f.cx || 0, (f.cz || 0) + bh / 2, f.cy || 0);
    mesh.castShadow = true;
    layers.furniture.add(mesh);

    // Bright edge outline
    const edgeLines = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.9, transparent: true })
    );
    edgeLines.position.copy(mesh.position);
    layers.furniture.add(edgeLines);

    // Label
    const label = makeTextSprite(`${f.label}\n${Math.round((f.confidence || 0) * 100)}%`);
    label.position.set(f.cx || 0, (f.cz || 0) + bh + 0.25, f.cy || 0);
    layers.furniture.add(label);
  });
}

// ── RF sample spheres & interpolated heatmap ───────────────────────
function buildRFSamples(geom) {
  const samples = geom.rf_samples || [];
  const filtered = ssidFilter ? samples.filter(s => s.ssid === ssidFilter) : samples;

  const rw = geom.room_width  || 5;
  const rd = geom.room_depth  || 5;
  const rh = geom.room_height || 2.5;

  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  filtered.forEach(s => {
    const rx = parseFloat(s.x ?? 0);
    const rz = parseFloat(s.y ?? s.z ?? 0);
    if (rx < minX) minX = rx;
    if (rx > maxX) maxX = rx;
    if (rz < minZ) minZ = rz;
    if (rz > maxZ) maxZ = rz;
  });

  const spanX = (maxX > minX) ? (maxX - minX) : 1.0;
  const spanZ = (maxZ > minZ) ? (maxZ - minZ) : 1.0;

  // Map every sample coordinate strictly INSIDE the cuboid room volume
  const mappedSamples = filtered.map(s => {
    const dbm = parseFloat(s.dbm ?? s.d ?? -90);
    const t = Math.max(0, Math.min(1, (dbm - DBM_MIN) / (DBM_MAX - DBM_MIN)));

    const rx = parseFloat(s.x ?? 0);
    const rz = parseFloat(s.y ?? s.z ?? 0);

    // Map X into [0.4 .. rw - 0.4]
    let px = (maxX > minX) ? (0.4 + ((rx - minX) / spanX) * (rw - 0.8)) : (rw / 2);
    px = Math.max(0.3, Math.min(rw - 0.3, px));

    // Map Z into [0.4 .. rd - 0.4]
    let pz = (maxZ > minZ) ? (0.4 + ((rz - minZ) / spanZ) * (rd - 0.8)) : (rd / 2);
    pz = Math.max(0.3, Math.min(rd - 0.3, pz));

    // Map Y (Height) strictly INSIDE ceiling height [0.2 .. rh - 0.25]
    let py = 0.2 + t * (rh - 0.45);
    py = Math.max(0.15, Math.min(rh - 0.25, py));

    return { ...s, dbm, t, px, py, pz };
  });

  let bestDBM = -999;

  // Spheres & Vertical Stems inside room cuboid
  if (viewMode === 'spheres' || viewMode === 'both' || viewMode === 'points') {
    mappedSamples.forEach(s => {
      if (s.dbm > bestDBM) bestDBM = s.dbm;

      const col = signalColor(s.dbm);
      const radius = sphereSize * (0.5 + s.t * 0.6);

      const geo = new THREE.SphereGeometry(radius, 10, 10);
      const mat = new THREE.MeshPhongMaterial({ color: col, emissive: col, emissiveIntensity: 0.35 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(s.px, s.py, s.pz);
      mesh.userData = { ssid: s.ssid, dbm: s.dbm };
      layers.rf.add(mesh);

      // Vertical stem line connecting sphere down to room floor
      const pts = [new THREE.Vector3(s.px, 0.01, s.pz), new THREE.Vector3(s.px, s.py, s.pz)];
      const stemGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const stemMat = new THREE.LineBasicMaterial({ color: col, opacity: 0.5, transparent: true });
      layers.rf.add(new THREE.Line(stemGeo, stemMat));
    });
  }

  // 3D Signal Surface Heatmap Plane on room floor
  if ((viewMode === 'heatmap' || viewMode === 'both') && mappedSamples.length >= 3) {
    const geo = new THREE.PlaneGeometry(rw - 0.2, rd - 0.2, heatmapRes, heatmapRes);
    geo.rotateX(-Math.PI / 2);
    geo.translate(rw / 2, 0.02, rd / 2);

    const pos = geo.attributes.position;
    const colors = new Float32Array(pos.count * 3);

    for (let i = 0; i < pos.count; i++) {
      const vx = pos.getX(i), vz = pos.getZ(i);
      let num = 0, den = 0;
      for (const s of mappedSamples) {
        const d2 = (vx - s.px)**2 + (vz - s.pz)**2 + 1e-4;
        const w = 1.0 / (d2 ** 1.2);
        num += w * s.dbm;
        den += w;
      }
      const dbmInterp = den > 0 ? num / den : -90;
      const col = new THREE.Color(signalColor(dbmInterp));
      colors[i * 3]     = col.r;
      colors[i * 3 + 1] = col.g;
      colors[i * 3 + 2] = col.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.MeshLambertMaterial({
      vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.75,
    });
    const heatMesh = new THREE.Mesh(geo, mat);
    layers.rf.add(heatMesh);
  }

  if (mappedSamples.length > 0 && bestDBM === -999) {
    bestDBM = Math.max(...mappedSamples.map(s => s.dbm));
  }

  document.getElementById('hud-samples').textContent = mappedSamples.length;
  document.getElementById('hud-dbm').textContent = bestDBM > -990 ? `${bestDBM.toFixed(0)} dBm` : '—';
  document.getElementById('stat-rf').textContent = mappedSamples.length;
  document.getElementById('stat-dbm').textContent = bestDBM > -990 ? `${bestDBM.toFixed(0)}` : '—';
}

// ── Hover Tooltip Raycasting ───────────────────────────────────────
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

window.addEventListener('mousemove', e => {
  const canvas = document.getElementById('three-canvas');
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return;

  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(layers.rf.children);
  if (hits.length > 0 && hits[0].object.userData.dbm !== undefined) {
    const data = hits[0].object.userData;
    showBanner(`📡 ${data.ssid || 'Network'}: ${data.dbm.toFixed(0)} dBm`, 'info', 1500);
  }
});

// ── Walk path line (inside cuboid) ──────────────────────────────────
function buildWalkPath(geom) {
  const path = geom.walk_path || [];
  if (path.length < 2) return;
  const rw = geom.room_width || 5, rd = geom.room_depth || 5;

  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  path.forEach(p => {
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] < minZ) minZ = p[1];
    if (p[1] > maxZ) maxZ = p[1];
  });
  const spanX = (maxX > minX) ? (maxX - minX) : 1.0;
  const spanZ = (maxZ > minZ) ? (maxZ - minZ) : 1.0;

  const points = path.map(p => {
    let px = (maxX > minX) ? (0.4 + ((p[0] - minX) / spanX) * (rw - 0.8)) : (rw / 2);
    let pz = (maxZ > minZ) ? (0.4 + ((p[1] - minZ) / spanZ) * (rd - 0.8)) : (rd / 2);
    return new THREE.Vector3(
      Math.max(0.3, Math.min(rw - 0.3, px)),
      0.05,
      Math.max(0.3, Math.min(rd - 0.3, pz))
    );
  });

  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineBasicMaterial({ color: 0x00d4ff, opacity: 0.8, transparent: true });
  layers.path.add(new THREE.Line(geo, mat));
}






// ── Camera position markers (N/E/S/W shots) ────────────────────────
function buildCameraMarkers(geom) {
  const dirs = geom.capture_directions || [];
  const rw = geom.room_width || 5, rd = geom.room_depth || 5;
  const cx = rw / 2, cz = rd / 2, y = 0.05;

  const positions = { N: [cx, y, 0.3], E: [rw - 0.3, y, cz], S: [cx, y, rd - 0.3], W: [0.3, y, cz] };
  const colors   = { N: 0x00d4ff, E: 0x7c3aed, S: 0xf59e0b, W: 0x22c55e };

  dirs.forEach(dir => {
    if (!positions[dir]) return;
    const pos = positions[dir];
    const col = colors[dir] || 0xffffff;

    // Camera icon (cone)
    const geo = new THREE.ConeGeometry(0.12, 0.25, 6);
    const mat = new THREE.MeshPhongMaterial({ color: col, emissive: col, emissiveIntensity: 0.4 });
    const cone = new THREE.Mesh(geo, mat);
    cone.position.set(...pos);
    cone.rotation.z = Math.PI;
    layers.cameras.add(cone);

    // Label
    const label = makeTextSprite(dir, 14, `#${col.toString(16).padStart(6, '0')}`);
    label.position.set(pos[0], pos[1] + 0.4, pos[2]);
    layers.cameras.add(label);
  });
}

// ── Text sprite helper ─────────────────────────────────────────────
function makeTextSprite(text, size = 12, color = '#ffffff') {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.roundRect(4, 4, canvas.width - 8, canvas.height - 8, 8);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.font = `bold ${size}px Inter, sans-serif`;
  ctx.textAlign = 'center';
  const lines = text.split('\n');
  lines.forEach((line, i) => ctx.fillText(line, canvas.width / 2, 44 + i * 28));
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.7, 0.35, 1);
  return sprite;
}

// ── Stats & UI updates ─────────────────────────────────────────────
function updateStats(geom) {
  document.getElementById('stat-w').textContent = (geom.room_width || 0).toFixed(1);
  document.getElementById('stat-d').textContent = (geom.room_depth || 0).toFixed(1);
  document.getElementById('stat-h').textContent = (geom.room_height || 0).toFixed(1);
  document.getElementById('stat-furn').textContent = (geom.furniture || []).length;
  document.getElementById('hud-room').textContent =
    `${(geom.room_width||0).toFixed(1)}×${(geom.room_depth||0).toFixed(1)}×${(geom.room_height||0).toFixed(1)} m`;
}

function updateSSIDFilter(geom) {
  const ssids = [...new Set((geom.rf_samples || []).map(s => s.ssid).filter(Boolean))];
  const sel = document.getElementById('ssid-filter');
  const prev = sel.value;
  sel.innerHTML = '<option value="">All Networks</option>' +
    ssids.map(s => `<option value="${s}"${s === prev ? ' selected' : ''}>${s}</option>`).join('');
}

function updateFurnitureLegend(geom) {
  const items = geom.furniture || [];
  const byLabel = {};
  items.forEach(f => { byLabel[f.label] = f.color; });
  const legend = document.getElementById('furniture-legend');
  legend.innerHTML = Object.entries(byLabel).map(([label, color]) =>
    `<div class="legend-item"><div class="legend-dot" style="background:${color}"></div>${label}</div>`
  ).join('') || '<p style="font-size:11px;color:var(--muted)">No furniture detected</p>';
}

// ── Layer toggle ───────────────────────────────────────────────────
window.toggleLayer = function(name) {
  if (layers[name]) layers[name].visible = !layers[name].visible;
};

// ── SSID filter ────────────────────────────────────────────────────
window.applySSIDFilter = function() {
  ssidFilter = document.getElementById('ssid-filter').value;
  if (currentGeom) {
    while (layers.rf.children.length) layers.rf.remove(layers.rf.children[0]);
    buildRFSamples(currentGeom);
  }
};

// ── Camera reset ───────────────────────────────────────────────────
window.resetCamera = function() {
  if (!currentGeom) return;
  const rw = currentGeom.room_width || 5, rd = currentGeom.room_depth || 5, rh = currentGeom.room_height || 2.5;
  const centre = new THREE.Vector3(rw / 2, rh / 2, rd / 2);
  controls.target.copy(centre);
  const dist = Math.max(rw, rd) * 1.3;
  camera.position.set(centre.x + dist * 0.7, centre.y + dist * 0.8, centre.z + dist * 1.1);
  controls.update();
};

// ── Status banner ──────────────────────────────────────────────────
function showBanner(msg, type = 'info', duration = 0) {
  const el = document.getElementById('status-banner');
  el.className = `show ${type}`;
  document.getElementById('status-text').textContent = msg;
  if (duration > 0) setTimeout(() => el.classList.remove('show'), duration);
}

// ── Load scene from API ────────────────────────────────────────────
window.loadScene = async function() {
  showBanner('Loading scene geometry…', 'info');
  document.getElementById('loading-overlay').classList.remove('hidden');
  document.getElementById('loading-text').textContent = 'Fetching room geometry…';
  try {
    const res = await fetch('/api/scene/geometry');
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showBanner(err.error || 'No scene geometry yet. Capture 4 shots and reconstruct.', 'error', 6000);
      document.getElementById('loading-overlay').classList.add('hidden');
      return;
    }
    const geom = await res.json();
    buildScene(geom);
    showBanner(`Scene loaded — ${(geom.furniture||[]).length} furniture items`, 'success', 4000);
  } catch (e) {
    showBanner('Failed to load scene: ' + e.message, 'error', 5000);
  } finally {
    document.getElementById('loading-overlay').classList.add('hidden');
  }
};

// ── Capture shots ──────────────────────────────────────────────────
window.takeShot = async function(dir) {
  showBanner(`Capturing ${dir} shot…`, 'info');
  try {
    const res = await fetch(`/api/scene/shot/${dir}`, { method: 'POST' });
    const data = await res.json();
    if (!data.ok) { showBanner(`Error: ${data.error}`, 'error', 4000); return; }

    // Mark button captured
    document.getElementById(`shot-${dir}`).classList.add('captured');
    document.getElementById(`status-${dir}`).textContent = '✓ Captured';

    // Load preview thumbnail
    const img = new Image();
    img.src = `/api/scene/preview/${dir}?t=${Date.now()}`;
    img.onload = () => {
      const btn = document.getElementById(`shot-${dir}`);
      const existing = btn.querySelector('img');
      if (existing) btn.removeChild(existing);
      img.style.cssText = 'width:100%;height:100%;object-fit:cover;position:absolute;inset:0;opacity:0.7;border-radius:8px;';
      btn.insertBefore(img, btn.firstChild);
    };

    updateProgress(data);
    if (data.shots_remaining && data.shots_remaining.length === 0) {
      document.getElementById('btn-reconstruct').disabled = false;
      showBanner('All 4 shots captured! Click "Reconstruct Room"', 'success', 0);
    } else {
      showBanner(`${dir} shot saved. Next: ${data.shots_remaining?.join(', ') || ''}`, 'success', 3000);
    }
  } catch (e) {
    showBanner('Shot failed: ' + e.message, 'error', 4000);
  }
};

// ── Trigger reconstruction ─────────────────────────────────────────
window.triggerReconstruct = async function() {
  showBanner('Starting room reconstruction (this takes ~30-60s on Pi 5)…', 'info');
  document.getElementById('loading-overlay').classList.remove('hidden');
  document.getElementById('loading-text').textContent = 'Running holistic 3D reconstruction…';
  try {
    const res = await fetch('/api/scene/reconstruct', { method: 'POST' });
    const data = await res.json();
    pollReconstructStatus();
  } catch (e) {
    showBanner('Reconstruction request failed: ' + e.message, 'error', 5000);
    document.getElementById('loading-overlay').classList.add('hidden');
  }
};

function pollReconstructStatus() {
  const poll = setInterval(async () => {
    try {
      const res = await fetch('/api/scene/status');
      const s = await res.json();
      updateProgress({ shots_taken: s.shots_taken, shots_remaining: s.shots_remaining, progress_pct: s.progress_pct });
      document.getElementById('loading-text').textContent =
        `Reconstructing… ${s.progress_pct.toFixed(0)}%`;

      if (s.state === 'ready') {
        clearInterval(poll);
        document.getElementById('loading-overlay').classList.add('hidden');
        showBanner(`Reconstruction complete! ${s.furniture_count} furniture items detected.`, 'success', 0);
        await loadScene();
      } else if (s.state === 'failed') {
        clearInterval(poll);
        document.getElementById('loading-overlay').classList.add('hidden');
        showBanner(`Reconstruction failed: ${s.error}`, 'error', 8000);
      }
    } catch (e) { /* network glitch, keep polling */ }
  }, 2000);
}

// ── Reset capture ──────────────────────────────────────────────────
window.resetCapture = async function() {
  await fetch('/api/scene/reset', { method: 'POST' });
  ['N','E','S','W'].forEach(d => {
    const btn = document.getElementById(`shot-${d}`);
    btn.classList.remove('captured');
    document.getElementById(`status-${d}`).textContent = 'Tap to capture';
    const img = btn.querySelector('img');
    if (img) btn.removeChild(img);
  });
  document.getElementById('btn-reconstruct').disabled = true;
  updateProgress({ shots_taken: [], shots_remaining: ['N','E','S','W'], progress_pct: 0 });

  // Clear 3D canvas layers so screen returns to blank
  currentGeom = null;
  for (const g of Object.values(layers)) { while (g.children.length) g.remove(g.children[0]); }

  showBanner('Session reset', 'info', 2000);
};

// ── Progress UI ────────────────────────────────────────────────────
function updateProgress(data) {
  const taken = data.shots_taken?.length || 0;
  const total = (data.shots_taken?.length || 0) + (data.shots_remaining?.length || 0);
  const pct = data.progress_pct ?? (total > 0 ? (taken / total * 60) : 0);
  document.getElementById('progress-bar-inner').style.width = `${pct}%`;
  document.getElementById('progress-label').textContent =
    total > 0 ? `${taken} / ${total} shots captured` : 'Ready';
}

// ── Auto-refresh RF samples ────────────────────────────────────────
async function refreshRFSamples() {
  if (!currentGeom) return;
  try {
    const res = await fetch('/api/scene/geometry');
    if (!res.ok) return;
    const geom = await res.json();
    currentGeom.rf_samples = geom.rf_samples || [];
    while (layers.rf.children.length) layers.rf.remove(layers.rf.children[0]);
    buildRFSamples(currentGeom);
  } catch {}
}
setInterval(refreshRFSamples, 3000);

// ── Init ───────────────────────────────────────────────────────────
try {
  initThree();
} catch (err) {
  console.error('3D engine init failed:', err);
  const lt = document.getElementById('loading-text');
  if (lt) lt.textContent = 'Engine error: ' + err.message;
} finally {
  const ov = document.getElementById('loading-overlay');
  if (ov) ov.classList.add('hidden');
}

// Sync capture UI status on load (screen remains BLANK until "Reconstruct Room" is clicked)
(async () => {
  try {
    const res = await fetch('/api/scene/status');
    if (res.ok) {
      const s = await res.json();
      (s.shots_taken || []).forEach(d => {
        document.getElementById(`shot-${d}`)?.classList.add('captured');
        const st = document.getElementById(`status-${d}`);
        if (st) st.textContent = '✓ Captured';
      });
      if (s.shots_remaining && s.shots_remaining.length === 0) {
        document.getElementById('btn-reconstruct').disabled = false;
        showBanner('Click "Reconstruct Room" to generate 3D room structure.', 'info', 0);
      } else {
        showBanner('Stand in room center and capture 4 shots (N, E, S, W).', 'info', 0);
      }
    }
  } catch (err) {
    console.error('Status sync error:', err);
  }
})();
