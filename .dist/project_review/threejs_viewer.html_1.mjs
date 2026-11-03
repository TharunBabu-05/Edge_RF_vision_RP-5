
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    // ─────────────────────────────────────────────────────────────
    // Room layout — loaded from Pi on startup, falls back to defaults
    // ─────────────────────────────────────────────────────────────
    const ROOM_COLORS = [0x00c2ff, 0x7c3aed, 0x22d38a, 0xf59e0b, 0xef4444, 0xe879f9];

    let rooms = [];
    let nextId = 3;
    let selectedRoomId = null;
    let activeLayoutName = 'My Home';

    // ── Layout management helpers ──────────────────────────────────
    function showLayoutStatus(msg, ok = true) {
      const el = document.getElementById('layout-status');
      el.textContent = msg;
      el.style.color = ok ? 'var(--accent)' : '#ef4444';
      el.style.display = 'block';
      setTimeout(() => el.style.display = 'none', 3000);
    }

    function updateLayoutTitle() {
      const t = document.getElementById('layout-name-title');
      if (t) t.textContent = activeLayoutName;
    }

    async function populateLayoutSelector(selectName) {
      try {
        const res = await fetch('/api/rooms/layouts');
        const data = await res.json();
        if (!data.ok) return;
        const sel = document.getElementById('layout-selector');
        sel.innerHTML = '';
        const names = data.names || [];
        names.forEach(n => {
          const opt = document.createElement('option');
          opt.value = n;
          opt.textContent = n;
          if (n === (selectName || data.active)) opt.selected = true;
          sel.appendChild(opt);
        });
        document.getElementById('layout-count-badge').textContent = names.length;
        activeLayoutName = sel.value || data.active || 'My Home';
        updateLayoutTitle();
      } catch (e) {}
    }

    async function loadLayout(name) {
      try {
        const res = await fetch(`/api/rooms/layout?name=${encodeURIComponent(name)}`);
        const data = await res.json();
        if (data.ok && Array.isArray(data.rooms)) {
          rooms = data.rooms.map(r => ({
            ...r, color: ROOM_COLORS[(r.id || 0) % ROOM_COLORS.length]
          }));
          nextId = data.nextId || (Math.max(0, ...rooms.map(r => r.id)) + 1);
          activeLayoutName = data.name || name;
        }
      } catch (e) {
        rooms = [
          { id: 0, name: 'Living Room', x: 0, y: 0, w: 6, d: 5, h: 2.6, color: ROOM_COLORS[0] },
          { id: 1, name: 'Bedroom', x: 6, y: 0, w: 4, d: 5, h: 2.6, color: ROOM_COLORS[1] },
          { id: 2, name: 'Kitchen', x: 0, y: 5, w: 10, d: 3.5, h: 2.6, color: ROOM_COLORS[2] },
        ];
        nextId = 3;
        activeLayoutName = 'My Home';
      }
      updateLayoutTitle();
      selectedRoomId = null;
      activeRoomId = null;
      roomSamples = {};
      document.getElementById('room-editor').style.display = 'none';
      setActiveRoomIndicator(null);
      refreshActiveRoomSelector();
      renderRoomList();
      buildRooms();
    }

    // Persist current layout to Pi
    async function saveLayout(silent = false) {
      const saveData = rooms.map(r => ({
        id: r.id, name: r.name, x: r.x, y: r.y, w: r.w, d: r.d, h: r.h
      }));
      try {
        const res = await fetch('/api/rooms/layout', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: activeLayoutName, rooms: saveData, nextId })
        });
        const data = await res.json();
        if (!silent) {
          const el = document.getElementById('save-status');
          el.textContent = data.ok ? `✓ Saved ${data.saved} rooms to "${activeLayoutName}"` : `✗ ${data.error}`;
          el.style.color = data.ok ? 'var(--accent)' : '#ef4444';
          el.style.display = 'block';
          setTimeout(() => el.style.display = 'none', 3000);
        }
      } catch (e) {
        if (!silent) {
          const el = document.getElementById('save-status');
          el.textContent = '✗ Save failed (network error)';
          el.style.color = '#ef4444';
          el.style.display = 'block';
          setTimeout(() => el.style.display = 'none', 3000);
        }
      }
    }

    // ─────────────────────────────────────────────────────────────
    // RF data + per-room sample store
    // ─────────────────────────────────────────────────────────────
    let rfSamples = [];        // live samples from backend
    let activeSSID = '';
    let showPath = true;
    let showAP = true;
    let dotSize = 0.18;

    let roomSamples = {};      // { roomId: [frozen {x,y,z,dbm,ssid}, ...] }
    let roomIntervals = {};    // { roomId: [{start: timestamp, end: timestamp}, ...] }
    let activeRoomId = null;   // which room the user is physically standing in
    let activeRoomStartTime = null;
    let globalLatestTimestamp = 0; // Server time of last seen sample

    function refreshActiveRoomSelector() {
      const sel = document.getElementById('active-room-sel');
      const prev = sel.value;
      sel.innerHTML = '<option value="">— Not set —</option>';
      rooms.forEach(r => {
        const opt = document.createElement('option');
        opt.value = r.id;
        const count = (roomSamples[r.id] || []).length;
        opt.textContent = `${r.name}${count ? ` (${count} pts)` : ''}`;
        if (r.id == activeRoomId) opt.selected = true;
        sel.appendChild(opt);
      });
      if (prev && rooms.find(r => r.id == prev)) sel.value = prev;
    }

    function setActiveRoomIndicator(room) {
      const ind = document.getElementById('active-room-indicator');
      const lbl = document.getElementById('active-room-name-display');
      if (room) {
        const hex = '#' + new THREE.Color(room.color).getHexString();
        lbl.textContent = room.name;
        ind.style.display = 'block';
        ind.style.borderColor = hex + '80';
        ind.style.background = hex + '18';
        ind.style.color = hex;
      } else {
        ind.style.display = 'none';
      }
    }

    function showRoomDataStatus(msg, ok = true) {
      const el = document.getElementById('room-data-status');
      el.textContent = msg;
      el.style.color = ok ? 'var(--accent)' : '#ef4444';
      el.style.display = 'block';
      setTimeout(() => el.style.display = 'none', 3000);
    }

    // ─────────────────────────────────────────────────────────────
    // Three.js scene setup

    // ─────────────────────────────────────────────────────────────
    const canvas = document.getElementById('three-canvas');
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07080f);
    scene.fog = new THREE.FogExp2(0x07080f, 0.018);

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
    camera.position.set(10, 14, 16);
    camera.lookAt(5, 0, 4);

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 2;
    controls.maxDistance = 60;
    controls.maxPolarAngle = Math.PI / 2.05;

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambient);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(8, 20, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.set(2048, 2048);
    scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0x00c2ff, 0.2);
    fillLight.position.set(-8, 5, -8);
    scene.add(fillLight);

    // Floor plane (infinite grid feel)
    const gridHelper = new THREE.GridHelper(80, 80, 0x1a2f44, 0x111f2e);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);

    // ─────────────────────────────────────────────────────────────
    // Scene groups
    // ─────────────────────────────────────────────────────────────
    const roomGroup = new THREE.Group(); scene.add(roomGroup);
    const heatmapGroup = new THREE.Group(); scene.add(heatmapGroup);
    const rfGroup = new THREE.Group(); scene.add(rfGroup);
    const pathGroup = new THREE.Group(); scene.add(pathGroup);
    const apGroup = new THREE.Group(); scene.add(apGroup);

    // View mode: 'heatmap' | 'spheres' | 'both'
    let viewMode = 'heatmap';
    let heatmapRes = 30;  // IDW grid cells per room axis

    // ─────────────────────────────────────────────────────────────
    // dBm → Color  (red=weak → yellow → green=strong)
    // ─────────────────────────────────────────────────────────────
    function dbmToColor(dbm) {
      const t = Math.max(0, Math.min(1, (dbm - (-90)) / 60));
      const r = t < 0.5 ? 1 : 1 - (t - 0.5) * 2;
      const g = t < 0.5 ? t * 2 : 1;
      return new THREE.Color(r, g, 0);
    }

    // ─────────────────────────────────────────────────────────────
    // IDW interpolation (pure JS, no scipy)
    // Returns estimated dBm at point (qx, qz) given sample arrays.
    // ─────────────────────────────────────────────────────────────
    function idwInterp(qx, qz, sxArr, szArr, dbmArr, power = 2) {
      let wsum = 0, vsum = 0;
      for (let i = 0; i < sxArr.length; i++) {
        const dx = qx - sxArr[i], dz = qz - szArr[i];
        const d2 = dx * dx + dz * dz;
        if (d2 < 1e-10) return dbmArr[i]; // exact hit
        const w = 1 / Math.pow(d2, power / 2);
        wsum += w;
        vsum += w * dbmArr[i];
      }
      return wsum > 0 ? vsum / wsum : -90;
    }

    // ─────────────────────────────────────────────────────────────
    // Build floor heatmap meshes (one subdivided plane per room)
    // ─────────────────────────────────────────────────────────────
    function buildHeatmap(remappedSamples) {
      // Clear old heatmap meshes
      while (heatmapGroup.children.length) {
        const c = heatmapGroup.children[0];
        c.geometry?.dispose();
        if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
        else c.material?.dispose();
        heatmapGroup.remove(c);
      }

      if (!remappedSamples || remappedSamples.length < 2) return;

      // Precompute sample arrays (world X/Z and dBm)
      const sxArr = remappedSamples.map(s => s.wx);
      const szArr = remappedSamples.map(s => s.wz);
      const dbmArr = remappedSamples.map(s => s.dbm);

      rooms.forEach(room => {
        const segsX = heatmapRes;
        const segsZ = Math.max(4, Math.round(heatmapRes * (room.d / room.w)));

        const geo = new THREE.PlaneGeometry(room.w, room.d, segsX, segsZ);
        geo.rotateX(-Math.PI / 2); // lie flat

        // Shift vertices to room origin
        const pos = geo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
          pos.setX(i, pos.getX(i) + room.x + room.w / 2);
          pos.setZ(i, pos.getZ(i) + room.y + room.d / 2);
          pos.setY(i, 0.003); // just above floor
        }
        pos.needsUpdate = true;
        geo.computeBoundingBox();

        // Assign vertex colors via IDW
        const colors = new Float32Array(pos.count * 3);
        for (let i = 0; i < pos.count; i++) {
          const vx = pos.getX(i);
          const vz = pos.getZ(i);
          const dBm = idwInterp(vx, vz, sxArr, szArr, dbmArr, 2);
          const col = dbmToColor(dBm);
          colors[i * 3] = col.r;
          colors[i * 3 + 1] = col.g;
          colors[i * 3 + 2] = col.b;
        }
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const mat = new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 1,
          metalness: 0,
          transparent: true,
          opacity: 0.82,
          side: THREE.DoubleSide,
        });

        heatmapGroup.add(new THREE.Mesh(geo, mat));
      });
    }

    // ─────────────────────────────────────────────────────────────
    // Build room geometry
    // ─────────────────────────────────────────────────────────────
    function buildRooms() {
      while (roomGroup.children.length) {
        roomGroup.children[0].traverse(c => {
          if (c.geometry) c.geometry.dispose();
          if (c.material) c.material.dispose();
        });
        roomGroup.remove(roomGroup.children[0]);
      }

      rooms.forEach(room => {
        const g = new THREE.Group();
        const cx = room.x + room.w / 2;
        const cz = room.y + room.d / 2;
        const h = room.h;
        const col = new THREE.Color(room.color);

        // Floor
        const floorGeo = new THREE.PlaneGeometry(room.w, room.d);
        const floorMat = new THREE.MeshStandardMaterial({
          color: col, opacity: .12, transparent: true, side: THREE.DoubleSide,
          roughness: 0.8, metalness: 0
        });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.set(cx, 0.001, cz);
        floor.receiveShadow = true;
        g.add(floor);

        // Floor outline
        const floorEdge = new THREE.LineLoop(
          new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(room.x, 0.002, room.y),
            new THREE.Vector3(room.x + room.w, 0.002, room.y),
            new THREE.Vector3(room.x + room.w, 0.002, room.y + room.d),
            new THREE.Vector3(room.x, 0.002, room.y + room.d),
          ]),
          new THREE.LineBasicMaterial({ color: col, opacity: .6, transparent: true })
        );
        g.add(floorEdge);

        // Walls: 4 panels
        const wallMat = new THREE.MeshStandardMaterial({
          color: col, opacity: .08, transparent: true,
          side: THREE.DoubleSide, roughness: 1, metalness: 0
        });
        const wallEdgeMat = new THREE.LineBasicMaterial({ color: col, opacity: .5, transparent: true });

        const wallDefs = [
          // south wall
          {
            pts: [
              new THREE.Vector3(room.x, 0, room.y), new THREE.Vector3(room.x + room.w, 0, room.y),
              new THREE.Vector3(room.x + room.w, h, room.y), new THREE.Vector3(room.x, h, room.y)
            ]
          },
          // north wall
          {
            pts: [
              new THREE.Vector3(room.x, 0, room.y + room.d), new THREE.Vector3(room.x + room.w, 0, room.y + room.d),
              new THREE.Vector3(room.x + room.w, h, room.y + room.d), new THREE.Vector3(room.x, h, room.y + room.d)
            ]
          },
          // west wall
          {
            pts: [
              new THREE.Vector3(room.x, 0, room.y), new THREE.Vector3(room.x, 0, room.y + room.d),
              new THREE.Vector3(room.x, h, room.y + room.d), new THREE.Vector3(room.x, h, room.y)
            ]
          },
          // east wall
          {
            pts: [
              new THREE.Vector3(room.x + room.w, 0, room.y), new THREE.Vector3(room.x + room.w, 0, room.y + room.d),
              new THREE.Vector3(room.x + room.w, h, room.y + room.d), new THREE.Vector3(room.x + room.w, h, room.y)
            ]
          },
        ];

        wallDefs.forEach(wd => {
          // Mesh quad
          const geo = new THREE.BufferGeometry();
          const verts = new Float32Array(18); // 2 triangles
          const p = wd.pts;
          // tri 1
          verts[0] = p[0].x; verts[1] = p[0].y; verts[2] = p[0].z;
          verts[3] = p[1].x; verts[4] = p[1].y; verts[5] = p[1].z;
          verts[6] = p[2].x; verts[7] = p[2].y; verts[8] = p[2].z;
          // tri 2
          verts[9] = p[0].x; verts[10] = p[0].y; verts[11] = p[0].z;
          verts[12] = p[2].x; verts[13] = p[2].y; verts[14] = p[2].z;
          verts[15] = p[3].x; verts[16] = p[3].y; verts[17] = p[3].z;
          geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
          geo.computeVertexNormals();
          g.add(new THREE.Mesh(geo, wallMat.clone()));

          // Edge wire
          g.add(new THREE.LineLoop(
            new THREE.BufferGeometry().setFromPoints([...wd.pts, wd.pts[0]]),
            wallEdgeMat.clone()
          ));
        });

        // Top cap line (ceiling edge) — neon glow effect
        const capPts = [
          new THREE.Vector3(room.x, h, room.y),
          new THREE.Vector3(room.x + room.w, h, room.y),
          new THREE.Vector3(room.x + room.w, h, room.y + room.d),
          new THREE.Vector3(room.x, h, room.y + room.d),
        ];
        g.add(new THREE.LineLoop(
          new THREE.BufferGeometry().setFromPoints(capPts),
          new THREE.LineBasicMaterial({ color: col, opacity: .9, transparent: true })
        ));

        // Room label (sprite)
        const canvas2d = document.createElement('canvas');
        canvas2d.width = 256; canvas2d.height = 64;
        const ctx2d = canvas2d.getContext('2d');
        ctx2d.clearRect(0, 0, 256, 64);
        ctx2d.fillStyle = `#${col.getHexString()}`;
        ctx2d.font = 'bold 22px Inter, sans-serif';
        ctx2d.textAlign = 'center';
        ctx2d.textBaseline = 'middle';
        ctx2d.fillText(room.name, 128, 32);
        const tex = new THREE.CanvasTexture(canvas2d);
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: .85 }));
        sprite.scale.set(2.4, 0.6, 1);
        sprite.position.set(cx, h + 0.35, cz);
        g.add(sprite);

        roomGroup.add(g);
      });
    }

    // ─────────────────────────────────────────────────────────────
    // Build RF sample spheres + heatmap — per-room spatial clamping
    // ─────────────────────────────────────────────────────────────
    function buildRFSamples() {
      // Clear all groups
      [rfGroup, pathGroup, apGroup, heatmapGroup].forEach(g => {
        while (g.children.length) {
          const c = g.children[0];
          c.geometry?.dispose();
          if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
          else c.material?.dispose();
          g.remove(c);
        }
      });

      let totalSamples = 0;
      let bestDBM = -Infinity;

      // Render each room: frozen snapshot for non-active rooms, live for active room
      rooms.forEach(room => {
        const isActive = room.id == activeRoomId;
        const samples = isActive
          ? (activeSSID ? rfSamples.filter(s => s.ssid === activeSSID) : rfSamples)
          : (roomSamples[room.id] || []).filter(s => !activeSSID || s.ssid === activeSSID);

        if (!samples.length) return;
        totalSamples += samples.length;

        // Per-room bounding box (world space)
        const rx = room.x, ry = room.y, rw = room.w, rd = room.d, rh = room.h;
        const margin = 0.3;

        // Compute sample VO bounding box so we can normalize them into this room
        let sxMin = Infinity, sxMax = -Infinity;
        let syMin = Infinity, syMax = -Infinity;
        let szMin = Infinity, szMax = -Infinity;
        samples.forEach(s => {
          sxMin = Math.min(sxMin, s.x ?? 0); sxMax = Math.max(sxMax, s.x ?? 0);
          syMin = Math.min(syMin, s.y ?? 0); syMax = Math.max(syMax, s.y ?? 0);
          szMin = Math.min(szMin, s.z ?? 0); szMax = Math.max(szMax, s.z ?? 0);
        });
        const sxSpan = Math.max(sxMax - sxMin, 0.001);
        const sySpan = Math.max(syMax - syMin, 0.001);
        const szSpan = Math.max(szMax - szMin, 0.001);
        const hasRealZ = szSpan > 0.3;

        // Jitter helper seeded per-sample
        function jitter(s, range) {
          const seed = (s.bssid || s.ssid || '').split('').reduce((a, c) => a + c.charCodeAt(0), 0)
            + Math.round((s.x || 0) * 100) + Math.round((s.y || 0) * 100);
          const r = ((seed * 1664525 + 1013904223) & 0xffffffff) / 0xffffffff;
          return (r - 0.5) * range;
        }

        // Remap VO sample → inside this room's world box
        function remapIntoRoom(s) {
          const wx = rx + margin + ((s.x - sxMin) / sxSpan) * (rw - margin * 2);
          const wz = ry + margin + ((s.y - syMin) / sySpan) * (rd - margin * 2);
          let wy;
          if (hasRealZ) {
            wy = Math.min(Math.max(0.08, ((s.z - szMin) / szSpan) * rh * 0.85), rh * 0.9);
          } else {
            const t = Math.max(0, Math.min(1, ((s.dbm ?? -90) - (-90)) / 60));
            wy = Math.min(Math.max(0.05, 0.15 + t * rh * 0.72 + jitter(s, 0.35)), rh * 0.92);
          }
          // Hard clamp to room boundaries
          return {
            wx: Math.max(rx + 0.05, Math.min(rx + rw - 0.05, wx)),
            wy: Math.max(0.05, Math.min(rh - 0.05, wy)),
            wz: Math.max(ry + 0.05, Math.min(ry + rd - 0.05, wz))
          };
        }

        const opacity = isActive ? 1.0 : 0.45;   // frozen rooms are dimmed
        const sphGeo = new THREE.SphereGeometry(dotSize * (isActive ? 1 : 0.75), 10, 8);

        samples.forEach(s => {
          const { wx, wy, wz } = remapIntoRoom(s);
          const col = dbmToColor(s.dbm);
          if (s.dbm > bestDBM) bestDBM = s.dbm;

          const mat = new THREE.MeshStandardMaterial({
            color: col, roughness: 0.3, metalness: 0.2,
            emissive: col, emissiveIntensity: isActive ? 0.35 : 0.1,
            transparent: !isActive, opacity,
          });
          const mesh = new THREE.Mesh(sphGeo, mat);
          mesh.position.set(wx, wy, wz);
          mesh.castShadow = isActive;
          mesh.userData = { dbm: s.dbm, ssid: s.ssid };
          rfGroup.add(mesh);
        });

        // ── Walk path (only for active room or if no active room selected) ──
        if (showPath && samples.length >= 2 && isActive) {
          const pts = samples.map(s => {
            const { wx, wz } = remapIntoRoom(s);
            return new THREE.Vector3(wx, 0.05, wz);
          });
          pathGroup.add(new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: new THREE.Color(room.color), opacity: 0.7, transparent: true }),
          ));
          // Start/end markers
          const mkGeo = new THREE.SphereGeometry(0.14, 8, 6);
          const mkS = new THREE.Mesh(mkGeo, new THREE.MeshStandardMaterial({ color: 0x22d38a, emissive: 0x22d38a, emissiveIntensity: 0.7 }));
          mkS.position.copy(pts[0]);
          pathGroup.add(mkS);
          const mkE = new THREE.Mesh(mkGeo, new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xef4444, emissiveIntensity: 0.7 }));
          mkE.position.copy(pts[pts.length - 1]);
          pathGroup.add(mkE);
        }

        // ── AP pins (only for active room) ────────────────────────
        if (showAP && isActive) {
          const bySSID = {};
          samples.forEach(s => { if (!bySSID[s.ssid] || s.dbm > bySSID[s.ssid].dbm) bySSID[s.ssid] = s; });
          Object.values(bySSID).forEach(ap => {
            const { wx, wz } = remapIntoRoom(ap);
            const apH = 1.5;
            pathGroup.add(new THREE.Line(
              new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(wx, 0, wz), new THREE.Vector3(wx, apH, wz)]),
              new THREE.LineBasicMaterial({ color: 0x00c2ff }),
            ));
            const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.18, 12, 10),
              new THREE.MeshStandardMaterial({ color: 0x00c2ff, emissive: 0x00c2ff, emissiveIntensity: 0.8, roughness: 0.3 }));
            bulb.position.set(wx, apH, wz);
            apGroup.add(bulb);
            // Label
            const c2 = document.createElement('canvas');
            c2.width = 256; c2.height = 64;
            const ct = c2.getContext('2d');
            ct.fillStyle = 'rgba(7,13,20,0.88)';
            ct.beginPath(); ct.roundRect(0, 0, 256, 64, 8); ct.fill();
            ct.strokeStyle = 'rgba(0,194,255,0.65)'; ct.lineWidth = 2;
            ct.beginPath(); ct.roundRect(1, 1, 254, 62, 7); ct.stroke();
            ct.fillStyle = '#dde6f0'; ct.font = 'bold 18px monospace';
            ct.textAlign = 'center'; ct.textBaseline = 'middle';
            ct.fillText(ap.ssid || '<hidden>', 128, 22);
            ct.fillStyle = '#00c2ff'; ct.font = '14px monospace';
            ct.fillText(`${ap.dbm.toFixed(0)} dBm`, 128, 44);
            const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c2), transparent: true, opacity: 0.95 }));
            sp.scale.set(1.8, 0.45, 1); sp.position.set(wx, apH + 0.48, wz);
            apGroup.add(sp);
          });
        }

        // ── Heatmap (only for active room) ───────────────────────
        if ((viewMode === 'heatmap' || viewMode === 'both') && isActive) {
          const remapped = samples.map(s => { const { wx, wz } = remapIntoRoom(s); return { wx, wz, dbm: s.dbm }; });
          buildHeatmap(remapped);
        }
      });

      // If no active room set, fall back to rendering all live samples globally (old behaviour)
      if (activeRoomId === null && rfSamples.length > 0) {
        const filtered = activeSSID ? rfSamples.filter(s => s.ssid === activeSSID) : rfSamples;
        if (filtered.length) {
          totalSamples = filtered.length;
          let sxMin=Infinity,sxMax=-Infinity,syMin=Infinity,syMax=-Infinity,szMin=Infinity,szMax=-Infinity;
          filtered.forEach(s=>{sxMin=Math.min(sxMin,s.x??0);sxMax=Math.max(sxMax,s.x??0);syMin=Math.min(syMin,s.y??0);syMax=Math.max(syMax,s.y??0);szMin=Math.min(szMin,s.z??0);szMax=Math.max(szMax,s.z??0);});
          const sxSpan=Math.max(sxMax-sxMin,0.001),sySpan=Math.max(syMax-syMin,0.001),szSpan=Math.max(szMax-szMin,0.001),hasRealZ=szSpan>0.3;
          let rxMin=Infinity,rxMax=-Infinity,ryMin=Infinity,ryMax=-Infinity;
          rooms.forEach(r=>{rxMin=Math.min(rxMin,r.x);rxMax=Math.max(rxMax,r.x+r.w);ryMin=Math.min(ryMin,r.y);ryMax=Math.max(ryMax,r.y+r.d);});
          const margin=0.4,rxSpan=Math.max(rxMax-rxMin-margin*2,0.001),rySpan=Math.max(ryMax-ryMin-margin*2,0.001),maxH=rooms.length?Math.min(...rooms.map(r=>r.h)):2.6;
          function jFb(s,range){const seed=(s.bssid||s.ssid||'').split('').reduce((a,c)=>a+c.charCodeAt(0),0)+Math.round((s.x||0)*100)+Math.round((s.y||0)*100);return(((seed*1664525+1013904223)&0xffffffff)/0xffffffff-0.5)*range;}
          function remapFb(s){const wx=rxMin+margin+((s.x-sxMin)/sxSpan)*rxSpan;const wz=ryMin+margin+((s.y-syMin)/sySpan)*rySpan;const t=Math.max(0,Math.min(1,((s.dbm??-90)-(-90))/60));const wy=hasRealZ?Math.min(Math.max(0.08,((s.z-szMin)/szSpan)*maxH*0.85),maxH*0.9):Math.min(Math.max(0.05,0.15+t*maxH*0.72+jFb(s,0.35)),maxH*0.92);return{wx,wy,wz};}
          const sphGeo=new THREE.SphereGeometry(dotSize,10,8);
          filtered.forEach(s=>{const{wx,wy,wz}=remapFb(s);if(s.dbm>bestDBM)bestDBM=s.dbm;const col=dbmToColor(s.dbm);const mesh=new THREE.Mesh(sphGeo,new THREE.MeshStandardMaterial({color:col,roughness:0.3,metalness:0.2,emissive:col,emissiveIntensity:0.35}));mesh.position.set(wx,wy,wz);mesh.userData={dbm:s.dbm,ssid:s.ssid};rfGroup.add(mesh);});
          if(showPath&&filtered.length>=2){const pts=filtered.map(s=>{const{wx,wz}=remapFb(s);return new THREE.Vector3(wx,0.05,wz);});pathGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color:0x7c3aed,opacity:0.7,transparent:true})));}
          if(viewMode==='heatmap'||viewMode==='both'){buildHeatmap(filtered.map(s=>{const{wx,wz}=remapFb(s);return{wx,wz,dbm:s.dbm};}));}
        }
      }

      rfGroup.visible = (viewMode === 'spheres' || viewMode === 'both');

      // Update stats
      const dbmVals = [...rfSamples, ...Object.values(roomSamples).flat()].map(s => s.dbm).filter(Boolean);
      if (dbmVals.length) {
        document.getElementById('st-best').textContent = Math.max(...dbmVals).toFixed(0) + ' dBm';
        document.getElementById('st-avg').textContent = (dbmVals.reduce((a,b) => a+b, 0) / dbmVals.length).toFixed(0) + ' dBm';
      }
      document.getElementById('st-samples').textContent = totalSamples;
      document.getElementById('hud-samples').textContent = totalSamples + ' samples';
      refreshActiveRoomSelector();
    }

    // ─────────────────────────────────────────────────────────────
    // Auto-fit camera to home bounding box
    // ─────────────────────────────────────────────────────────────
    function autoFit() {
      if (!rooms.length) return;
      let xMin = Infinity, xMax = -Infinity, zMin = Infinity, zMax = -Infinity;
      rooms.forEach(r => {
        xMin = Math.min(xMin, r.x); xMax = Math.max(xMax, r.x + r.w);
        zMin = Math.min(zMin, r.y); zMax = Math.max(zMax, r.y + r.d);
      });
      const cx = (xMin + xMax) / 2, cz = (zMin + zMax) / 2;
      const span = Math.max(xMax - xMin, zMax - zMin, 6);
      controls.target.set(cx, 0, cz);
      camera.position.set(cx + span * 0.6, span * 0.9, cz + span * 0.9);
      controls.update();
    }

    // ─────────────────────────────────────────────────────────────
    // Fetch RF data from Flask
    // ─────────────────────────────────────────────────────────────
    async function fetchRFData() {
      document.getElementById('loader').classList.remove('hidden');
      document.getElementById('loader-txt').textContent = 'Loading RF samples…';
      try {
        // fetch SSIDs
        const ssidRes = await fetch('/api/rf/ssids').then(r => r.json()).catch(() => null);
        if (ssidRes && ssidRes.ssids) {
          const sel = document.getElementById('ssid-sel');
          const prev = sel.value;
          sel.innerHTML = '<option value="">All Networks</option>';
          ssidRes.ssids.forEach(s => {
            const o = document.createElement('option');
            o.value = s; o.textContent = s; sel.appendChild(o);
          });
          sel.value = prev;
          document.getElementById('st-ssids').textContent = ssidRes.ssid_count || 0;
        }

        // fetch samples
        let url = `/api/rf/samples?max=3000`;
        if (activeSSID) url += `&ssid=${encodeURIComponent(activeSSID)}`;
        const sRes = await fetch(url).then(r => r.json()).catch(() => null);
        
        if (sRes && Array.isArray(sRes.samples)) {
          const allSamples = sRes.samples;
          if (allSamples.length > 0) {
            globalLatestTimestamp = allSamples[allSamples.length - 1].timestamp;
          }
          
          // Clear arrays before re-categorizing
          rooms.forEach(r => roomSamples[r.id] = []);
          rfSamples = []; 
          
          const openIntervalStart = activeRoomId !== null ? activeRoomStartTime : Infinity;
          
          allSamples.forEach(s => {
            let matched = false;
            // 1. Check closed intervals
            for (const r of rooms) {
              const intervals = roomIntervals[r.id] || [];
              for (const iv of intervals) {
                if (s.timestamp >= iv.start && s.timestamp <= iv.end) {
                  // If it belongs to the currently active room, put it in rfSamples so it renders as live data
                  if (r.id === activeRoomId) {
                    rfSamples.push(s);
                  } else {
                    roomSamples[r.id].push(s);
                  }
                  matched = true;
                  break;
                }
              }
              if (matched) break;
            }
            
            // 2. If not matched, check open interval (current live scan)
            if (!matched) {
              if (activeRoomId !== null && s.timestamp >= openIntervalStart) {
                rfSamples.push(s);
              } else if (activeRoomId === null) {
                // If no active room is set, everything unmatched just shows globally
                rfSamples.push(s);
              }
            }
          });
        }

        buildRFSamples();

      } catch (e) {
        console.error('fetchRFData:', e);
      } finally {
        document.getElementById('loader').classList.add('hidden');
      }
    }


    // ─────────────────────────────────────────────────────────────
    // UI: Rooms sidebar
    // ─────────────────────────────────────────────────────────────
    function renderRoomList() {
      const list = document.getElementById('room-list');
      list.innerHTML = '';
      rooms.forEach(r => {
        const isCollecting = r.id == activeRoomId;
        const frozenCount = (roomSamples[r.id] || []).length;
        const el = document.createElement('div');
        el.className = 'room-item' + (r.id === selectedRoomId ? ' active' : '');
        if (isCollecting) el.style.outline = `1px solid #${new THREE.Color(r.color).getHexString()}88`;
        el.innerHTML = `
      <div class="room-dot" style="background:#${new THREE.Color(r.color).getHexString()}"></div>
      <div class="room-label">${r.name}${isCollecting ? ' <span style="color:#00c2ff;font-size:10px">📡</span>' : ''}${frozenCount && !isCollecting ? ` <span style="font-size:9px;color:var(--muted)">(${frozenCount}pts)</span>` : ''}</div>
      <button class="room-del" data-id="${r.id}" title="Remove">✕</button>
    `;
        el.addEventListener('click', e => {
          if (e.target.classList.contains('room-del')) return;
          selectRoom(r.id);
        });
        el.querySelector('.room-del').addEventListener('click', () => deleteRoom(r.id));
        list.appendChild(el);
      });
      document.getElementById('room-count-badge').textContent = rooms.length + ' room' + (rooms.length !== 1 ? 's' : '');
    }


    function selectRoom(id) {
      selectedRoomId = id;
      const r = rooms.find(x => x.id === id);
      if (!r) { document.getElementById('room-editor').style.display = 'none'; return; }
      document.getElementById('room-editor').style.display = '';
      document.getElementById('re-name').value = r.name;
      document.getElementById('re-x').value = r.x;
      document.getElementById('re-y').value = r.y;
      document.getElementById('re-w').value = r.w;
      document.getElementById('re-d').value = r.d;
      document.getElementById('re-h').value = r.h;
      document.getElementById('re-h-val').textContent = r.h;
      renderRoomList();
    }

    function deleteRoom(id) {
      rooms = rooms.filter(r => r.id !== id);
      if (selectedRoomId === id) {
        selectedRoomId = null;
        document.getElementById('room-editor').style.display = 'none';
      }
      renderRoomList();
      buildRooms();
      saveLayout(true);  // auto-save
    }

    document.getElementById('btn-apply-room').addEventListener('click', () => {
      const r = rooms.find(x => x.id === selectedRoomId);
      if (!r) return;
      r.name = document.getElementById('re-name').value || r.name;
      r.x = parseFloat(document.getElementById('re-x').value) || 0;
      r.y = parseFloat(document.getElementById('re-y').value) || 0;
      r.w = Math.max(0.5, parseFloat(document.getElementById('re-w').value) || 4);
      r.d = Math.max(0.5, parseFloat(document.getElementById('re-d').value) || 3);
      r.h = parseFloat(document.getElementById('re-h').value) || 2.5;
      renderRoomList();
      buildRooms();
      saveLayout(true);  // auto-save after edit
    });

    document.getElementById('re-h').addEventListener('input', e => {
      document.getElementById('re-h-val').textContent = parseFloat(e.target.value).toFixed(1);
    });

    document.getElementById('btn-del-room').addEventListener('click', () => {
      if (selectedRoomId !== null) deleteRoom(selectedRoomId);
    });

    // Add room modal
    document.getElementById('btn-add-room').addEventListener('click', () => {
      document.getElementById('add-modal').classList.add('open');
    });
    document.getElementById('mod-cancel').addEventListener('click', () => {
      document.getElementById('add-modal').classList.remove('open');
    });
    document.getElementById('mod-confirm').addEventListener('click', () => {
      const name = document.getElementById('mod-name').value || `Room ${nextId}`;
      rooms.push({
        id: nextId,
        name,
        x: parseFloat(document.getElementById('mod-x').value) || 0,
        y: parseFloat(document.getElementById('mod-y').value) || 0,
        w: parseFloat(document.getElementById('mod-w').value) || 4,
        d: parseFloat(document.getElementById('mod-d').value) || 3,
        h: 2.6,
        color: ROOM_COLORS[nextId % ROOM_COLORS.length]
      });
      nextId++;
      document.getElementById('add-modal').classList.remove('open');
      renderRoomList();
      buildRooms();
      saveLayout(true);  // auto-save after add
    });

    // ─────────────────────────────────────────────────────────────
    // Display controls
    // ─────────────────────────────────────────────────────────────
    document.getElementById('ssid-sel').addEventListener('change', e => {
      activeSSID = e.target.value;
      buildRFSamples();
    });
    document.getElementById('sel-mode').addEventListener('change', e => {
      viewMode = e.target.value;
      // Show/hide resolution slider based on mode
      const showHeatmapControls = viewMode !== 'spheres';
      document.getElementById('row-resolution').style.display = showHeatmapControls ? '' : 'none';
      document.getElementById('row-size').style.display = (viewMode !== 'heatmap') ? '' : 'none';
      buildRFSamples();
    });
    document.getElementById('sl-res').addEventListener('input', e => {
      heatmapRes = parseInt(e.target.value);
      document.getElementById('sl-res-val').textContent = heatmapRes;
      buildRFSamples();
    });
    document.getElementById('sl-size').addEventListener('input', e => {
      dotSize = parseFloat(e.target.value);
      document.getElementById('sl-size-val').textContent = dotSize.toFixed(2);
      buildRFSamples();
    });
    document.getElementById('sel-path').addEventListener('change', e => {
      showPath = e.target.value === '1';
      buildRFSamples();
    });
    document.getElementById('sel-ap').addEventListener('change', e => {
      showAP = e.target.value === '1';
      buildRFSamples();
    });
    // Initialise visibility based on default mode (heatmap)
    document.getElementById('row-size').style.display = 'none';

    // ─────────────────────────────────────────────────────────────
    // Header buttons
    // ─────────────────────────────────────────────────────────────
    document.getElementById('btn-refresh').addEventListener('click', fetchRFData);
    document.getElementById('btn-reset-cam').addEventListener('click', () => {
      camera.position.set(10, 14, 16); camera.lookAt(5, 0, 4); controls.update();
    });
    document.getElementById('btn-auto-fit').addEventListener('click', autoFit);

    document.getElementById('btn-export-png').addEventListener('click', () => {
      renderer.render(scene, camera);
      const link = document.createElement('a');
      link.href = canvas.toDataURL('image/png');
      link.download = 'rf_digital_twin_3d.png';
      link.click();
    });
    document.getElementById('btn-dl-csv').addEventListener('click', () => {
      let url = '/api/rf/export/csv';
      if (activeSSID) url += `?ssid=${encodeURIComponent(activeSSID)}`;
      window.location.href = url;
    });
    document.getElementById('btn-clear').addEventListener('click', async () => {
      if (!confirm('Clear all RF samples?')) return;
      await fetch('/api/rf/clear', { method: 'POST' });
      rfSamples = [];
      buildRFSamples();
      document.getElementById('st-samples').textContent = '0';
      document.getElementById('hud-samples').textContent = '0 samples';
    });

    // ─────────────────────────────────────────────────────────────
    // Resize handler
    // ─────────────────────────────────────────────────────────────
    function onResize() {
      const main = document.querySelector('main');
      const w = main.clientWidth, h = main.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    new ResizeObserver(onResize).observe(document.querySelector('main'));
    onResize();

    // ─────────────────────────────────────────────────────────────
    // IMU Gyro Sync (Head-Tracking / Handheld Rotation)
    // ─────────────────────────────────────────────────────────────
    let gyroSyncEnabled = false;
    let targetYawRad = 0;
    let currentYawRad = 0;

    function toggleGyroSync() {
      gyroSyncEnabled = !gyroSyncEnabled;
      const label = gyroSyncEnabled ? '🧭 Gyro Sync: ON' : '🧭 Gyro Sync: OFF';
      const hudBtn = document.getElementById('btn-hud-gyro');
      const sideBtn = document.getElementById('btn-gyro-toggle');
      if (hudBtn) {
        hudBtn.textContent = gyroSyncEnabled ? '🧭 Gyro: ON' : '🧭 Gyro: OFF';
        hudBtn.classList.toggle('primary', gyroSyncEnabled);
      }
      if (sideBtn) {
        sideBtn.textContent = label;
        sideBtn.classList.toggle('btn-primary', gyroSyncEnabled);
        sideBtn.classList.toggle('btn-secondary', !gyroSyncEnabled);
      }
    }

    document.getElementById('btn-gyro-toggle')?.addEventListener('click', toggleGyroSync);
    document.getElementById('btn-hud-gyro')?.addEventListener('click', toggleGyroSync);

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

    // ─────────────────────────────────────────────────────────────
    // Render loop
    // ─────────────────────────────────────────────────────────────
    let _frame = 0;
    function animate() {
      requestAnimationFrame(animate);

      if (gyroSyncEnabled) {
        // Smooth lerp toward target yaw
        let diff = targetYawRad - currentYawRad;
        while (diff < -Math.PI) diff += Math.PI * 2;
        while (diff > Math.PI) diff -= Math.PI * 2;
        currentYawRad += diff * 0.12;

        // Reposition camera using THREE.Spherical — the correct way
        // to programmatically set OrbitControls azimuth angle
        const offset = camera.position.clone().sub(controls.target);
        const spherical = new THREE.Spherical().setFromVector3(offset);
        spherical.theta = currentYawRad;   // azimuth (horizontal rotation)
        offset.setFromSpherical(spherical);
        camera.position.copy(controls.target).add(offset);
        camera.lookAt(controls.target);
      }

      controls.update();
      _frame++;
      // Gentle pulsing for spheres mode
      if ((viewMode === 'spheres' || viewMode === 'both') && _frame % 3 === 0) {
        rfGroup.children.forEach((m, i) => {
          if (m.material && !m.material.vertexColors) {
            m.material.emissiveIntensity = 0.25 + 0.15 * Math.sin(_frame * 0.04 + i * 0.5);
          }
        });
      }
      renderer.render(scene, camera);
    }

    // ─────────────────────────────────────────────────────────────
    // Tooltip on hover
    // ─────────────────────────────────────────────────────────────
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let tooltip = null;

    canvas.addEventListener('mousemove', e => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(rfGroup.children);
      if (hits.length) {
        const d = hits[0].object.userData;
        canvas.title = `${d.ssid || 'Unknown'}: ${d.dbm} dBm`;
      } else {
        canvas.title = '';
      }
    });

    // ─────────────────────────────────────────────────────────────
    // Auto-refresh every 8 s
    // ─────────────────────────────────────────────────────────────
    let autoRefreshTimer = setInterval(fetchRFData, 8000);

    // ─────────────────────────────────────────────────────────────
    // Save button handler
    // ─────────────────────────────────────────────────────────────
    document.getElementById('btn-save-layout').addEventListener('click', () => saveLayout(false));

    // ─────────────────────────────────────────────────────────────
    // Layout selector & management buttons
    // ─────────────────────────────────────────────────────────────
    document.getElementById('layout-selector').addEventListener('change', async (e) => {
      const name = e.target.value;
      if (name && name !== activeLayoutName) {
        await loadLayout(name);
        showLayoutStatus(`Loaded "${name}"`);
      }
    });

    document.getElementById('btn-new-layout').addEventListener('click', async () => {
      const name = prompt('Enter a name for the new layout:');
      if (!name || !name.trim()) return;
      // Save current rooms as the new layout
      activeLayoutName = name.trim();
      rooms = [];
      nextId = 0;
      updateLayoutTitle();
      renderRoomList();
      buildRooms();
      await saveLayout(true);
      await populateLayoutSelector(activeLayoutName);
      showLayoutStatus(`Created "${activeLayoutName}"`);
    });

    document.getElementById('btn-rename-layout').addEventListener('click', async () => {
      const newName = prompt(`Rename "${activeLayoutName}" to:`, activeLayoutName);
      if (!newName || !newName.trim() || newName.trim() === activeLayoutName) return;
      try {
        const res = await fetch('/api/rooms/layout/rename', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ oldName: activeLayoutName, newName: newName.trim() })
        });
        const data = await res.json();
        if (data.ok) {
          activeLayoutName = newName.trim();
          updateLayoutTitle();
          await populateLayoutSelector(activeLayoutName);
          showLayoutStatus(`Renamed to "${activeLayoutName}"`);
        } else {
          showLayoutStatus(data.error || 'Rename failed', false);
        }
      } catch (e) {
        showLayoutStatus('Rename failed (network error)', false);
      }
    });

    document.getElementById('btn-delete-layout').addEventListener('click', async () => {
      if (!confirm(`Delete layout "${activeLayoutName}"? This cannot be undone.`)) return;
      try {
        const res = await fetch('/api/rooms/layout/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: activeLayoutName })
        });
        const data = await res.json();
        if (data.ok) {
          showLayoutStatus(`Deleted "${activeLayoutName}"`);
          await populateLayoutSelector(data.active);
          await loadLayout(data.active || 'My Home');
        } else {
          showLayoutStatus(data.error || 'Delete failed', false);
        }
      } catch (e) {
        showLayoutStatus('Delete failed (network error)', false);
      }
    });

    // ─────────────────────────────────────────────────────────────
    // Active Room selector handler
    // ─────────────────────────────────────────────────────────────
    document.getElementById('active-room-sel').addEventListener('change', (e) => {
      const newId = e.target.value === '' ? null : parseInt(e.target.value, 10);
      const now = globalLatestTimestamp > 0 ? globalLatestTimestamp : (Date.now() / 1000);

      // 1. Close the current active room's interval
      if (activeRoomId !== null) {
        if (!roomIntervals[activeRoomId]) roomIntervals[activeRoomId] = [];
        roomIntervals[activeRoomId].push({ start: activeRoomStartTime || 0, end: now });
      }

      // 2. Open interval for the new room
      activeRoomId = newId;
      activeRoomStartTime = newId !== null ? now : null;
      
      const room = newId !== null ? rooms.find(r => r.id === newId) : null;
      setActiveRoomIndicator(room);

      if (room) {
        showRoomDataStatus(`Now collecting in "${room.name}"`);
      } else {
        showRoomDataStatus('No active room — showing all data globally');
      }
      
      // We manually clear rfSamples here immediately so the UI reflects the change 
      // instantly before the next 8s auto-refresh fetches the real data.
      rfSamples = [];
      buildRFSamples();
    });


    document.getElementById('btn-clear-room-data').addEventListener('click', () => {
      if (activeRoomId === null) { showRoomDataStatus('No active room selected', false); return; }
      const room = rooms.find(r => r.id === activeRoomId);
      if (!confirm(`Clear all collected data for "${room ? room.name : activeRoomId}"?`)) return;
      
      // Wipe closed intervals and live snapshot for this room
      roomIntervals[activeRoomId] = [];
      roomSamples[activeRoomId] = [];
      rfSamples = [];
      
      // Start a fresh open interval for this room exactly from NOW, ignoring all past server data
      activeRoomStartTime = globalLatestTimestamp > 0 ? globalLatestTimestamp : (Date.now() / 1000);
      
      buildRFSamples();
      showRoomDataStatus(`Cleared data for "${room ? room.name : 'room'}"`);
    });

    document.getElementById('btn-clear-all-rooms').addEventListener('click', () => {
      if (!confirm('Clear ALL room data? This removes every frozen snapshot.')) return;
      
      roomIntervals = {};
      rooms.forEach(r => roomSamples[r.id] = []);
      rfSamples = [];
      activeRoomStartTime = globalLatestTimestamp > 0 ? globalLatestTimestamp : (Date.now() / 1000);
      
      buildRFSamples();
      showRoomDataStatus('All room data cleared');
    });

    // ─────────────────────────────────────────────────────────────
    // Bootstrap — load layout list & active layout from Pi
    // ─────────────────────────────────────────────────────────────
    (async function bootstrap() {
      await populateLayoutSelector();
      await loadLayout(activeLayoutName);
      autoFit();
      animate();
      fetchRFData();
    })();


  