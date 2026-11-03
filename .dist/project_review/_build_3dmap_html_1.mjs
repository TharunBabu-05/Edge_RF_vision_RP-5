
        // Global variables for floor plan and samples
        let floorPlan = null;
        let samples = [];
        let activeSSID = '';

        // UI Element handles
        const canvas = document.getElementById('iso-canvas');
        const ctx = canvas.getContext('2d');
        const overlay = document.getElementById('loading-overlay');
        const overlayText = document.getElementById('loading-text');
        const heatmapImg = document.getElementById('heatmap-png');

        // Isometric View Projection Variables
        let rotation = 45 * Math.PI / 180; // default 45 degrees
        let tilt = 35 * Math.PI / 180;     // default 35 degrees
        let zoom = 35;
        let wallHeightScalar = 3.0;        // height factor in pixels
        
        let cx = 0; // relative origin centers (computed from floor plan bounds)
        let cy = 0;

        // Mouse Drag Interaction State
        let isDragging = false;
        let lastMouseX = 0;
        let lastMouseY = 0;

        // Init function
        window.addEventListener('DOMContentLoaded', async () => {
            setupEventListeners();
            resizeCanvas();
            await fetchMetadata();
            await loadDigitalTwinData();
        });

        function resizeCanvas() {
            const container = canvas.parentElement;
            canvas.width = container.clientWidth - 10;
            canvas.height = container.clientHeight - 65;
            draw();
        }

        window.addEventListener('resize', resizeCanvas);

        function setupEventListeners() {
            // Tab switching
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
                    btn.classList.add('active');
                    
                    const tab = btn.dataset.tab;
                    if (tab === 'canvas') {
                        document.getElementById('canvas-wrapper').style.display = 'flex';
                        document.getElementById('png-wrapper').style.display = 'none';
                        draw();
                    } else {
                        document.getElementById('canvas-wrapper').style.display = 'none';
                        document.getElementById('png-wrapper').style.display = 'flex';
                        refreshStaticPng();
                    }
                });
            });

            // Input handlers
            document.getElementById('ssid-select').addEventListener('change', async (e) => {
                activeSSID = e.target.value;
                await loadDigitalTwinData();
                if (document.getElementById('png-wrapper').style.display === 'flex') {
                    refreshStaticPng();
                }
            });

            document.getElementById('grid-resolution').addEventListener('input', (e) => {
                document.getElementById('grid-val').textContent = e.target.value;
            });

            // Sliders manual adjustment
            document.getElementById('slider-rotation').addEventListener('input', (e) => {
                rotation = parseFloat(e.target.value) * Math.PI / 180;
                draw();
            });
            document.getElementById('slider-tilt').addEventListener('input', (e) => {
                tilt = parseFloat(e.target.value) * Math.PI / 180;
                draw();
            });
            document.getElementById('slider-zoom').addEventListener('input', (e) => {
                zoom = parseFloat(e.target.value);
                draw();
            });
            document.getElementById('slider-height').addEventListener('input', (e) => {
                wallHeightScalar = parseFloat(e.target.value) / 10;
                draw();
            });

            document.getElementById('btn-reset-view').addEventListener('click', () => {
                rotation = 45 * Math.PI / 180;
                tilt = 35 * Math.PI / 180;
                zoom = 35;
                wallHeightScalar = 3.0;

                document.getElementById('slider-rotation').value = 45;
                document.getElementById('slider-tilt').value = 35;
                document.getElementById('slider-zoom').value = 35;
                document.getElementById('slider-height').value = 30;
                draw();
            });

            // Action buttons
            document.getElementById('btn-render-png').addEventListener('click', () => {
                // Switch to png tab and reload
                document.querySelector('.tab-btn[data-tab="png"]').click();
            });

            document.getElementById('btn-rebuild-floorplan').addEventListener('click', rebuildFloorPlan);
            
            document.getElementById('btn-download-csv').addEventListener('click', () => {
                let url = '/api/rf/export/csv';
                if (activeSSID) url += '?ssid=' + encodeURIComponent(activeSSID);
                window.location.href = url;
            });

            document.getElementById('btn-clear-db').addEventListener('click', async () => {
                if (!confirm("Clear all RF samples? This will wipe the active data session.")) return;
                showLoading("Clearing Database...");
                try {
                    await fetch('/api/rf/clear', { method: 'POST' });
                    await loadDigitalTwinData();
                } catch (e) {
                    console.error(e);
                } finally {
                    hideLoading();
                }
            });

            // Canvas drag interaction
            canvas.addEventListener('mousedown', (e) => {
                isDragging = true;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
            });

            window.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                const deltaX = e.clientX - lastMouseX;
                const deltaY = e.clientY - lastMouseY;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;

                rotation += deltaX * 0.006;
                tilt = Math.max(15 * Math.PI / 180, Math.min(80 * Math.PI / 180, tilt - deltaY * 0.006));

                // update sliders to match
                document.getElementById('slider-rotation').value = Math.round(rotation * 180 / Math.PI);
                document.getElementById('slider-tilt').value = Math.round(tilt * 180 / Math.PI);

                draw();
            });

            window.addEventListener('mouseup', () => { isDragging = false; });
        }

        function showLoading(text) {
            overlayText.textContent = text || 'Loading...';
            overlay.style.display = 'flex';
        }

        function hideLoading() {
            overlay.style.display = 'none';
        }

        // Fetch unique SSIDs to fill select options
        async function fetchMetadata() {
            try {
                const s = await fetch('/api/rf/ssids').then(r => r.json());
                const sel = document.getElementById('ssid-select');
                
                // save selected value
                const prev = sel.value;
                sel.innerHTML = '<option value="">All Networks</option>';
                
                if (s.ssids && Array.isArray(s.ssids)) {
                    s.ssids.forEach(ssid => {
                        const o = document.createElement('option');
                        o.value = ssid;
                        o.textContent = ssid;
                        sel.appendChild(o);
                    });
                }
                sel.value = prev;
            } catch (err) {
                console.error("Failed to load metadata", err);
            }
        }

        // Load data from backend
        async function loadDigitalTwinData() {
            showLoading("Syncing spatial digital twin...");
            try {
                // Fetch floor plan bounds and path
                const fpResponse = await fetch('/api/rf/floorplan');
                if (fpResponse.status === 400 || fpResponse.status === 500) {
                    floorPlan = null;
                } else {
                    floorPlan = await fpResponse.json();
                }

                // Fetch samples
                const url = activeSSID ? `/api/rf/samples?ssid=${encodeURIComponent(activeSSID)}&max=2000` : '/api/rf/samples?max=2000';
                const sRes = await fetch(url).then(r => r.json());
                samples = sRes.samples || [];

                updateStats();
                draw();
            } catch (err) {
                console.error("Error loading twin data", err);
            } finally {
                hideLoading();
            }
        }

        // Force rebuild floor plan
        async function rebuildFloorPlan() {
            showLoading("Reconstructing room boundaries...");
            try {
                const r = await fetch('/api/rf/floorplan/rebuild', { method: 'POST' }).then(r => r.json());
                if (!r.ok) {
                    alert("Rebuild failed: " + (r.error || '?'));
                } else {
                    await loadDigitalTwinData();
                }
            } catch (e) {
                alert("Rebuild failed.");
            } finally {
                hideLoading();
            }
        }

        // Refresh static Matplotlib PNG
        async function refreshStaticPng() {
            showLoading("Generating Matplotlib 3D isometric surface...");
            const res = document.getElementById('grid-resolution').value;
            let url = `/api/rf/render3d?grid=${res}&t=${Date.now()}`;
            if (activeSSID) url += `&ssid=${encodeURIComponent(activeSSID)}`;
            
            heatmapImg.onload = () => {
                hideLoading();
            };
            heatmapImg.onerror = () => {
                hideLoading();
                alert("Failed to load 3D render. Verify that you have collected enough samples (>=2) and matplotlib is installed on the Pi.");
            };
            heatmapImg.src = url;
        }

        function updateStats() {
            document.getElementById('stat-samples').textContent = samples.length;
            
            if (floorPlan) {
                const w = (floorPlan.x_max - floorPlan.x_min).toFixed(1);
                const d = (floorPlan.y_max - floorPlan.y_min).toFixed(1);
                document.getElementById('stat-dimensions').textContent = `${w} x ${d} VO units`;
                document.getElementById('stat-aps').textContent = floorPlan.ap_locations ? floorPlan.ap_locations.length : 0;
            } else {
                document.getElementById('stat-dimensions').textContent = "Not reconstructed";
                document.getElementById('stat-aps').textContent = 0;
            }

            fetch('/api/rf/ssids')
                .then(r => r.json())
                .then(s => {
                    document.getElementById('stat-ssids').textContent = s.ssid_count || 0;
                }).catch(()=>{});
        }

        // Isometric projection formula: centers coords around (cx, cy) and rotates/tilts
        function project(x, y, z) {
            const dx = x - cx;
            const dy = y - cy;

            // Rotate about Z-axis
            const rx = dx * Math.cos(rotation) - dy * Math.sin(rotation);
            const ry = dx * Math.sin(rotation) + dy * Math.cos(rotation);

            // Apply tilt and vertical scale to Z
            const px = rx * zoom + (canvas.width / 2);
            const py = (ry * Math.sin(tilt) - z * wallHeightScalar) * zoom + (canvas.height / 2) + 20;

            return { x: px, y: py, depth: ry };
        }

        function dbmToColor(dbm) {
            if (dbm >= -45) return '#22d38a'; // pure green
            if (dbm <= -85) return '#ef4444'; // pure red
            
            const pct = (dbm - (-85)) / 40.0;
            if (pct < 0.5) {
                const factor = pct * 2;
                const r = Math.round(239 + (245 - 239) * factor);
                const g = Math.round(68 + (158 - 68) * factor);
                const b = Math.round(68 + (11 - 68) * factor);
                return `rgb(${r},${g},${b})`;
            } else {
                const factor = (pct - 0.5) * 2;
                const r = Math.round(245 + (34 - 245) * factor);
                const g = Math.round(158 + (211 - 158) * factor);
                const b = Math.round(11 + (138 - 11) * factor);
                return `rgb(${r},${g},${b})`;
            }
        }

        // 3D Canvas rendering engine
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (!floorPlan || !floorPlan.rooms || floorPlan.rooms.length === 0) {
                ctx.fillStyle = 'var(--muted)';
                ctx.font = '14px var(--font)';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText("No spatial model available yet. Complete a survey in the dashboard to generate a floor plan.", canvas.width / 2, canvas.height / 2);
                return;
            }

            // Update projection origin
            cx = (floorPlan.x_min + floorPlan.x_max) / 2;
            cy = (floorPlan.y_min + floorPlan.y_max) / 2;

            // 1. Draw floor grid at z = 0
            drawGrid();

            // 2. Draw Walk path trace on the floor (z = 0)
            drawWalkPath();

            // 3. Compile all 3D drawables to sort them by depth (back to front)
            const drawables = [];

            // Add walls
            const wallHeight = 2.0; // wall height in meters
            floorPlan.rooms.forEach(room => {
                const segments = [
                    [room.x_min, room.y_min, room.x_max, room.y_min],
                    [room.x_max, room.y_min, room.x_max, room.y_max],
                    [room.x_max, room.y_max, room.x_min, room.y_max],
                    [room.x_min, room.y_max, room.x_min, room.y_min]
                ];
                segments.forEach(seg => {
                    const mx = (seg[0] + seg[2]) / 2;
                    const my = (seg[1] + seg[3]) / 2;
                    // Project center of segment to determine depth
                    const dx = mx - cx;
                    const dy = my - cy;
                    const depth = dx * Math.sin(rotation) + dy * Math.cos(rotation);
                    
                    drawables.push({
                        type: 'wall',
                        depth: depth,
                        data: seg,
                        height: wallHeight
                    });
                });
            });

            // Add RF samples
            samples.forEach(s => {
                const dx = s.x - cx;
                const dy = s.y - cy;
                const depth = dx * Math.sin(rotation) + dy * Math.cos(rotation);

                drawables.push({
                    type: 'sample',
                    depth: depth,
                    data: s
                });
            });

            // Add Access Points (APs)
            if (floorPlan.ap_locations) {
                floorPlan.ap_locations.forEach(ap => {
                    const dx = ap.x - cx;
                    const dy = ap.y - cy;
                    const depth = dx * Math.sin(rotation) + dy * Math.cos(rotation);

                    drawables.push({
                        type: 'ap',
                        depth: depth,
                        data: ap
                    });
                });
            }

            // Sort depth: smaller depth values are drawn first (painted over by closer things)
            drawables.sort((a, b) => a.depth - b.depth);

            // 4. Render drawables back-to-front
            drawables.forEach(d => {
                if (d.type === 'wall') {
                    renderWall(d.data[0], d.data[1], d.data[2], d.data[3], d.height);
                } else if (d.type === 'sample') {
                    renderSample(d.data);
                } else if (d.type === 'ap') {
                    renderAP(d.data);
                }
            });
        }

        function drawGrid() {
            ctx.strokeStyle = 'rgba(26, 47, 68, 0.25)';
            ctx.lineWidth = 1;

            const gridSpacing = 1.0;
            const x_min = Math.floor(floorPlan.x_min);
            const x_max = Math.ceil(floorPlan.x_max);
            const y_min = Math.floor(floorPlan.y_min);
            const y_max = Math.ceil(floorPlan.y_max);

            // Draw grid line intersections
            for (let x = x_min; x <= x_max; x += gridSpacing) {
                ctx.beginPath();
                const start = project(x, y_min, 0);
                const end = project(x, y_max, 0);
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);
                ctx.stroke();
            }

            for (let y = y_min; y <= y_max; y += gridSpacing) {
                ctx.beginPath();
                const start = project(x_min, y, 0);
                const end = project(x_max, y, 0);
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);
                ctx.stroke();
            }
        }

        function drawWalkPath() {
            if (!floorPlan.walk_path || floorPlan.walk_path.length < 2) return;

            ctx.beginPath();
            const startPt = project(floorPlan.walk_path[0][0], floorPlan.walk_path[0][1], 0.02);
            ctx.moveTo(startPt.x, startPt.y);

            for (let i = 1; i < floorPlan.walk_path.length; i++) {
                const pt = project(floorPlan.walk_path[i][0], floorPlan.walk_path[i][1], 0.02);
                ctx.lineTo(pt.x, pt.y);
            }

            ctx.strokeStyle = 'rgba(124, 58, 237, 0.55)'; // violet trail
            ctx.lineWidth = 2.5;
            ctx.setLineDash([3, 3]);
            ctx.stroke();
            ctx.setLineDash([]); // reset

            // Start & End markers
            const start = project(floorPlan.walk_path[0][0], floorPlan.walk_path[0][1], 0.02);
            const end = project(floorPlan.walk_path[floorPlan.walk_path.length - 1][0], floorPlan.walk_path[floorPlan.walk_path.length - 1][1], 0.02);
            
            ctx.beginPath();
            ctx.arc(start.x, start.y, 4, 0, 2*Math.PI);
            ctx.fillStyle = 'var(--good)';
            ctx.fill();
            
            ctx.beginPath();
            ctx.arc(end.x, end.y, 4, 0, 2*Math.PI);
            ctx.fillStyle = 'var(--danger)';
            ctx.fill();
        }

        function renderWall(x1, y1, x2, y2, height) {
            const base1 = project(x1, y1, 0);
            const base2 = project(x2, y2, 0);
            const top1 = project(x1, y1, height);
            const top2 = project(x2, y2, height);

            // Draw filled wall polygon
            ctx.beginPath();
            ctx.moveTo(base1.x, base1.y);
            ctx.lineTo(base2.x, base2.y);
            ctx.lineTo(top2.x, top2.y);
            ctx.lineTo(top1.x, top1.y);
            ctx.closePath();

            // Translucent blue glass fill
            ctx.fillStyle = 'rgba(0, 194, 255, 0.06)';
            ctx.fill();

            // Wireframe borders
            ctx.strokeStyle = 'rgba(26, 47, 68, 0.4)';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Bright wall cap line
            ctx.beginPath();
            ctx.moveTo(top1.x, top1.y);
            ctx.lineTo(top2.x, top2.y);
            ctx.strokeStyle = 'rgba(0, 194, 255, 0.35)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        function renderSample(s) {
            const floorPt = project(s.x, s.y, 0);
            const zPt = project(s.x, s.y, s.z);

            // Floor anchor point
            ctx.beginPath();
            ctx.arc(floorPt.x, floorPt.y, 1.8, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(122, 155, 181, 0.3)';
            ctx.fill();

            // Vertical line representing sample height (z)
            if (Math.abs(s.z) > 0.05) {
                ctx.beginPath();
                ctx.moveTo(floorPt.x, floorPt.y);
                ctx.lineTo(zPt.x, zPt.y);
                ctx.strokeStyle = 'rgba(122, 155, 181, 0.25)';
                ctx.setLineDash([2, 2]);
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.setLineDash([]);
            }

            // Colored sample dot
            ctx.beginPath();
            ctx.arc(zPt.x, zPt.y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = dbmToColor(s.dbm);
            ctx.fill();
            
            ctx.strokeStyle = 'rgba(7, 13, 20, 0.6)';
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }

        function renderAP(ap) {
            const floorPt = project(ap.x, ap.y, 0);
            const poleHeight = 1.3; // standard router placement height
            const topPt = project(ap.x, ap.y, poleHeight);

            // Pole
            ctx.beginPath();
            ctx.moveTo(floorPt.x, floorPt.y);
            ctx.lineTo(topPt.x, topPt.y);
            ctx.strokeStyle = 'var(--accent)';
            ctx.lineWidth = 1.8;
            ctx.stroke();

            // Antenna bulb
            ctx.beginPath();
            ctx.arc(topPt.x, topPt.y, 5.5, 0, 2 * Math.PI);
            ctx.fillStyle = 'var(--accent)';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1;
            ctx.stroke();

            // White center dot
            ctx.beginPath();
            ctx.arc(topPt.x, topPt.y, 1.8, 0, 2 * Math.PI);
            ctx.fillStyle = '#ffffff';
            ctx.fill();

            // SSID Name tags
            ctx.fillStyle = 'rgba(7, 13, 20, 0.85)';
            ctx.strokeStyle = 'var(--border)';
            ctx.lineWidth = 1;
            
            const label = ap.ssid || '<hidden>';
            ctx.font = '10px var(--mono)';
            const textWidth = ctx.measureText(label).width;
            
            const padX = 6;
            const padY = 4;
            const boxW = textWidth + padX * 2;
            const boxH = 14 + padY * 2;
            
            const boxX = topPt.x - boxW / 2;
            const boxY = topPt.y - 25;

            // Draw label box
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(boxX, boxY, boxW, boxH, 4);
            } else {
                ctx.rect(boxX, boxY, boxW, boxH);
            }
            ctx.fill();
            ctx.stroke();

            // Draw label text
            ctx.fillStyle = 'var(--text)';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(label, topPt.x, boxY + boxH / 2);
        }
    