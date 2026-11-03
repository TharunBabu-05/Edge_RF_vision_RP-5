
    // ─── State ────────────────────────────────────────────────────────
    let surveyActive = false, surveyCompleted = 0, surveyTotal = 0;
    let lastMapClickMs = 0;
    let scannerLastError = null;
    let currentSSIDs = [];

    // ─── Signal helpers ───────────────────────────────────────────────
    function dbmToColor(dbm) {
        const t = Math.max(0, Math.min(1, (dbm + 90) / 60));
        const r = Math.round(255 * (1 - t));
        const g = Math.round(255 * t);
        return `rgb(${r},${g},0)`;
    }
    function signalBars(dbm) {
        const t = Math.max(0, Math.min(1, (dbm + 90) / 60));
        const filled = Math.round(t * 4);
        const color = dbmToColor(dbm);
        let html = '';
        for (let i = 0; i < 4; i++) {
            const h = 4 + i * 4;
            const bg = i < filled ? color : 'rgba(255,255,255,0.12)';
            html += `<div class="net-bar" style="height:${h}px;background:${bg}"></div>`;
        }
        return html;
    }

    // ─── Status polling ───────────────────────────────────────────────
    async function pollStatus() {
        try {
            const s = await fetch('/api/status').then(r => r.json());
            document.getElementById('kpi-fps').textContent = (+s.fps || 0).toFixed(1);
            document.getElementById('fps-badge').textContent = (+s.fps || 0).toFixed(1) + ' fps';
            document.getElementById('kpi-dbm').textContent = (+s.signal_strength_dbm || -90).toFixed(0);
            document.getElementById('kpi-ci').textContent = (+s.combined_intensity || 0).toFixed(2);
            document.getElementById('kpi-uptime').textContent = Math.floor(s.uptime_seconds || 0);
            scannerLastError = s.scanner_last_error || null;
            const ns = document.getElementById('network-status');
            if (ns) ns.textContent = scannerLastError
                ? ('Scan error: ' + scannerLastError)
                : ('Scanner OK | latency: ' + (+s.scan_latency_ms || 0).toFixed(0) + ' ms');
        } catch (_) {}
    }

    // ─── VO pose polling ──────────────────────────────────────────────
    async function pollPose() {
        try {
            const p = await fetch('/api/rf/pose').then(r => r.json());
            document.getElementById('vo-x').textContent = (+p.x || 0).toFixed(2);
            document.getElementById('vo-y').textContent = (+p.y || 0).toFixed(2);
            document.getElementById('vo-z').textContent = (+p.z || 0).toFixed(2);
            document.getElementById('vo-pts').textContent = p.tracked_points || 0;
            document.getElementById('vo-conf').textContent = (+p.confidence || 0).toFixed(2);
            const status = (p.status || 'INITIALIZING').toUpperCase();
            const statusEl = document.getElementById('vo-overlay-status');
            if (statusEl) {
                statusEl.textContent = status;
                statusEl.className = 'vo-status-' + status.toLowerCase();
            }
            const badge = document.getElementById('vo-status-badge');
            const track = document.getElementById('tracking-badge');
            if (status === 'TRACKING') {
                badge.style.display = 'none';
                track.style.display = '';
                track.textContent = `✓ TRACKING (${p.tracked_points} pts)`;
            } else {
                badge.style.display = '';
                badge.textContent = 'VO ' + status;
                track.style.display = 'none';
            }
        } catch (_) {}
    }

    // ─── RF DB status ─────────────────────────────────────────────────
    async function pollRFDB() {
        try {
            const s = await fetch('/api/rf/db_status').then(r => r.json());
            document.getElementById('ss-total').textContent = s.total_samples || 0;
            document.getElementById('ss-ssids').textContent = s.ssid_count || 0;
            document.getElementById('db-sample-badge').textContent = (s.total_samples || 0) + ' samples';

            // Update SSID select
            const sel = document.getElementById('ssid-select');
            const prev = sel.value;
            const ssids = Array.isArray(s.ssids) ? s.ssids : [];
            if (JSON.stringify(ssids) !== JSON.stringify(currentSSIDs)) {
                currentSSIDs = ssids;
                sel.innerHTML = '<option value="">All networks</option>';
                for (const id of ssids) {
                    const o = document.createElement('option');
                    o.value = id; o.textContent = id;
                    sel.appendChild(o);
                }
                if (prev) sel.value = prev;
            }
        } catch (_) {}
    }

    // ─── Heatmap ──────────────────────────────────────────────────────
    function refreshHeatmap() {
        const ssid = document.getElementById('ssid-select').value;
        const method = document.getElementById('method-select').value;
        const params = new URLSearchParams({ method });
        if (ssid) params.set('ssid', ssid);
        document.getElementById('rf-heatmap').src = '/api/rf/heatmap?' + params.toString() + '&t=' + Date.now();
        document.getElementById('heatmap-status').textContent = `Updated: ${method} interpolation${ssid ? ' for ' + ssid : ''}`;
    }

    async function clearRFDB() {
        if (!confirm('Clear all RF samples from the database?')) return;
        await fetch('/api/rf/clear', { method: 'POST' });
        refreshHeatmap();
        pollRFDB();
    }

    async function resetVOPose() {
        await fetch('/api/rf/reset_pose', { method: 'POST' });
        document.getElementById('heatmap-status').textContent = 'VO pose reset to origin (0,0,0).';
    }

    // ─── Networks ─────────────────────────────────────────────────────
    async function pollNetworks() {
        try {
            const nets = await fetch('/api/networks').then(r => r.json());
            const list = document.getElementById('networks');
            list.innerHTML = '';
            if (!Array.isArray(nets) || !nets.length) {
                list.innerHTML = '<div style="padding:12px 14px;font-size:12px;color:var(--muted)">No networks detected</div>';
                return;
            }
            for (const n of nets) {
                const dbm = +(n.dbm || -100);
                const color = dbmToColor(dbm);
                const div = document.createElement('div');
                div.className = 'net-item';
                div.innerHTML =
                    '<div class="net-bars">' + signalBars(dbm) + '</div>' +
                    '<div class="net-info">' +
                      '<div class="net-ssid">' + (n.ssid || '<hidden>') + '</div>' +
                      '<div class="net-meta">' + (n.bssid || '') + ' · ' + (n.freq || 0) + ' MHz</div>' +
                    '</div>' +
                    '<div class="net-dbm" style="color:' + color + '">' + dbm.toFixed(0) + ' dBm</div>';
                list.appendChild(div);
            }
        } catch (_) {}
    }

    async function triggerScan() {
        const btn = document.getElementById('scan-now');
        if (btn) btn.disabled = true;
        try {
            await fetch('/api/scan/now', { method: 'POST' });
            await pollNetworks();
            await pollStatus();
        } catch (_) {}
        if (btn) btn.disabled = false;
    }

    // ─── 2D Mapper ────────────────────────────────────────────────────
    function refreshMapperImage() {
        const img = document.getElementById('map-view');
        if (img) img.src = '/api/mapper/heatmap?t=' + Date.now();
    }
    async function pollMapperStatus() {
        try {
            const s = await fetch('/api/mapper/status').then(r => r.json());
            const node = document.getElementById('mapper-status');
            if (node) node.textContent = s.has_position
                ? `Position: (${(+s.position_x).toFixed(2)}, ${(+s.position_y).toFixed(2)}) | Samples: ${s.sample_count}`
                : 'Position not set';
        } catch (_) {}
    }
    async function onMapClick(e) {
        const img = document.getElementById('map-view');
        if (!img) return;
        const now = Date.now();
        if (now - lastMapClickMs < 180) return;
        lastMapClickMs = now;
        const rect = img.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        try {
            await fetch('/api/mapper/position', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ x, y, sample_now: true }),
            });
            refreshMapperImage(); pollMapperStatus();
        } catch (_) {}
    }
    function setMapIoStatus(t) { const n=document.getElementById('map-io-status'); if(n) n.textContent=t||''; }
    async function refreshSnapshotList(pref) {
        try {
            const d = await fetch('/api/mapper/snapshots').then(r => r.json());
            const list = document.getElementById('map-snapshot-list');
            if (!list) return;
            const snaps = Array.isArray(d.snapshots) ? d.snapshots : [];
            list.innerHTML = snaps.length ? '' : '<option value="">No snapshots</option>';
            for (const n of snaps) {
                const o = document.createElement('option');
                o.value = n; o.textContent = n; list.appendChild(o);
            }
            if (pref) list.value = pref;
        } catch (_) {}
    }
    async function saveSnapshot() {
        const input = document.getElementById('map-name');
        const name = input ? input.value || '' : '';
        try {
            const r = await fetch('/api/mapper/save', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name}) });
            const d = await r.json();
            setMapIoStatus(d.ok ? 'Saved: ' + d.name : 'Save failed');
            if (d.ok) await refreshSnapshotList(d.name);
        } catch(_) { setMapIoStatus('Save failed'); }
    }
    async function loadSnapshot() {
        const list = document.getElementById('map-snapshot-list');
        const sel = list ? list.value : '';
        if (!sel) { setMapIoStatus('Select a snapshot'); return; }
        try {
            const r = await fetch('/api/mapper/load', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name:sel}) });
            const d = await r.json();
            setMapIoStatus(d.ok ? 'Loaded: ' + sel : 'Load failed');
            if (d.ok) { refreshMapperImage(); pollMapperStatus(); }
        } catch(_) { setMapIoStatus('Load failed'); }
    }
    async function clearMapper() {
        await fetch('/api/mapper/clear', {method:'POST'});
        refreshMapperImage(); pollMapperStatus();
    }

    // ─── Guided Survey ────────────────────────────────────────────────
    function setSurveyFeedback(t) { const n=document.getElementById('survey-feedback'); if(n) n.textContent=t||''; }
    function updateSurveyButtons() {
        const has = surveyCompleted > 0;
        document.getElementById('survey-start').disabled   = surveyActive;
        document.getElementById('survey-capture').disabled = !surveyActive;
        document.getElementById('survey-reset').disabled   = !surveyActive;
        document.getElementById('survey-export-png').disabled  = !has;
        document.getElementById('survey-export-json').disabled = !has;
        document.getElementById('survey-export-csv').disabled  = !has;
        const bar = document.getElementById('survey-bar');
        if (bar && surveyTotal > 0) bar.style.width = (surveyCompleted/surveyTotal*100).toFixed(1) + '%';
    }
    async function pollSurveyStatus() {
        try {
            const s = await fetch('/api/survey/status').then(r => r.json());
            surveyActive = !!s.active;
            surveyCompleted = +s.completed_points || 0;
            surveyTotal = +s.total_points || 0;
            const statusEl = document.getElementById('survey-status');
            const detailEl = document.getElementById('survey-detail');
            if (s.completed) {
                if (statusEl) statusEl.textContent = '✓ Survey complete!';
                if (detailEl) detailEl.textContent = `Captured ${s.completed_points}/${s.total_points} points`;
            } else if (s.active) {
                if (statusEl) statusEl.textContent = `Step ${s.current_index}/${s.total_points}: move to target, then Capture`;
                if (detailEl) detailEl.textContent = `Target: (${(+s.target_x).toFixed(2)}, ${(+s.target_y).toFixed(2)}) | Done: ${s.completed_points}`;
            } else {
                if (statusEl) statusEl.textContent = 'Ready to start';
                if (detailEl) detailEl.textContent = 'Set rows and cols, then click Start.';
            }
            updateSurveyButtons();
        } catch (_) {}
    }
    async function startSurvey() {
        const rows = +document.getElementById('survey-rows').value || 3;
        const cols = +document.getElementById('survey-cols').value || 3;
        try {
            const r = await fetch('/api/survey/start', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({rows,cols,margin:0.08,serpentine:true}) });
            const d = await r.json();
            setSurveyFeedback(d.ok ? 'Survey started. Go to target and click Capture.' : 'Start failed: '+(d.error||'?'));
            pollSurveyStatus();
        } catch(_) {}
    }
    async function capturePoint() {
        try {
            const r = await fetch('/api/survey/capture', {method:'POST'});
            const d = await r.json();
            setSurveyFeedback(d.ok ? `Saved ${d.completed_points}/${surveyTotal}` : 'Capture failed: '+(d.error||'?'));
            pollSurveyStatus(); pollMapperStatus(); refreshMapperImage();
        } catch(_) {}
    }
    async function resetSurvey() {
        await fetch('/api/survey/reset', {method:'POST'});
        setSurveyFeedback('Survey reset.'); pollSurveyStatus();
    }
    function downloadBlob(url, filename) {
        fetch(url).then(r=>r.blob()).then(blob=>{
            const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=filename;
            document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(a.href);
        }).catch(_=>{});
    }
    function exportSurveyJSON() { if(surveyCompleted<=0){setSurveyFeedback('No captures yet');return;} downloadBlob('/api/survey/data','survey_data.json'); setSurveyFeedback('JSON downloaded.'); }
    function exportSurveyPNG()  { if(surveyCompleted<=0){setSurveyFeedback('No captures yet');return;} downloadBlob('/api/report/png','survey_report_map.png'); setSurveyFeedback('PNG downloaded.'); }
    function exportSurveyCSV()  { if(surveyCompleted<=0){setSurveyFeedback('No captures yet');return;} downloadBlob('/api/report/csv','survey_report.csv'); setSurveyFeedback('CSV downloaded.'); }

    // ─── Map click ────────────────────────────────────────────────────
    document.getElementById('map-view').addEventListener('click', onMapClick);

    // ─── Init & polling ───────────────────────────────────────────────
    pollStatus(); pollPose(); pollNetworks(); pollMapperStatus(); pollSurveyStatus(); pollRFDB();
    refreshSnapshotList(); refreshMapperImage();

    setInterval(pollStatus,   2000);
    setInterval(pollPose,     1000);
    setInterval(pollNetworks, 3000);
    setInterval(pollMapperStatus, 3000);
    setInterval(pollSurveyStatus, 3000);
    setInterval(pollRFDB, 4000);
    setInterval(refreshHeatmap, 8000);   // auto-refresh heatmap every 8s
    setInterval(refreshMapperImage, 3000);
