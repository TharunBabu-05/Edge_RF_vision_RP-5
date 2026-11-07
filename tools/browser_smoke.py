"""Exercise an isolated demo server in Chromium; save visual evidence locally."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.dist'/'engineering'
OUT.mkdir(parents=True,exist_ok=True)
BASE='http://127.0.0.1:5056'
log=(OUT/'browser-server.log').open('w',encoding='utf-8')
server=subprocess.Popen([sys.executable,'app.py','--demo','--host','127.0.0.1','--port','5056','--data-dir',str(OUT/'browser-data')],cwd=ROOT,stdout=log,stderr=log)
try:
    deadline=time.monotonic()+25
    while True:
        try:
            with urlopen(BASE+'/api/overview',timeout=1) as response: json.load(response)
            break
        except OSError:
            if time.monotonic()>deadline or server.poll() is not None:raise
            time.sleep(.2)
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(channel='chrome',headless=True,args=['--enable-unsafe-swiftshader'])
        context=browser.new_context(viewport={'width':1440,'height':1050},device_scale_factor=1)
        page=context.new_page()
        errors=[]
        page.on('pageerror',lambda error: errors.append(str(error)))
        page.goto(BASE,wait_until='domcontentloaded')
        expect(page.locator('#connection')).to_have_text('Pipeline online',timeout=15000)
        expect(page.locator('#networks tr')).to_have_count(3)
        page.locator('#new-survey').click()
        expect(page.locator('#new-dialog')).to_be_visible()
        page.get_by_role('button',name='Archive & start').click()
        expect(page.locator('#toast')).to_contain_text('New survey started')
        expect(page.locator('#samples')).not_to_have_text('0',timeout=10000)
        page.locator('#record').click()
        expect(page.locator('#record-state')).to_have_text('Paused')
        page.locator('#record').click()
        expect(page.locator('#record-state')).to_have_text('Recording')
        page.locator('#search').fill('Engineering')
        expect(page.locator('#networks tr')).to_have_count(1)
        page.locator('#search').fill('')
        expect(page.locator('#toast')).to_be_hidden(timeout=7000)
        page.locator('#search').blur()
        page.screenshot(path=str(OUT/'dashboard-desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        page.locator('#scan').scroll_into_view_if_needed()
        expect(page.locator('#scan')).to_be_in_viewport()
        page.locator('#band').select_option('5')
        expect(page.locator('#networks tr')).to_have_count(1)
        page.locator('#band').select_option('')
        page.locator('h1').scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/'dashboard-mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile layout overflows viewport'
        page.set_viewport_size({'width':1440,'height':1000})
        page.goto(BASE+'/3dmap',wait_until='domcontentloaded')
        expect(page.locator('.viewer-tools')).to_be_visible(timeout=15000)
        page.locator('#btn-gyro-toggle').click()
        expect(page.locator('.orientation-status')).to_contain_text('DEMO IMU',timeout=10000)
        page.get_by_role('button',name='Recenter IMU',exact=True).click()
        with page.expect_download() as download:
            page.get_by_role('button',name='Save view ↓',exact=True).click()
        download.value.save_as(str(OUT/'exported-view.png'))
        assert (OUT/'exported-view.png').read_bytes().startswith(b'\x89PNG')
        page.screenshot(path=str(OUT/'3d-explorer.png'),full_page=True)
        page.goto(BASE+'/scene',wait_until='domcontentloaded')
        expect(page.locator('.viewer-tools')).to_be_visible()
        # Exercise the full synthetic reconstruction without touching live captures.
        for direction in ('N','E','S','W'):
            page.locator('#shot-'+direction).click()
            expect(page.locator('#status-'+direction)).to_contain_text('Captured',timeout=10000)
        page.locator('#btn-reconstruct').click()
        expect(page.locator('#loading-overlay')).to_have_class('hidden',timeout=90000)
        expect(page.locator('#status-banner')).to_contain_text('Experimental geometry',timeout=10000)
        page.locator('#sel-view-mode').select_option('spheres')
        page.locator('#sl-sphere-size').fill('0.25')
        expect(page.locator('#val-sphere-size')).to_have_text('0.25')
        page.locator('#sel-view-mode').select_option('both')
        page.get_by_role('button',name='🔍 Auto-Fit View',exact=True).click()
        with page.expect_download() as download:
            page.get_by_role('button',name='📸 Export PNG Snapshot',exact=True).click()
        download.value.save_as(str(OUT/'exported-scene.png'))
        page.locator('#btn-scene-gyro').click()
        expect(page.locator('.orientation-status')).to_contain_text('DEMO IMU',timeout=10000)
        page.screenshot(path=str(OUT/'scene-viewer.png'),full_page=True)
        page.reload(wait_until='domcontentloaded')
        expect(page.locator('#status-banner')).to_contain_text('Experimental geometry',timeout=10000)
        # Verify the preserved manual-survey dashboard still boots.
        page.goto(BASE+'/legacy',wait_until='domcontentloaded')
        expect(page.locator('body')).to_contain_text('Survey')
        browser.close()
        assert not errors,errors
        print(json.dumps({'result':'passed','page_errors':errors,'screenshots':str(OUT)},indent=2))
finally:
    server.terminate()
    server.wait(timeout=10)
    log.close()
