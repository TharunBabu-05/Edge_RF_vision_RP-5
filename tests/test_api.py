import threading
import time

import cv2
import numpy as np
import pytest

import app as application
from demo import DemoCamera, DemoPoseEstimator, DemoScanner
from rf_database import RFDatabase
from rf_field import RFFieldModel
from rf_interpolator import RFInterpolator
from signal_processor import SignalSmoother
from spatial_mapper import SpatialWifiMapper


@pytest.fixture
def setup(tmp_path):
    scanner=DemoScanner(scan_interval=.1)
    scanner.scan_once()
    runtime=application.RuntimeContext(scanner=scanner, camera=DemoCamera(320,240,15),
        field_model=RFFieldModel(frame_width=320,frame_height=240), smoother=SignalSmoother(),
        mapper=SpatialWifiMapper(), pose_estimator=DemoPoseEstimator(),
        rf_db=RFDatabase(storage_path=str(tmp_path / 'survey.sqlite3')), rf_interpolator=RFInterpolator(),
        demo=True,frame_width=320,frame_height=240)
    worker=application.StreamWorker(runtime,1)
    app=application.create_app(runtime,worker)
    app.testing=True
    yield app.test_client(),runtime,worker
    worker.stop()


def test_dashboard_and_api_contract(setup):
    client,runtime,worker=setup
    assert b'Make the invisible visible' in client.get('/').data
    assert client.get('/static/dashboard.js').status_code==200
    assert client.get('/api/overview').json['mode']=='demo'
    assert client.get('/healthz').status_code==503  # workers intentionally not started
    assert client.post('/api/recording',json={'enabled':False}).json['recording'] is False
    assert not runtime.recording
    assert client.post('/api/recording',json={'enabled':'false'}).status_code==400
    assert client.get('/api/rf/export/json').json['units']=='synthetic metres'
    assert client.get('/api/overview').headers['Cache-Control']=='no-store'


@pytest.mark.parametrize('body',[[], 'hello', {'x':'nan'}, {'x':2}, {'y':None}])
def test_bad_payloads_return_400_not_internal_errors(setup,body):
    client,_,_=setup
    assert client.post('/api/mapper/position',json=body).status_code==400


def test_cross_origin_write_rejected(setup):
    client,_,_=setup
    assert client.post('/api/rf/clear',headers={'Origin':'https://unrelated.example'}).status_code==403


def test_archive_and_new_survey_preserve_old_data(setup,tmp_path):
    client,runtime,_=setup
    runtime.rf_db.add_sample(0,0,0,runtime.scanner.get_networks())
    assert client.post('/api/rf/reset_pose').status_code==409
    result=client.post('/api/session/new',json={})
    assert result.status_code==200
    assert (tmp_path/'archives'/result.json['archived']).exists()
    assert runtime.rf_db.sample_count()==0 and runtime.recording
    history=client.get('/api/sessions').json['sessions']
    assert history[0]['name']==result.json['archived']
    assert client.get('/api/sessions/'+history[0]['name']).status_code==200
    assert client.get('/api/sessions/README.md').status_code==404


def test_restart_cannot_mix_coordinate_origins(setup):
    client,runtime,worker=setup
    runtime.rf_db.add_sample(0,0,0,runtime.scanner.get_networks())
    resumed=application.create_app(runtime,worker).test_client()
    assert resumed.post('/api/recording',json={'enabled':True}).status_code==409
    assert resumed.post('/api/session/new',json={}).status_code==200
    assert resumed.post('/api/recording',json={'enabled':True}).status_code==200


def test_floorplan_cache_refreshes_when_measurements_change(setup):
    client,runtime,_=setup
    runtime.rf_db.add_sample(0,0,0,runtime.scanner.get_networks())
    runtime.rf_db.add_sample(1,1,0,runtime.scanner.get_networks())
    first=client.get('/api/rf/floorplan').json
    runtime.rf_db.add_sample(20,20,0,runtime.scanner.get_networks())
    second=client.get('/api/rf/floorplan').json
    assert first!=second and second['sample_count']>first['sample_count']


def test_layout_validation_and_atomic_save(setup):
    client,_,_=setup
    assert client.post('/api/rooms/layout',json={'rooms':[{'id':0,'w':-1}]}).status_code==400
    room=dict(id=0,name='<script>test</script>',x=0,y=0,w=3,d=4,h=2.5)
    assert client.post('/api/rooms/layout',json={'name':'Lab','rooms':[room],'nextId':1}).status_code==200
    assert client.get('/api/rooms/layout?name=Lab').json['rooms']==[room]


def test_pipeline_produces_valid_jpeg_and_stops(setup):
    client,runtime,worker=setup
    worker.start()
    worker.start() # idempotent
    sequence=-1
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        sequence,jpeg=worker.get_latest_jpeg(sequence,.2)
        if sequence>1:break
    assert sequence>1 and runtime.pipeline_error is None
    image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
    assert image.shape==(240,320,3)
    assert client.get('/healthz').status_code==200
    worker.stop()
    assert not any(t.is_alive() for t in (worker._capture_thread,worker._process_thread,worker._encode_thread))


def test_failed_scan_blocks_manual_sampling(setup,monkeypatch):
    client,runtime,_=setup
    monkeypatch.setattr(runtime.scanner,'_run_scan_command',lambda:(False,'','No radio'))
    runtime.scanner.scan_once()
    assert client.post('/api/mapper/position',json={'x':.3,'y':.4}).status_code==409
    assert client.post('/api/survey/capture',json={}).status_code==409


def test_png_report_export(setup):
    pytest.importorskip('matplotlib')
    client,runtime,_=setup
    for i in range(4): runtime.rf_db.add_sample(i,i/2,0,runtime.scanner.get_networks())
    response=client.get('/api/rf/render3d?grid=20')
    assert response.status_code==200,response.json
    assert response.data.startswith(b'\x89PNG')


def test_csv_export_treats_ssid_as_text(setup):
    client,runtime,_=setup
    runtime.rf_db.add_sample(1,1,0,[{'ssid':'=1+1','bssid':'00:00:00:00:00:00','dbm':-55,'freq':2437}])
    assert "'=1+1" in client.get('/api/rf/export/csv').text
