import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import numpy as np
from PIL import Image
from test_learned_store import FakeProvider
from workbench import make_workbench,static_assets


class ReadyFake(FakeProvider):
    def status(self):return dict(state='ready',message='test double')


class HTTPChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.provider=ReadyFake()
        self.server=make_workbench(self.root/'data',provider=self.provider,identity_tag='test')
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        rgb=np.random.default_rng(3).integers(40,200,(32,48,3),dtype=np.uint8)
        self.ids=[]
        for name,frame in [('target',rgb),('reference',rgb+10)]:
            path=self.root/(name+'.png');Image.fromarray(frame).save(path)
            self.ids.append(self.server.store.add_media(path)['id'])
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()
    def request(self,path,data=None,headers=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=15)
        opts={'Content-Type':'application/json',**(headers or {})}
        connection.request('POST' if data is not None else 'GET',path,
                           json.dumps(data) if data is not None else None,opts)
        response=connection.getresponse();body=response.read();status=response.status;connection.close()
        return status,body
    def analyze(self,engine='model'):
        status,raw=self.request('/api/analyze',dict(target_id=self.ids[0],reference_id=self.ids[1],engine=engine))
        self.assertEqual(status,200,raw)
        return json.loads(raw)['mapping_id']

    def test_served_ui_and_cross_site_rejection(self):
        status,body=self.request('/')
        self.assertEqual(status,200);self.assertIn('模型基线'.encode(),body)
        self.assertIn(b'/experiment.js',body)
        self.assertIn(b'engine:document.getElementById',static_assets()['/app.js'][0])
        self.assertEqual(self.request('/api/engine-state',headers={'Origin':'https://example.com'})[0],403)
        self.assertEqual(self.request('/api/analyze',dict(target_id=self.ids[0],reference_id=self.ids[1],engine='unknown'))[0],400)

    def test_preview_and_project_restore_engine(self):
        mapping=self.analyze()
        status,raw=self.request('/api/preview',dict(mapping_id=mapping,strength=0,interactive=True))
        self.assertEqual(status,200);preview=json.loads(raw);self.assertEqual(preview['before'],preview['after'])
        status,_=self.request('/api/projects',dict(mapping_id=mapping,strength=.5,output_settings=dict(kind='png')))
        self.assertEqual(status,201)
        status,raw=self.request('/api/projects/latest')
        self.assertEqual(status,200);self.assertEqual(json.loads(raw)['engine'],'model')
        baseline=self.analyze('baseline');self.assertNotEqual(mapping,baseline)

    def test_png_and_cube_export(self):
        mapping=self.analyze()
        for kind in ('png','cube'):
            status,raw=self.request('/api/export',dict(mapping_id=mapping,media_id=self.ids[0],strength=1,kind=kind))
            self.assertEqual(status,202,raw);identifier=json.loads(raw)['id']
            deadline=time.monotonic()+15
            while True:
                _,raw=self.request('/api/jobs/'+identifier);job=json.loads(raw)
                if job['status'] in ('done','error','cancelled'):break
                self.assertLess(time.monotonic(),deadline);time.sleep(.02)
            self.assertEqual(job['status'],'done',job)
            status,body=self.request(job['result_url']);self.assertEqual(status,200)
            self.assertTrue(body.startswith(b'\x89PNG') if kind=='png' else b'LUT_3D_SIZE 64' in body)


if __name__=='__main__':unittest.main()
