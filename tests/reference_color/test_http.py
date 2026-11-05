"""HTTP contracts, actual upload→mapping→preview→export, and publication failures."""
import base64
import http.client
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import quote

import numpy as np
from PIL import Image
from reference_color_app import jobs, media
from reference_color_app.server import make_server, MAX_UPLOAD


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.server = make_server(cls.root / 'data', token='test-token')
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        y, x = np.mgrid[:24, :32]
        cls.rgb = np.stack([x*7, y*10, (x+y)*4], axis=-1).astype(np.uint8)
        cls.image = cls.encode(cls.rgb)
        cls.reference_image = cls.encode(np.clip(cls.rgb.astype(int)+[12, 0, -10], 0, 255).astype(np.uint8))
        cls.target = cls.upload(cls.image, '长名字与空格 测试图.png')
        cls.reference = cls.upload(cls.reference_image, '参考图.png')
        code, _, raw = cls.request('POST', '/api/analyze', dict(target_id=cls.target['id'], reference_id=cls.reference['id'], time_seconds=0))
        if code != 200: raise AssertionError(raw)
        cls.mapping = json.loads(raw)['mapping_id']

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(); cls.temp.cleanup()

    @staticmethod
    def encode(rgb):
        stream = io.BytesIO(); Image.fromarray(rgb).save(stream, format='PNG'); return stream.getvalue()

    @classmethod
    def request(cls, method, path, body=None, headers=None, port=None):
        connection = http.client.HTTPConnection('127.0.0.1', port or cls.server.server_port, timeout=60)
        values = dict(headers or {})
        if isinstance(body, dict): body = json.dumps(body).encode(); values.setdefault('Content-Type', 'application/json')
        connection.request(method, path, body=body, headers=values)
        response = connection.getresponse(); result = (response.status, dict(response.getheaders()), response.read()); connection.close(); return result

    @classmethod
    def upload(cls, raw, name):
        code, _, result = cls.request('POST', '/api/media', raw, {'Content-Type':'application/octet-stream','X-Filename':quote(name)})
        if code != 201: raise AssertionError(result)
        return json.loads(result)

    def test_ten_bit_sdr_upload_is_explicit_compatibility_conversion(self):
        p=self.root/'upload-sdr10.mkv'
        subprocess.run([str(media.ffmpeg_path()),'-hide_banner','-nostdin','-y',
            '-f','lavfi','-i','color=red:size=16x16:rate=4:duration=1',
            '-c:v','ffv1','-pix_fmt','yuv444p10le','-colorspace','bt709',str(p)],
            check=True,capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        code,_,raw=self.request('POST','/api/media',p.read_bytes(),
            {'Content-Type':'application/octet-stream','X-Filename':'sdr10.mkv'})
        self.assertEqual(code,201,raw)
        record=json.loads(raw)
        self.assertEqual(record['metadata']['component_bit_depths'],[10,10,10])
        self.assertEqual(record['metadata']['working_precision_bits'],8)
        self.assertTrue(record['metadata']['precision_conversion'])
        result=self.post('/api/frame',dict(media_id=record['id'],time_seconds=0))
        rgb=np.array(Image.open(io.BytesIO(base64.b64decode(result['image'].split(',')[1]))))
        self.assertEqual(rgb.dtype,np.uint8)
        self.assertGreater(float(rgb[...,0].mean()),240)

    def test_twelve_bit_sdr_upload_stays_rejected(self):
        p=self.root/'upload-sdr12.mkv'
        subprocess.run([str(media.ffmpeg_path()),'-hide_banner','-nostdin','-y',
            '-f','lavfi','-i','color=red:size=16x16:rate=4:duration=1',
            '-c:v','ffv1','-pix_fmt','yuv444p12le','-colorspace','bt709',str(p)],
            check=True,capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        code,_,raw=self.request('POST','/api/media',p.read_bytes(),
            {'Content-Type':'application/octet-stream','X-Filename':'sdr12.mkv'})
        self.assertEqual(code,400,raw)
        self.assertIn('暂不支持此视频位深',json.loads(raw)['error'])

    def post(self, path, body, expected=200):
        code, _, raw = self.request('POST', path, body); self.assertEqual(code, expected, raw); return json.loads(raw)

    def wait(self, identifier):
        deadline = time.monotonic()+30
        while time.monotonic()<deadline:
            code, _, raw = self.request('GET', '/api/jobs/'+identifier); self.assertEqual(code, 200, raw)
            result = json.loads(raw)
            if result['status'] in ('done', 'error', 'cancelled'): return result
            time.sleep(.02)
        self.fail('Export timed out')

    def test_actual_image_workflow_preserves_preview_export_and_zero(self):
        self.assertNotIn('path', self.target)
        before = self.post('/api/frame', {'media_id':self.target['id']})
        np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(base64.b64decode(before['image'].split(',')[1])))), self.rgb)
        for strength in (0, .63):
            preview = self.post('/api/preview', dict(mapping_id=self.mapping, strength=strength))
            job = self.post('/api/export', dict(mapping_id=self.mapping, media_id=self.target['id'], strength=strength, kind='png'), 202)
            completed = self.wait(job['id']); self.assertEqual(completed['status'], 'done', completed)
            code, headers, raw = self.request('GET', completed['result_url']); self.assertEqual(code, 200)
            result = np.array(Image.open(io.BytesIO(raw)))
            rendered = np.array(Image.open(io.BytesIO(base64.b64decode(preview['after'].split(',')[1]))))
            np.testing.assert_array_equal(result, rendered); self.assertEqual(result.shape, self.rgb.shape)
            if strength == 0: np.testing.assert_array_equal(result, self.rgb)
            self.assertIn("filename*=UTF-8''", headers['Content-Disposition'])
            self.assertNotIn('\r', headers['Content-Disposition'])
            self.assertEqual(self.request('GET', completed['report_url'])[0], 200)
        code, _, raw = self.request('GET', '/api/diagnostics/'+self.mapping)
        self.assertEqual(code, 200); self.assertIn('algorithm_fingerprint', json.loads(raw))

    def test_project_restore_validates_owned_mapping_across_restart(self):
        # UI's serialized writer can finish an older snapshot before submitting
        # the latest; a new server must restore that final strength AND format.
        self.post('/api/projects', dict(mapping_id=self.mapping, strength=.2, output_settings={'kind':'png'}), 201)
        saved = self.post('/api/projects', dict(mapping_id=self.mapping, strength=.8, output_settings={'kind':'cube', 'clone_enabled':False}), 201)
        second = make_server(self.root/'data'); thread = threading.Thread(target=second.serve_forever, daemon=True); thread.start()
        try:
            code, _, raw = self.request('GET', '/api/projects/latest', port=second.server_port)
            self.assertEqual(code, 200, raw); restored=json.loads(raw)
            self.assertEqual(restored['id'], saved['project_id']); self.assertAlmostEqual(restored['strength'], .8, places=6)
            self.assertEqual(restored['output_settings'], {'kind':'cube', 'clone_enabled':False})
            self.assertEqual(restored['target']['id'], self.target['id']); self.assertNotIn('path', restored['target'])
        finally: second.shutdown(); second.server_close(); thread.join()

    def test_clone_enable_setting_is_optional_and_strictly_boolean(self):
        for settings in ({'kind':'png'}, {'kind':'mp4', 'clone_enabled':True}):
            self.post('/api/projects', dict(mapping_id=self.mapping, strength=.8, output_settings=settings), 201)
            code, _, raw = self.request('GET', '/api/projects/latest')
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(raw)['output_settings'], settings)
        for invalid in (0, 1, None, 'false', [], {}):
            self.post('/api/projects', dict(mapping_id=self.mapping, strength=.8,
                output_settings={'kind':'png', 'clone_enabled':invalid}), 400)
        self.post('/api/projects', dict(mapping_id=self.mapping, strength=.8,
            output_settings={'kind':'png', 'unexpected':True}), 400)

    def test_ids_traversal_strict_json_and_origin(self):
        for path in ('/../storage.py', '/%2e%2e/storage.py', '/api/media/../../x', '/api/jobs/not-an-id/result', '/api/media/'+'0'*32):
            code, _, raw = self.request('GET', path); self.assertIn(code, (400,404), raw); self.assertIn('error', json.loads(raw))
        for body in (dict(mapping_id='../x',strength=.2),dict(mapping_id=self.mapping,strength=True),dict(mapping_id=self.mapping,strength=.5,path='C:/private')):
            self.post('/api/preview', body, 400)
        for headers in ({'Origin':'https://evil.example'}, {'Host':'evil.example'}, {'Sec-Fetch-Site':'cross-site'}):
            self.assertEqual(self.request('GET','/api/health',headers=headers)[0],403)
        self.assertEqual(self.request('OPTIONS','/api/media')[0],403)
        self.assertEqual(self.request('POST','/api/preview',b'{}',{'Content-Type':'text/plain'})[0],415)
        self.assertEqual(self.request('POST','/api/preview',b'{}',{'Content-Type':'application/json','Content-Length':'65537'})[0],413)
        self.assertEqual(self.request('POST','/api/preview',b'{"mapping_id":"x","strength":NaN}',{'Content-Type':'application/json'})[0],400)

    def test_upload_limits_and_true_decode(self):
        self.assertEqual(self.request('POST','/api/media',b'x',{'Content-Type':'application/octet-stream','Content-Length':str(MAX_UPLOAD+1)})[0],413)
        code, _, raw = self.request('POST','/api/media',b'not an image',{'Content-Type':'application/octet-stream','X-Filename':'fake.png'})
        self.assertEqual(code,400,raw); self.assertEqual(list((self.root/'data').glob('upload-*.tmp')),[])
        self.assertEqual(self.request('POST','/api/media',self.image,{'Content-Type':'image/png'})[0],415)
        self.post('/api/analyze',dict(target_id=self.target['id'],reference_id=self.reference['id'],time_seconds=-1),400)

    def test_stale_mapping_and_mismatched_target_are_rejected(self):
        record = self.server.store._record('mappings',self.mapping); original=record.read_bytes()
        try:
            data=json.loads(original);data['cache_key']='0'*64;record.write_text(json.dumps(data),encoding='utf-8')
            self.post('/api/preview',dict(mapping_id=self.mapping,strength=.5),400)
            self.post('/api/export',dict(mapping_id=self.mapping,media_id=self.target['id'],strength=.5,kind='png'),400)
        finally:record.write_bytes(original)
        self.post('/api/export',dict(mapping_id=self.mapping,media_id=self.reference['id'],strength=.5,kind='png'),400)
        self.post('/api/export',dict(mapping_id=self.mapping,media_id=self.target['id'],strength=.5,kind='mp4'),400)

    def test_video_raw_upload_frame_reference_validation_and_export(self):
        video_path=self.root/'source.mp4'
        subprocess.run([str(media.ffmpeg_path()),'-hide_banner','-loglevel','error','-y','-f','lavfi','-i','testsrc2=size=32x24:rate=3','-t','1','-c:v','libx264','-pix_fmt','yuv420p',str(video_path)],check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        video=self.upload(video_path.read_bytes(),'真实上传视频.mp4');self.assertEqual(video['kind'],'video')
        preview=self.post('/api/frame',{'media_id':video['id'],'time_seconds':.34})
        self.assertGreaterEqual(preview['metadata']['actual_frame_pts_seconds'],.34)
        self.post('/api/frame',{'media_id':video['id'],'time_seconds':2},400)
        self.post('/api/analyze',dict(target_id=self.target['id'],reference_id=video['id']),400)
        mapping=self.post('/api/analyze',dict(target_id=video['id'],reference_id=self.reference['id'],time_seconds=.34))['mapping_id']
        job=self.post('/api/export',dict(mapping_id=mapping,media_id=video['id'],strength=.5,kind='mp4'),202)
        completed=self.wait(job['id']);self.assertEqual(completed['status'],'done',completed)
        code,headers,raw=self.request('GET',completed['result_url']);self.assertEqual(code,200);self.assertEqual(headers['Content-Type'],'video/mp4')
        output=self.root/'download.mp4';output.write_bytes(raw);self.assertEqual(media.probe_media(output)['width'],32)

    def test_error_and_cancelled_jobs_never_publish_downloads(self):
        with patch.object(jobs, 'apply_mapping', side_effect=ValueError('controlled export failure')):
            job=self.post('/api/export',dict(mapping_id=self.mapping,media_id=self.target['id'],strength=.5,kind='png'),202)
            self.assertEqual(self.wait(job['id'])['status'],'error')
        for artifact in ('result','report'):
            self.assertEqual(self.request('GET',f"/api/jobs/{job['id']}/{artifact}")[0],400)
        entered=threading.Event(); release=threading.Event(); real=jobs.apply_mapping
        def blocking(*args): entered.set();release.wait(20);return real(*args)
        with patch.object(jobs,'apply_mapping',side_effect=blocking):
            job=self.post('/api/export',dict(mapping_id=self.mapping,media_id=self.target['id'],strength=.5,kind='png'),202)
            self.assertTrue(entered.wait(10))
            try:
                self.assertEqual(self.request('GET',f"/api/jobs/{job['id']}/result")[0],400)
                self.post('/api/stop',{'token':'test-token'},409)
                cancelled=self.post(f"/api/jobs/{job['id']}/cancel",{});self.assertEqual(cancelled['status'],'cancelled')
            finally:release.set()
        self.assertNotIn('result_url',self.wait(job['id']))
        self.assertEqual(self.request('GET',f"/api/jobs/{job['id']}/result")[0],400)
        self.post('/api/stop',{'token':'wrong'},403)

    def test_cli_ready_file_health_and_token_stop(self):
        ready=self.root/'ready.json'
        process=subprocess.Popen([sys.executable,'-m','reference_color_app.server','--port','0','--data-dir',str(self.root/'cli-data'),'--ready-file',str(ready)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            deadline=time.monotonic()+20
            while not ready.exists() and time.monotonic()<deadline:time.sleep(.05)
            self.assertTrue(ready.exists()); data=json.loads(ready.read_text(encoding='utf-8'))
            self.assertEqual(data['pid'],process.pid);port=int(data['url'].rsplit(':',1)[1])
            self.assertEqual(json.loads(self.request('GET','/api/health',port=port)[2])['app'],'reference-color-app')
            self.assertEqual(self.request('POST','/api/stop',{'token':data['token']},port=port)[0],200)
            self.assertEqual(process.wait(timeout=10),0)
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=10)
            process.communicate()


if __name__ == '__main__': unittest.main()
