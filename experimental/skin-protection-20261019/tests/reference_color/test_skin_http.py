"""Protection contract, bypass and atomic error reporting through real HTTP."""
import base64
import io
import json
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import test_http


class SkinHttpTests(unittest.TestCase):
    setUpClass = classmethod(test_http.HttpTests.setUpClass.__func__)
    tearDownClass = classmethod(test_http.HttpTests.tearDownClass.__func__)
    encode = staticmethod(test_http.HttpTests.encode)
    request = classmethod(test_http.HttpTests.request.__func__)
    upload = classmethod(test_http.HttpTests.upload.__func__)
    post = test_http.HttpTests.post
    wait = test_http.HttpTests.wait

    def test_flag_strict_and_cube_never_silently_loses_mask(self):
        for invalid in (None, 0, 1, 'true', [], {}):
            self.post('/api/preview', dict(mapping_id=self.mapping,strength=.8,skin_protection=invalid),400)
            self.post('/api/export',dict(mapping_id=self.mapping,media_id=self.target['id'],strength=.8,kind='png',skin_protection=invalid),400)
            self.post('/api/projects',dict(mapping_id=self.mapping,strength=.8,output_settings={'kind':'png','skin_protection':invalid}),400)
        for strength in (0,.8):
            error=self.post('/api/export',dict(mapping_id=self.mapping,media_id=self.target['id'],strength=strength,kind='cube',skin_protection=True),400)
            self.assertIn('CUBE',error['error'])

    def test_off_and_zero_bypass_skin_engine(self):
        with patch('reference_color_app.skin.protect_image',side_effect=AssertionError('Must bypass model')) as mocked:
            for flag,strength in ((False,.8),(True,0)):
                result=self.post('/api/preview',dict(mapping_id=self.mapping,strength=strength,skin_protection=flag))
                self.assertFalse(result['skin']['enabled'])
                job=self.post('/api/export',dict(mapping_id=self.mapping,media_id=self.target['id'],strength=strength,kind='png',skin_protection=flag),202)
                self.assertEqual(self.wait(job['id'])['status'],'done')
            mocked.assert_not_called()

    def test_actual_model_png_matches_preview_and_project_restore(self):
        preview=self.post('/api/preview',dict(mapping_id=self.mapping,strength=.8,skin_protection=True))
        self.assertTrue(preview['skin']['enabled'])
        self.assertIn('skin_detected',preview['skin'])
        job=self.post('/api/export',dict(mapping_id=self.mapping,media_id=self.target['id'],strength=.8,kind='png',skin_protection=True),202)
        done=self.wait(job['id']);self.assertEqual(done['status'],'done',done)
        raw=self.request('GET',done['result_url'])[2]
        after=base64.b64decode(preview['after'].split(',')[1])
        np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(raw))),np.array(Image.open(io.BytesIO(after))))
        self.post('/api/projects',dict(mapping_id=self.mapping,strength=.8,output_settings={'kind':'png','skin_protection':True}),201)
        restored=json.loads(self.request('GET','/api/projects/latest')[2])
        self.assertTrue(restored['output_settings']['skin_protection'])

    def test_model_failure_is_visible_without_publishing_result(self):
        with patch('reference_color_app.skin.protect_image',side_effect=ValueError('保护模型加载失败')):
            error=self.post('/api/preview',dict(mapping_id=self.mapping,strength=.8,skin_protection=True),400)
            self.assertIn('保护模型',error['error'])
            job=self.post('/api/export',dict(mapping_id=self.mapping,media_id=self.target['id'],strength=.8,kind='png',skin_protection=True),202)
            done=self.wait(job['id']);self.assertEqual(done['status'],'error',done)
            self.assertEqual(self.request('GET','/api/jobs/'+job['id']+'/result')[0],400)

    def test_cancel_does_not_allow_shutdown_before_worker_cleanup(self):
        entered=threading.Event();release=threading.Event()
        def blocked(original,graded):
            entered.set();release.wait(10)
            return graded,{'skin_detected':False}
        with patch('reference_color_app.skin.protect_image',side_effect=blocked):
            job=self.post('/api/export',dict(mapping_id=self.mapping,media_id=self.target['id'],strength=.8,kind='png',skin_protection=True),202)
            try:
                self.assertTrue(entered.wait(10))
                self.post('/api/jobs/'+job['id']+'/cancel',{})
                self.post('/api/stop',{'token':'test-token'},409)
            finally:release.set()
            deadline=time.monotonic()+10
            while self.server.manager.has_pending_cleanup() and time.monotonic()<deadline:time.sleep(.01)
            self.assertFalse(self.server.manager.has_pending_cleanup())
            self.assertEqual(self.request('GET','/api/jobs/'+job['id']+'/result')[0],400)
