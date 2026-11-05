"""Protection contracts plus offline model integration; no claimed accuracy score."""
import json
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from reference_color_app import skin

ROOT = Path(__file__).resolve().parents[2]


class SkinTests(unittest.TestCase):
    def test_lazy_session_and_import(self):
        code = "import sys; from reference_color_app.skin import SkinSession; s=SkinSession(); s.reset(); assert 'mediapipe' not in sys.modules; assert 'cv2' not in sys.modules"
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))

    def test_input_rejected_before_inference(self):
        with patch.object(skin, '_predict') as predict:
            with self.assertRaises(skin.SkinProtectionError):
                skin.protect_image(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)))
            predict.assert_not_called()

    def test_empty_mask_returns_exact_graded(self):
        original = np.zeros((8, 8, 3), np.uint8)
        graded = np.full_like(original, 93)
        with patch.object(skin, '_predict', return_value=(np.zeros((8, 8), np.float32), np.zeros((8, 8), np.uint8), {'skin_detected': False})):
            output, diag = skin.protect_image(original, graded)
        np.testing.assert_array_equal(output, graded)
        self.assertFalse(diag['skin_detected'])

    def test_original_input_and_error_not_silently_ignored(self):
        original = np.zeros((8, 8, 3), np.uint8)
        graded = np.full_like(original, 50)
        with patch.object(skin, '_predict', side_effect=skin.SkinProtectionError('test failure')) as predict:
            with self.assertRaisesRegex(skin.SkinProtectionError, 'test failure'):
                skin.protect_image(original, graded)
            self.assertIs(predict.call_args.args[0], original)

    def test_session_histories_independent_and_no_skin_resets(self):
        # Fake inference here isolates history ownership from model quality.
        import cv2
        a, b = skin.SkinSession(), skin.SkinSession()
        rgb = np.full((20, 20, 3), 100, np.uint8)
        mask = np.ones((20, 20), np.float32)
        with patch.object(skin, '_cv', cv2), patch.object(skin, '_predict', return_value=(mask, rgb[:, :, 0], {'skin_detected': True})):
            _, first = a.process(rgb, rgb, temporal=True)
            self.assertEqual(first['temporal_state'], 'new_history')
            self.assertIsNone(b._previous_mask)
            _, second = a.process(rgb, rgb, temporal=True)
            self.assertEqual(second['temporal_state'], 'motion_aligned')
            b.process(rgb, rgb, temporal=True)
            self.assertIsNot(a._previous_mask, b._previous_mask)
        with patch.object(skin, '_predict', return_value=(mask * 0, rgb[:, :, 0], {'skin_detected': False})):
            a.process(rgb, rgb, temporal=True)
        self.assertIsNone(a._previous_mask)
        self.assertIsNotNone(b._previous_mask)

    def test_chroma_only_mix_and_empty_mask(self):
        import cv2
        original = np.full((15, 11, 3), [120, 100, 90], np.uint8)
        graded = np.full_like(original, [80, 100, 130])
        with patch.object(skin, '_cv', cv2):
            np.testing.assert_array_equal(skin._blend(original, graded, np.zeros((5, 5), np.float32)), graded)
            protected = skin._blend(original, graded, np.ones((5, 5), np.float32))
        y = np.array([.299, .587, .114])
        self.assertLess(float(np.abs(protected @ y - graded @ y).max()), 1)
        def chroma(x):
            x = x.astype(float)
            return x - (x @ y)[..., None]
        self.assertLess(float(np.abs(chroma(protected) - chroma(original)).mean()), float(np.abs(chroma(graded) - chroma(original)).mean()))

    def test_scene_cut_resets_history(self):
        import cv2
        session = skin.SkinSession()
        session._previous_gray = np.zeros((20, 20), np.uint8)
        session._previous_mask = np.ones((20, 20), np.float32)
        current = np.full((20, 20), .8, np.float32)
        with patch.object(skin, '_cv', cv2):
            result, state = session._temporal(current, np.full((20, 20), 255, np.uint8))
        self.assertEqual(state, 'scene_cut')
        np.testing.assert_array_equal(current, result)

    def test_missing_model_is_explicit_and_no_failed_engine_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(skin, '_engine', None), patch.object(skin, 'MODEL_PATH', Path(directory) / 'missing.tflite'):
                with skin._engine_lock:
                    with self.assertRaisesRegex(skin.SkinProtectionError, '无法加载'):
                        skin._get_engine()
                    self.assertIsNone(skin._engine)

    def test_failed_processing_clears_clip_history(self):
        session = skin.SkinSession()
        session._previous_mask = np.ones((9, 9), np.float32)
        rgb = np.zeros((9, 9, 3), np.uint8)
        with patch.object(skin, '_predict', side_effect=RuntimeError('bad output')):
            with self.assertRaises(skin.SkinProtectionError):
                session.process(rgb, rgb, temporal=True)
        self.assertIsNone(session._previous_mask)

    def test_full_size_output_and_512_input_bound(self):
        sample = ROOT / 'tests' / 'reference_color' / 'fixtures' / 'skin-official.jpg'
        original = np.asarray(Image.open(sample).convert('RGB').resize((1025, 769)))
        before = original.copy()
        graded = np.clip(original.astype(float) * [.8, .95, 1.1] + [0, 0, 8], 0, 255).astype(np.uint8)
        output, diag = skin.protect_image(original, graded)
        self.assertTrue(diag['skin_detected'])
        self.assertEqual(output.shape, original.shape)
        self.assertEqual(max(diag['analysis_width'], diag['analysis_height']), 512)
        np.testing.assert_array_equal(original, before)

    def test_concurrent_static_requests_have_no_history_interference(self):
        from concurrent.futures import ThreadPoolExecutor
        sample = ROOT / 'tests' / 'reference_color' / 'fixtures' / 'skin-official.jpg'
        original = np.asarray(Image.open(sample).convert('RGB'))
        graded = np.full_like(original, [60, 90, 130])
        with ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(lambda _: skin.protect_image(original, graded), range(4)))
        for output, diag in values:
            self.assertTrue(diag['skin_detected'])
            np.testing.assert_array_equal(output, values[0][0])

    def test_real_offline_model_person_and_no_person(self):
        sample = ROOT / 'tests' / 'reference_color' / 'fixtures' / 'skin-official.jpg'
        self.assertTrue(sample.is_file(), 'official integration fixture must ship with developer tests')
        original = np.asarray(Image.open(sample).convert('RGB'))
        graded = np.clip(original.astype(float) * [.8, .95, 1.1] + [0, 0, 8], 0, 255).astype(np.uint8)
        output, diag = skin.protect_image(original, graded)
        self.assertTrue(diag['skin_detected'])
        self.assertEqual(output.shape, original.shape)
        self.assertTrue(np.any(output != graded))
        same, cached = skin.protect_image(original, graded)
        self.assertTrue(cached['cache_hit'])
        np.testing.assert_array_equal(same, output)
        chart = np.asarray(Image.open(ROOT / 'reference_color_app' / 'demo' / 'source.png').convert('RGB'))
        result, empty = skin.protect_image(chart, chart)
        self.assertFalse(empty['skin_detected'])
        np.testing.assert_array_equal(result, chart)
        self.assertLessEqual(max(diag['analysis_width'], diag['analysis_height']), 512)

    def test_cache_is_bounded_and_does_not_depend_on_graded(self):
        from collections import OrderedDict
        import cv2
        rgb = np.zeros((9, 9, 3), np.uint8)
        class ArrayMask:
            def numpy_view(self):
                return np.zeros((9, 9), np.float32)
        class Engine:
            calls = 0
            def segment(self, image):
                self.calls += 1
                return type('Result', (), {'category_mask': ArrayMask(), 'confidence_masks': [ArrayMask()] * 6})()
        engine = Engine()
        class MP:
            ImageFormat = type('Format', (), {'SRGB': 1})
            @staticmethod
            def Image(**kwargs): return kwargs['data']
        with patch.object(skin, '_mask_cache', OrderedDict()), patch.object(skin, '_cv', cv2), patch.object(skin, '_mp', MP), patch.object(skin, '_get_engine', return_value=engine):
            skin.protect_image(rgb, rgb)
            skin.protect_image(rgb, rgb + 1)
            self.assertEqual(engine.calls, 1)
            for value in range(1, 9):
                skin.protect_image(rgb + value, rgb)
            self.assertEqual(len(skin._mask_cache), skin.CACHE_LIMIT)


if __name__ == '__main__':
    unittest.main()
