"""Independent fixture checks; no application algorithm is used as an oracle."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'outputs/jianying-color/deep/test-native-cache'
BASELINE = ROOT / 'outputs/reference-color-validation/baseline.json'

class EvidenceTests(unittest.TestCase):
    def baseline(self):
        return json.loads(BASELINE.read_text(encoding='utf-8'))

    def test_native_packet_sizes_and_float32_precision(self):
        source = next(CACHE.glob('*source_feature*.match')).read_bytes()
        target = next(CACHE.glob('*target_feature*.match')).read_bytes()
        self.assertEqual(len(source), 252)
        self.assertEqual(len(target), 228)
        # Independent frozen binary identities; never read expected values from probe JSON.
        self.assertEqual(hashlib.sha256(source).hexdigest(), '406c2c635d981054637dce73e09acd24fdb5f040430817af70dfd5727d94a8ce')
        self.assertEqual(hashlib.sha256(target).hexdigest(), '6e8d2d564cbc8cd289d6556c1118f95743ba3b36567e3843f99334d9e92b496a')
        self.assertEqual(target[53*4:54*4], bytes.fromhex('cdcccc3e'))
        self.assertEqual(struct.unpack_from('<f', target, 53*4)[0], 0.4000000059604645)
        self.assertNotEqual(struct.unpack_from('<f', target, 53*4)[0], 0.4)
        self.assertEqual(struct.unpack_from('<3f', source, 12*4), (1., 0., 0.05777696147561073))

    def test_baseline_schema_and_all_file_identities(self):
        b = self.baseline()
        for key in ('schema_version', 'files', 'exports', 'decode', 'cache_mode', 'input_binding', 'unknowns', 'direction'):
            self.assertIn(key, b)
        self.assertEqual([x['ui_strength'] for x in b['exports']], [0, 50, 100])
        self.assertFalse(b['exports'][0]['effect_enabled'])
        self.assertTrue(b['unknowns'])
        for f in b['files']:
            p = Path(f['path'])
            self.assertFalse(p.is_absolute())
            self.assertNotIn('..', p.parts)
            data = (ROOT / p).read_bytes()
            self.assertEqual(len(data), f['bytes'], f['path'])
            self.assertEqual(hashlib.sha256(data).hexdigest(), f['sha256'], f['path'])
            if p.suffix == '.png':
                from PIL import Image
                with Image.open(ROOT / p) as image:
                    self.assertEqual(list(image.size), f['size'])

    def test_independent_cached_reference_direction(self):
        from PIL import Image
        import numpy as np
        b = self.baseline()
        cached = np.asarray(Image.open(next(CACHE.glob('color_match_*.png'))).convert('RGB'), dtype=float)
        errors = {}
        for name in ('source.png', 'reference-warm.png'):
            image = Image.open(ROOT / 'outputs/jianying-color/test-kit' / name).convert('RGB')
            a = np.asarray(image.resize((708, 398), Image.Resampling.BILINEAR), dtype=float)
            errors[name] = float(np.abs(cached-a).mean())
        self.assertAlmostEqual(errors['reference-warm.png'], 0.13719965174270599, places=12)
        self.assertAlmostEqual(errors['source.png'], 11.7561725766308, places=12)
        self.assertLess(errors['reference-warm.png'], errors['source.png']/50)
        self.assertEqual(b['direction']['binding_status'], 'reference-cache-supported; native-target-pixels-unknown')

    def test_decode_at_one_second_matches_preserved_frames(self):
        from PIL import Image
        b = self.baseline()
        self.assertEqual(b['decode']['timestamp_seconds'], 1.0)
        for export in b['exports']:
            command = [str(ROOT / b['decode']['ffmpeg_path']), '-hide_banner', '-nostdin', '-loglevel', 'error', '-ss', '1.0', '-i', str(ROOT / export['path']), '-frames:v', '1', '-vf', 'scale=960:540:flags=area', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
            result = subprocess.run(command, capture_output=True, check=True)
            self.assertEqual(len(result.stdout), 960*540*3)
            self.assertEqual(result.stdout, Image.open(ROOT / export['decoded_frame']).convert('RGB').tobytes())
            self.assertEqual(export['size'], [960, 540])
            self.assertEqual(export['duration_seconds'], 8.01)

if __name__ == '__main__':
    unittest.main()
