"""Real FFmpeg pipeline tests; mocked segmentation is not a quality claim."""
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from reference_color_app import media, protected_video as video
from reference_color_app.contracts import Mapping


def identity():
    axis = np.linspace(0, 1, 64, dtype=np.float32)
    cube = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
    return Mapping(np.stack([cube, cube]), {})


class RecordingSession:
    instances = []

    def __init__(self):
        self.frames = []
        self.closed = False
        self.instances.append(self)

    def process(self, original, graded, temporal=False):
        self.frames.append((original.copy(), graded.copy(), temporal))
        return graded, dict(detected=True, skin_fraction=.2, test_double=True)

    def close(self):
        self.closed = True


class ProtectedVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        def cleanup():
            # Windows may release a killed process's inherited file handles
            # a few milliseconds after wait() signals termination. Bound this
            # filesystem retry; every child is independently asserted dead.
            deadline = time.monotonic() + 2
            while True:
                try:
                    self.tmp.cleanup()
                    return
                except PermissionError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
        self.addCleanup(cleanup)
        self.root = Path(self.tmp.name)
        RecordingSession.instances.clear()
        self.fake = patch.dict(sys.modules, {'reference_color_app.skin': types.SimpleNamespace(SkinSession=RecordingSession)})
        self.fake.start()
        self.addCleanup(self.fake.stop)
        self.mapping = identity()

    def ffmpeg(self, args):
        result = subprocess.run([str(media.ffmpeg_path()), '-hide_banner', '-nostdin', '-y', *map(str, args)],
            capture_output=True, timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', 'replace')[-3000:])
        return result

    def make_video(self, audio=False, odd=False):
        source = self.root / ('audio.mkv' if audio else 'silent.mkv')
        args = ['-f', 'lavfi', '-i', 'testsrc=size=32x24:rate=12:duration=0.75']
        if odd:
            image = self.root / 'odd.png'
            Image.new('RGB', (13, 9), (180, 70, 50)).save(image)
            args = ['-loop', '1', '-framerate', '12', '-i', image]
        if audio:
            # Audio ends well before video. -shortest would lose its tail.
            args += ['-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.25']
        args += ['-t', '0.75', '-c:v', 'ffv1', '-pix_fmt', 'bgr0']
        if audio:
            args += ['-c:a', 'pcm_s16le']
        self.ffmpeg([*args, source])
        return source

    def export(self, source, cancelled=lambda: False, progress=None, process_groups=None):
        progress = [] if progress is None else progress
        process_groups = [] if process_groups is None else process_groups
        output = self.root / 'partial.mp4'
        settings = video.export_protected_video(source, output, self.mapping, .8,
            self.root / 'job', cancelled, progress.append, lambda ps: process_groups.append(list(ps)))
        return output, settings, progress, process_groups

    def test_cfr_silent_per_frame_original_mapping_and_complete_output(self):
        source = self.make_video()
        # Make an obvious synthetic change: model must still see original RGB.
        def grade(original, mapping, strength):
            return 255 - original
        with patch.object(video, 'apply_mapping', side_effect=grade):
            output, settings, progress, groups = self.export(source)
        instance = RecordingSession.instances[-1]
        self.assertTrue(instance.closed)
        self.assertGreater(len(instance.frames), 20)
        self.assertEqual(len(instance.frames), settings['frames'])
        self.assertEqual(settings['verified_frames'], settings['frames'])
        self.assertEqual(settings['fps'], 30)
        for original, graded, temporal in instance.frames:
            np.testing.assert_array_equal(graded, 255 - original)
            self.assertTrue(temporal)
        self.assertFalse(media.probe_media(output)['has_audio'])
        self.assertIn('30 fps', media.probe_media(output)['video_stream'])
        self.assertEqual(progress[0], 0)
        self.assertEqual(progress[-1], 1)
        self.assertEqual(groups[-1], [])
        self.assertTrue(all(p.poll() is not None for group in groups for p in group))

    def test_short_audio_does_not_truncate_video_and_odd_padding(self):
        source = self.make_video(audio=True, odd=True)
        output, settings, _, _ = self.export(source)
        meta = media.probe_media(output)
        self.assertTrue(meta['has_audio'])
        self.assertGreaterEqual(settings['video_duration_seconds'], .7)
        frame, _ = media.decode_frame(output, .65)
        self.assertEqual(frame.shape[:2], (10, 14))
        self.assertEqual((settings['source_display_width'], settings['source_display_height']), (13, 9))

    def test_vfr_becomes_thirty_fps_not_source_timestamps(self):
        for index, color in enumerate(('red', 'green', 'blue')):
            Image.new('RGB', (32, 24), color).save(self.root / f'{index}.png')
        manifest = self.root / 'concat.txt'
        manifest.write_text("file '0.png'\nduration 0.12\nfile '1.png'\nduration 0.32\nfile '2.png'\nduration 0.20\nfile '2.png'\n", encoding='ascii')
        source = self.root / 'vfr.mkv'
        self.ffmpeg(['-f', 'concat', '-safe', '0', '-i', manifest, '-fps_mode', 'vfr', '-c:v', 'ffv1', source])
        output, settings, _, _ = self.export(source)
        self.assertIn('VFR timestamps not preserved', settings['frame_rate_conversion'])
        self.assertIn('30 fps', media.probe_media(output)['video_stream'])
        self.assertGreaterEqual(settings['frames'], 18)
        self.assertLessEqual(settings['frames'], 22)
        colors = [rgb.mean(axis=(0, 1)).argmax() for rgb, _, _ in RecordingSession.instances[-1].frames]
        self.assertEqual(set(colors), {0, 1, 2})

    def test_rotation_uses_oriented_dimensions(self):
        raw = self.root / 'raw.mp4'
        source = self.root / 'rotated.mp4'
        self.ffmpeg(['-f', 'lavfi', '-i', 'testsrc=size=32x24:rate=10:duration=0.4', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', raw])
        self.ffmpeg(['-display_rotation:v:0', '90', '-i', raw, '-c', 'copy', source])
        expected, _ = media.decode_frame(source, 0)
        self.assertEqual(expected.shape[:2], (32, 24))
        output, settings, _, _ = self.export(source)
        actual, _ = media.decode_frame(output, 0)
        self.assertEqual(actual.shape[:2], expected.shape[:2])
        self.assertEqual((settings['width'], settings['height']), (24, 32))

    def test_ten_bit_sdr_is_explicitly_rgb8_working_output(self):
        source = self.root / 'ten-bit.mkv'
        self.ffmpeg(['-f', 'lavfi', '-i', 'testsrc=size=32x24:rate=12:duration=0.5',
            '-c:v', 'ffv1', '-pix_fmt', 'yuv444p10le', '-colorspace', 'bt709', source])
        output, settings, _, _ = self.export(source)
        self.assertTrue(settings['precision_conversion'])
        self.assertEqual(settings['input_component_bit_depths'], [10, 10, 10])
        self.assertEqual(settings['working_precision_bits'], 8)
        self.assertEqual(media.probe_media(output)['component_bit_depths'], [8, 8, 8])

    def test_physically_truncated_source_is_not_published(self):
        source = self.make_video()
        broken = self.root / 'truncated.mkv'
        data = source.read_bytes()
        broken.write_bytes(data[:int(len(data) * .7)])
        groups = []
        with self.assertRaises(ValueError):
            self.export(broken, process_groups=groups)
        self.assertEqual(groups[-1], [])
        self.assertFalse((self.root / 'partial.mp4').exists())
        self.assertTrue(all(p.poll() is not None for group in groups for p in group))

    def test_model_failure_cleans_all_children_and_partial(self):
        source = self.make_video()
        groups = []
        with patch.object(RecordingSession, 'process', side_effect=RuntimeError('model failure')):
            with self.assertRaisesRegex(RuntimeError, 'model failure'):
                self.export(source, process_groups=groups)
        self.assertFalse((self.root / 'partial.mp4').exists())
        self.assertEqual(groups[-1], [])
        self.assertTrue(all(p.poll() is not None for group in groups for p in group))

    def test_cancel_after_frames_cleans_both_processes(self):
        source = self.make_video()
        cancelled = threading.Event()
        groups = []
        def progress(value):
            if value > .1:
                cancelled.set()
        with self.assertRaises(video.ProtectedVideoCancelled):
            video.export_protected_video(source, self.root / 'partial.mp4', self.mapping,
                .8, self.root / 'job', cancelled.is_set, progress, lambda ps: groups.append(list(ps)))
        self.assertFalse((self.root / 'partial.mp4').exists())
        self.assertEqual(groups[-1], [])
        self.assertTrue(all(p.poll() is not None for group in groups for p in group))

    def test_cancel_blocked_pipe_read_terminates_children(self):
        source = self.make_video()
        real_popen = subprocess.Popen
        cancelled = threading.Event()
        groups = []
        calls = 0
        def launch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                # A real process that holds stdout open without producing RGB.
                args = ([sys.executable, '-I', '-c', 'import time; time.sleep(30)'],)
                threading.Timer(.3, cancelled.set).start()
            return real_popen(*args, **kwargs)
        started = time.monotonic()
        # Preflight decoding also uses Popen, so complete it before patching.
        rgb, meta = media.decode_frame(source, 0)
        with patch.object(video, '_preflight', return_value=(meta, {'yuv420p': (8,8,8)})), patch.object(video.subprocess, 'Popen', side_effect=launch):
            with self.assertRaises(video.ProtectedVideoCancelled):
                self.export(source, cancelled.is_set, process_groups=groups)
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(groups[-1], [])
        self.assertTrue(all(p.poll() is not None for group in groups for p in group))

    def test_truncated_pipe_frame_and_short_writes(self):
        self.assertIsNone(video._read_frame(io.BytesIO(b''), 9))
        with self.assertRaisesRegex(ValueError, '不完整'):
            video._read_frame(io.BytesIO(b'12345678'), 9)
        class ShortWriter:
            def __init__(self): self.value = bytearray()
            def write(self, value):
                size = min(5, len(value)); self.value.extend(value[:size]); return size
        writer = ShortWriter()
        pixels = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        video._write_frame(writer, pixels)
        self.assertEqual(writer.value, pixels.tobytes())

    def test_decoder_failure_after_complete_frames_is_not_success(self):
        source = self.make_video()
        rgb, meta = media.decode_frame(source, 0)
        real_popen = subprocess.Popen
        calls = 0
        def launch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                size = rgb.size
                args = ([sys.executable, '-I', '-c', f'import sys;sys.stdout.buffer.write(bytes({size}));sys.stdout.buffer.flush();sys.exit(3)'],)
            return real_popen(*args, **kwargs)
        groups = []
        with patch.object(video, '_preflight', return_value=(meta, {'yuv420p': (8,8,8)})), patch.object(video.subprocess, 'Popen', side_effect=launch):
            with self.assertRaisesRegex(ValueError, '完整解码失败'):
                self.export(source, process_groups=groups)
        self.assertFalse((self.root / 'partial.mp4').exists())
        self.assertEqual(groups[-1], [])
        self.assertTrue(all(p.poll() is not None for group in groups for p in group))

    def test_cancel_blocked_encoder_write_terminates_both(self):
        source = self.make_video()
        rgb, meta = media.decode_frame(source, 0)
        real_popen = subprocess.Popen
        cancelled = threading.Event()
        groups = []
        calls = 0
        def launch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                args = ([sys.executable, '-I', '-c', 'import time;time.sleep(30)'],)
                threading.Timer(.3, cancelled.set).start()
            return real_popen(*args, **kwargs)
        started = time.monotonic()
        with patch.object(video, '_preflight', return_value=(meta, {'yuv420p': (8,8,8)})), patch.object(video.subprocess, 'Popen', side_effect=launch):
            with self.assertRaises(video.ProtectedVideoCancelled):
                self.export(source, cancelled.is_set, process_groups=groups)
        self.assertLess(time.monotonic() - started, 10)
        self.assertFalse((self.root / 'partial.mp4').exists())
        self.assertEqual(groups[-1], [])
        self.assertTrue(all(p.poll() is not None for group in groups for p in group))

    def test_invalid_strength_and_pre_cancel_do_not_load_model(self):
        source = self.make_video()
        with self.assertRaisesRegex(ValueError, '零强度'):
            video.export_protected_video(source, self.root / 'a.mp4', self.mapping, 0,
                self.root / 'job', lambda: False, lambda p: None, lambda p: None)
        with self.assertRaises(video.ProtectedVideoCancelled):
            self.export(source, cancelled=lambda: True)
        self.assertEqual(RecordingSession.instances, [])

    def test_cancel_during_both_preflight_processes_is_bounded(self):
        source = self.make_video()
        real_popen = subprocess.Popen
        for blocked_stage in (1, 2):
            with self.subTest(stage=blocked_stage):
                calls = 0
                cancelled = threading.Event()
                groups = []
                def launch(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == blocked_stage:
                        args = ([sys.executable, '-I', '-c', 'import time;time.sleep(30)'],)
                        threading.Timer(.3, cancelled.set).start()
                    return real_popen(*args, **kwargs)
                started = time.monotonic()
                with patch.object(video.subprocess, 'Popen', side_effect=launch):
                    with self.assertRaises(video.ProtectedVideoCancelled):
                        self.export(source, cancelled.is_set, process_groups=groups)
                self.assertLess(time.monotonic() - started, 10)
                self.assertEqual(groups[-1], [])
                self.assertTrue(all(p.poll() is not None for group in groups for p in group))
                self.assertEqual(RecordingSession.instances, [])

    def test_cancel_during_final_verification_is_bounded(self):
        source = self.make_video()
        real_popen = subprocess.Popen
        cancelled = threading.Event()
        groups = []
        calls = 0
        def launch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 5:  # descriptors, preflight, decoder, encoder, verifier
                args = ([sys.executable, '-I', '-c', 'import time;time.sleep(30)'],)
                threading.Timer(.3, cancelled.set).start()
            return real_popen(*args, **kwargs)
        started = time.monotonic()
        with patch.object(video.subprocess, 'Popen', side_effect=launch):
            with self.assertRaises(video.ProtectedVideoCancelled):
                self.export(source, cancelled.is_set, process_groups=groups)
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(groups[-1], [])
        self.assertFalse((self.root / 'partial.mp4').exists())
        self.assertTrue(all(p.poll() is not None for group in groups for p in group))


if __name__ == '__main__':
    unittest.main()
