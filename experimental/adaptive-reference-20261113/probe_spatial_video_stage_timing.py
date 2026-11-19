"""Measure and verify the existing 13 workbench video export without modifying it."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
from unittest.mock import patch
from urllib.request import Request, urlopen
from contextlib import ExitStack

import torch

from reference_color_app import media
import spatial_jobs
from spatial_render import PreparedSpatial
from spatial_workbench import make_spatial_workbench


CHECKPOINT = 'd8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'
ASSETS = {
    'person': ('skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4',
               '3687fd072b64bce70f3855c750efb0b219aaa9be5f59745a4927ad3f9a0847d9', False),
    'audio': ('adaptive-reference-20261113/spatial-workbench/course-handoff-http-20261118-v1/input-audio.mp4',
              '2517228729b5b2296799034aa1a016f17fd823ccbb88e9a7a182ceaf4c5b1514', True),
}
REFERENCE = ('adaptive-reference-20261113/spatial-workbench/course-handoff-http-20261118-v1/reference.png',
             '822af4aaf8e5ce80a62a6933a64a9516127e5cf2a05acaf5cfafaf0ba5e6346a')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(root, path, expected):
    raw = (root / path).read_bytes()
    if digest(raw) != expected:
        raise ValueError(f'Changed fixture: {path}')
    return raw


def request(url, route, body=None, filename=None):
    headers = {}
    if isinstance(body, dict):
        body = json.dumps(body).encode()
        headers['Content-Type'] = 'application/json'
    if filename:
        headers = {'Content-Type': 'application/octet-stream', 'X-Filename': filename}
    with urlopen(Request(url + route, data=body, headers=headers), timeout=90) as response:
        raw = response.read()
        return json.loads(raw) if response.headers.get_content_type() == 'application/json' else raw


class Trace:
    def __init__(self):
        self.phase = threading.local()
        self.samples = defaultdict(list)
        self.events = []
        self.finished = threading.Event()

    def time(self, key, function):
        def timed(*args, **kwargs):
            if not getattr(self.phase, 'export', False):
                return function(*args, **kwargs)
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.samples[key].append(time.perf_counter() - start)
        return timed

    def attach(self, stack, server):
        manager = server.manager
        original = manager._run

        def job(*args):
            self.phase.export = True
            try:
                return self.time('worker_total', original)(*args)
            finally:
                self.phase.export = False
                self.finished.set()

        engine = server.store.engine
        original_grid = engine.model.apply_grid

        def grid(*args, **kwargs):
            if not getattr(self.phase, 'export', False):
                return original_grid(*args, **kwargs)
            left, right = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            left.record()
            result = original_grid(*args, **kwargs)
            right.record()
            self.events.append((left, right))
            return result

        def spawn(args, **kwargs):
            decoder = kwargs.get('stdout') == subprocess.PIPE
            role = 'decoder' if decoder else 'encoder'
            proc = self.time(role + '_spawn', subprocess.Popen)(args, **kwargs)
            pipe = proc.stdout if decoder else proc.stdin
            fname = 'read' if decoder else 'write'
            setattr(pipe, fname, self.time(role + '_' + fname, getattr(pipe, fname)))
            if not decoder:
                pipe.close = self.time('encoder_close', pipe.close)
            proc.wait = self.time(role + '_wait', proc.wait)
            return proc

        class ProcessModule:
            def __getattr__(self, name):
                return spawn if name == 'Popen' else getattr(subprocess, name)

        stack.enter_context(patch.object(manager, '_run', job))
        stack.enter_context(patch.object(spatial_jobs, 'subprocess', ProcessModule()))
        stack.enter_context(patch.object(engine.model, 'apply_grid', grid))
        for host, attr, label in ((server.store, 'decode', 'initial_media_decode'),
                                   (engine, 'analysis', 'analysis_resize_and_upload'),
                                   (engine, 'prepare', 'grid_prepare_inclusive'),
                                   (PreparedSpatial, 'apply', 'spatial_render'),
                                   (spatial_jobs, 'finish_spatial', 'strength_and_adjustments'),
                                   (spatial_jobs, 'grade_video_frame', 'grade_inclusive'),
                                   (media, 'probe_media', 'media_probes_inclusive')):
            stack.enter_context(patch.object(host, attr, self.time(label, getattr(host, attr))))

    def report(self, frames):
        if not self.finished.wait(10):
            raise TimeoutError('Export worker did not finish')
        frame_stages = ('grade_inclusive', 'spatial_render', 'strength_and_adjustments')
        if len(self.events) != frames or any(len(self.samples[key]) != frames for key in frame_stages):
            raise ValueError('Unexpected fixed-LUT or missing frame timing')
        self.samples['model_grid_gpu'] = [left.elapsed_time(right) / 1000 for left, right in self.events]
        return {key: {'seconds': sum(values), 'calls': len(values)} for key, values in self.samples.items()}


def run_export(root, source, reference, instrumented):
    root.mkdir()
    server = make_spatial_workbench(root / 'store', enhanced=True)
    if server.store.engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed 13 checkpoint')
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with ExitStack() as stack:
            trace = Trace() if instrumented else None
            if trace:
                trace.attach(stack, server)
            target = request(url, '/api/media', source, 'target.mp4')['id']
            guide = request(url, '/api/media', reference, 'reference.png')['id']
            start = time.perf_counter()
            mapping = request(url, '/api/analyze', {'target_id': target, 'reference_id': guide})
            analysis = time.perf_counter() - start
            start = time.perf_counter()
            job = request(url, '/api/export', {'mapping_id': mapping['mapping_id'], 'media_id': target,
                                               'strength': .8, 'kind': 'mp4'})
            while time.perf_counter() - start <= 120:
                state = request(url, '/api/jobs/' + job['id'])
                if state['status'] not in ('queued', 'running'):
                    break
                time.sleep(.05)
            wall = time.perf_counter() - start
            if state['status'] != 'done':
                raise RuntimeError(state.get('error') or state['status'])
            final = root / 'result.mp4'
            final.write_bytes(request(url, state['result_url']))
            settings = state['settings']
            phases = trace.report(settings['frames']) if trace else None
    finally:
        server.shutdown()
        server.server_close()
        worker.join()

    start = time.perf_counter()
    arg = [str(media.ffmpeg_path()), '-v', 'error', '-i', str(final), '-map', '0:v:0', '-map', '0:a:0?',
           '-pix_fmt', 'rgb24', '-fps_mode', 'passthrough', '-f', 'framemd5', 'pipe:1']
    hashes = subprocess.run(arg, check=True, capture_output=True, timeout=60).stdout
    (root / 'frames.md5').write_bytes(hashes)
    video = sum(line.startswith(b'0,') for line in hashes.splitlines())
    audio = sum(line.startswith(b'1,') for line in hashes.splitlines())
    if video != settings['frames'] or settings['fps'] != 30:
        raise ValueError('Unexpected output sequence')
    return {'analysis_http_seconds': analysis, 'export_http_seconds': wall, 'phases': phases,
            'verification_seconds': time.perf_counter()-start, 'frames_sha256': digest(hashes),
            'video_frames': video, 'audio_frames': audio,
            'dimensions': [settings['width'], settings['height']], 'fps': settings['fps'],
            'has_audio': settings['output_metadata']['has_audio'], 'temporal': settings['temporal']}


def run(root, out):
    torch.set_num_threads(4)
    reference = checked(root, *REFERENCE)
    out.mkdir()
    rows = {}
    for name, (path, hash_value, audio) in ASSETS.items():
        source = checked(root, path, hash_value)
        runs = {arm: run_export(out / f'{name}-{arm}', source, reference, arm == 'measured')
                for arm in ('normal', 'measured')}
        baseline, measured = runs.values()
        equal = all(baseline[key] == measured[key] for key in ('frames_sha256', 'video_frames', 'audio_frames',
                                                              'dimensions', 'fps', 'has_audio', 'temporal'))
        if not equal or measured['has_audio'] != audio or bool(measured['audio_frames']) != audio:
            raise ValueError(f'Measurement changed {name} export or its expected audio')
        rows[name] = runs
    (out / 'results.json').write_text(json.dumps({'checkpoint': CHECKPOINT, 'cases': rows}, indent=2) + '\n',
                                      encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
