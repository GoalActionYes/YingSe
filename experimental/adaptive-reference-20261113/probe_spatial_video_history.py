"""Measure cut and occluder history in the current 13 HTTP export."""

import argparse
import json
from pathlib import Path
import subprocess
import threading
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from probe_spatial_video_stage_timing import ASSETS, CHECKPOINT, REFERENCE, checked, digest, request
from reference_color_app import media
import spatial_jobs
from spatial_workbench import make_spatial_workbench
from temporal_grid import TemporalGrid


SHAPE = (360, 640, 3)
START, STOP, CUT = 24, 36, 30
KEY_FRAMES = tuple(range(24, 41)) + tuple(range(5))


def ffmpeg(*args, data=None):
    command = [str(media.ffmpeg_path()), '-nostdin', '-v', 'error', *map(str, args)]
    return subprocess.run(command, input=data, capture_output=True, check=True, timeout=90).stdout


def decode(path, cfr=True):
    rate = ('-vf', 'fps=30') if cfr else ()
    data = ffmpeg('-i', path, '-map', '0:v:0', *rate, '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1')
    return np.frombuffer(data, np.uint8).reshape(-1, *SHAPE)


def fixture(root, output, relight=False):
    shots = {}
    for key, (name, sha, _) in ASSETS.items():
        checked(root, name, sha)
        shots[key] = decode(root / name)
    clean = shots['person']
    if len(clean) != 90 or len(shots['audio']) != 90:
        raise ValueError('Changed development clip length')
    blocked = clean.copy()
    blocked[START:STOP, :, :SHAPE[1]//2] = 40
    joined = np.concatenate([shots['audio'][:CUT], clean])
    prepared = {'clean': clean, 'cut': joined, 'occluded': blocked}
    if relight:
        bright = clean.copy()
        bright[START:STOP] = np.clip(clean[START:STOP].astype(np.int16) + 48, 0, 255)
        prepared['relit'] = bright
    paths = {}
    for name, pixels in prepared.items():
        path = output / f'{name}-input.mp4'
        ffmpeg('-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x360', '-r', '30', '-i', 'pipe:0',
               '-an', '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'fast', path, data=pixels.tobytes())
        np.testing.assert_array_equal(decode(path, False), pixels)
        paths[name] = path
    return paths


def export(url, data, ref, folder, indices, temporal_class):
    reasons = []
    holds = []
    captured = {}
    original_smooth, original_grade = temporal_class.smooth, spatial_jobs.grade_video_frame
    index = 0

    def observe(self, prepared, frame):
        result = original_smooth(self, prepared, frame)
        reason = result.diagnostics['temporal_reset_reason']
        if reason is not None:
            reasons.append([index, reason])
        if result.diagnostics.get('temporal_occlusion_hold'):
            holds.append(index)
        return result

    def mark(*args, **kwargs):
        nonlocal index
        pixels = original_grade(*args, **kwargs)
        if index in indices:
            captured[index] = pixels.copy()
        index += 1
        return pixels

    guide = request(url, '/api/media', ref, 'reference.png')['id']
    target = request(url, '/api/media', data.read_bytes(), data.name)['id']
    mapping = request(url, '/api/analyze', {'target_id': target, 'reference_id': guide})
    with patch.object(temporal_class, 'smooth', observe), patch.object(spatial_jobs, 'grade_video_frame', mark):
        job = request(url, '/api/export', {'mapping_id': mapping['mapping_id'], 'media_id': target,
                                          'strength': .8, 'kind': 'mp4'})
        end = time.monotonic() + 120
        while time.monotonic() < end:
            outcome = request(url, '/api/jobs/' + job['id'])
            if outcome['status'] not in ('queued', 'running'):
                break
            time.sleep(.05)
        if outcome['status'] != 'done':
            raise RuntimeError(outcome.get('error') or outcome['status'])
    frames = outcome['settings']['frames']
    if index != frames or set(indices) != set(captured):
        raise ValueError('Missing ordinary frame renders')
    folder.mkdir()
    pngs = {}
    for index, frame in captured.items():
        path = folder / f'{index:03d}.png'
        Image.fromarray(frame).save(path)
        pngs[str(index)] = digest(path.read_bytes())
    output = folder / 'result.mp4'
    output.write_bytes(request(url, outcome['result_url']))
    md5 = ffmpeg('-i', output, '-map', '0:v:0', '-pix_fmt', 'rgb24', '-fps_mode', 'passthrough', '-f', 'framemd5', 'pipe:1')
    if b'#tb 0: 1/30' not in md5 or sum(x.startswith(b'0,') for x in md5.splitlines()) != frames:
        raise ValueError('Unexpected encoded sequence')
    (folder / 'frames.md5').write_bytes(md5)
    report = {'frames': frames, 'resets': reasons, 'pngs': pngs, 'md5_sha256': digest(md5)}
    if temporal_class is not TemporalGrid:
        report['hold_frames'] = holds
    return captured, report


def region_diff(a, b):
    change = np.abs(a.astype(np.int16) - b)
    mae, p95 = float(change.mean()), float(np.quantile(change, .95))
    return {'mae': mae, 'p95': p95, 'passed': mae <= 1 and p95 <= 3}


def run(root, out, without_temporal=False, with_relight=False, hold=False):
    if without_temporal and hold:
        raise ValueError('Holding the grid requires temporal state')
    torch.set_num_threads(4)
    out.mkdir()
    ref = checked(root, *REFERENCE)
    paths = fixture(root, out, with_relight)
    if hold:
        from temporal_occlusion_hold import OcclusionHold
        temporal_class = OcclusionHold
    else:
        temporal_class = TemporalGrid
    all_frames, details = {}, {}
    saved = {'clean': KEY_FRAMES, 'cut': range(CUT, CUT + 5), 'occluded': range(START, STOP + 5)}
    if with_relight:
        saved['relit'] = range(START, STOP + 5)
    for name, indices in saved.items():
        server = make_spatial_workbench(out / f'{name}-store', enhanced=True)
        if server.store.engine.checkpoint_sha != CHECKPOINT:
            raise ValueError('Changed checkpoint')
        if without_temporal:
            server.manager.temporal_enabled = False
        if hold:
            server.manager.temporal_state = temporal_class
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            frames, row = export(f'http://127.0.0.1:{server.server_port}', paths[name], ref, out / name, indices, temporal_class)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()
        if row['frames'] != (90 + CUT if name == 'cut' else 90):
            raise ValueError('Frame count differs from test input')
        all_frames[name], details[name] = frames, row
    clean, cut, blocked = (all_frames[name] for name in ('clean', 'cut', 'occluded'))
    right = np.s_[:, SHAPE[1]//2:]
    checks = {'cut': [region_diff(cut[CUT+i], clean[i]) for i in range(5)],
              'visible': [region_diff(blocked[i][right], clean[i][right]) for i in range(START, STOP)],
              'uncovered': [region_diff(blocked[i], clean[i]) for i in range(STOP, STOP+5)]}
    cut_reset = any(frame == CUT and reason.startswith('scene_cut') for frame, reason in details['cut']['resets'])
    (out / 'results.json').write_text(json.dumps({'exports': details, 'checks': checks, 'cut_reset': cut_reset}, indent=2)
                                      + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--without-temporal', action='store_true')
    parser.add_argument('--with-relight', action='store_true')
    parser.add_argument('--hold', action='store_true')
    run(**vars(parser.parse_args()))
