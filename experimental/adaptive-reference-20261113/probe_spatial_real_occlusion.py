"""Run the frozen video candidate on real person-to-person occlusion."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import threading

import numpy as np
import torch

from probe_spatial_video_history import export, ffmpeg
from probe_spatial_video_stage_timing import CHECKPOINT, REFERENCE, checked
from spatial_workbench import make_spatial_workbench
from temporal_grid import TemporalGrid
from temporal_occlusion_hold import OcclusionEpisode


@dataclass(frozen=True)
class Fixture:
    sha256: str
    shape: tuple
    frames: int
    windows: dict
    areas: dict


FIXTURES = {
    'bedroom': Fixture('1be76d5d19b066e8ad7c565d88a98e11a8f8d456a707508a7aa35390def70e30', (540, 960, 3), 200,
                       {'before': (4, 20), 'occlusion': (40, 104), 'recovery': (120, 136), 'motion': (160, 176)},
                       {'wall': np.s_[115:220, 730:785], 'right': np.s_[150:280, 815:890]}),
    'street': Fixture('aeba09546f01df7d14964f809592fb0d603cdf1e593eff6b35cdc4f3e9bf73e5', (360, 640, 3), 86,
                      {'before': (0, 12), 'occlusion': (19, 65), 'recovery': (66, 79)},
                      {'left_sign': np.s_[12:42, 20:66], 'left_wall': np.s_[18:52, 80:122]}),
}


def source_frames(path, resample, fixture):
    modes = ('-vf', 'fps=30') if resample else ('-fps_mode', 'passthrough')
    raw = ffmpeg('-i', path, '-map', '0:v:0', *modes, '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1')
    pixels = np.frombuffer(raw, np.uint8).reshape(-1, *fixture.shape)
    if len(pixels) != fixture.frames:
        raise ValueError('Changed shot count')
    return pixels


def arm(root, label, temporal_class, source, ref, indices):
    server = make_spatial_workbench(root / f'{label}-store', enhanced=True)
    if server.store.engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed checkpoint')
    if temporal_class is OcclusionEpisode:
        server.manager.temporal_state = temporal_class
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        return export(f'http://127.0.0.1:{server.server_port}', source, ref, root / label, indices, temporal_class)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def measure(source, frames, decoded, fixture):
    result = {}
    for name, (start, stop) in fixture.windows.items():
        rows = {}
        for region, area in fixture.areas.items():
            inputs = np.stack([source[i][area] for i in range(start, stop)]).astype(np.int16)
            qualifier = np.abs(np.diff(inputs, axis=0)).mean(axis=(1, 2, 3)) <= 1.
            expected = np.stack([frames[i][area] for i in range(start, stop)]).astype(np.int16)
            recorded = decoded[start:stop][(slice(None), *area)].astype(np.int16)
            series = {'preencode': expected, 'decoded': recorded}
            rows[region] = {'eligible_pairs': int(qualifier.sum())}
            for state, output in series.items():
                step = np.abs(np.diff(output - inputs, axis=0)).mean(axis=(1, 2, 3))[qualifier]
                rows[region][state] = {'mean': float(step.mean()), 'maximum': float(step.max())} if len(step) else None
        result[name] = rows
    return result


def run(root, source, out, profile='bedroom'):
    torch.set_num_threads(4)
    fixture = FIXTURES[profile]
    checked(source.parent, source.name, fixture.sha256)
    ref = checked(root, *REFERENCE)
    pixels = source_frames(source, True, fixture)
    indices = [i for start, stop in fixture.windows.values() for i in range(start, stop)]
    out.mkdir()
    results = {}
    for label, temporal_class in [('baseline', TemporalGrid), ('episode', OcclusionEpisode)]:
        frames, report = arm(out, label, temporal_class, source, ref, indices)
        if report['frames'] != len(pixels):
            raise ValueError('Output frame count changed')
        decoded = source_frames(out / label / 'result.mp4', False, fixture)
        results[label] = {'export': report, 'scores': measure(pixels, frames, decoded, fixture)}
    base, candidate = results['baseline']['scores'], results['episode']['scores']
    decisions = {}
    for name in fixture.areas:
        before, after = [data['occlusion'][name] for data in (base, candidate)]
        gain = before['preencode']['maximum'] - after['preencode']['maximum'] if before['preencode'] else None
        eligible = before['eligible_pairs'] >= 32 and before['preencode']['maximum'] >= 1
        original, proposed = [scores['recovery'][name]['preencode'] for scores in (base, candidate)]
        recovery = proposed['maximum'] - original['maximum'] if original else None
        decisions[name] = {'eligible': eligible, 'drift_reduction': gain, 'recovery_drift_increase': recovery,
                           'passed': eligible and gain >= max(.25, .25*before['preencode']['maximum'])
                           and recovery is not None and recovery <= .25}
    holds = results['episode']['export']['hold_frames']
    controls = [fixture.windows[key] for key in ('before', 'motion') if key in fixture.windows]
    false_holds = [i for i in holds if any(start <= i < stop for start, stop in controls)]
    report = {'source_sha256': fixture.sha256, 'results': results, 'decisions': decisions, 'false_holds': false_holds,
              'passed': not false_holds and all(row['passed'] for row in decisions.values())}
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--profile', choices=FIXTURES, default='bedroom')
    run(**vars(parser.parse_args()))
