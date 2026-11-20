"""Compare full-coverage RAW and sRGB edits under the same offline mask."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from probe_playraw_rendered_input_operator import (
    SOURCE_SHA, STRENGTH, accepted, checked_json, chroma, compose, digest, finish_spatial,
    json_to, panel, read, save, stats,
)
from probe_target_mirror_chroma import checked_rgb

MASK_SHA = '3261b6d8eb3865b12272abc9628549278e4d66ebdff8eabe4d266b562cb88c41'
SRGB_SHA = 'a28dc25d4ddb88cead6b24e0229130914e7fb6920f0e0710a890abc3c47d3814'
SRGB_RUNS = ('playraw-disable-import-sharpen-v2', 'playraw-disable-import-sharpen-recheck2')


def hash_pixels(frame):
    return sha256(frame.tobytes()).hexdigest()


def score(before, raw, srgb, author, active):
    def relative(frame):
        return chroma(frame) - chroma(before)
    wanted, a, b = map(relative, (author, raw, srgb))
    first, second = a - wanted, b - a
    np.testing.assert_allclose(first + second, b - wanted, rtol=0, atol=1e-12)
    return {name: dict(raw=stats(a, wanted, area), srgb=stats(b, wanted, area),
                       pipeline_mae=float(np.abs(second[area]).mean()),
                       external_mae=float(np.abs(first[area]).mean()))
            for name, area in active.items()}


def run(base, output):
    origin = base / 'playraw-mixed-light-skin-v1/replay-v2'
    source = checked_json(origin / 'results.json', SOURCE_SHA)
    before, _ = read(origin / 'off.tif', False)
    after, mask = read(origin / 'on.tif', True)
    if ([hash_pixels(frame) for frame in (before, after)] != source['rgb_sha256']
            or hash_pixels(mask) != source['mask_pixels_sha256']):
        raise ValueError('Changed original or mask')
    author = finish_spatial(before, after, STRENGTH)
    y, x = np.indices(mask.shape)
    parity = (x // 8 + y // 8) % 2
    active = dict(full=mask == 255, high=(mask >= 204) & (mask < 255),
                  middle=(mask >= 64) & (mask < 204), low=(mask > 0) & (mask < 64), zero=mask == 0,
                  fold0=(mask >= 204) & (parity != 0), fold1=(mask >= 204) & (parity != 1))
    raw_cache = base / 'playraw-mask-response-v1'
    archive = checked_json(raw_cache / 'results.json', MASK_SHA)
    output.mkdir()
    trials = []
    for i, directory in enumerate(SRGB_RUNS):
        folder = base / directory
        prior = checked_json(folder / 'results.json', SRGB_SHA)['source']
        old_tif = {state: read(folder / 'mixed' / f'{state}.tif', False)[0] for state in ('off', 'on')}
        if {key: hash_pixels(img) for key, img in old_tif.items()} != prior['rgb_sha']:
            raise ValueError('Changed sRGB endpoints')
        np.testing.assert_array_equal(old_tif['off'], before)
        raw_full = checked_rgb(raw_cache / f'uniform-{i}.png', archive['trials'][i]['image_sha256'])
        weight = mask.astype(float) / 255
        raw, raw_integrity = compose(before, dict(off=before, on=raw_full), weight)
        srgb, srgb_integrity = compose(before, old_tif, weight)
        frozen = checked_rgb(folder / 'mixed/candidate.png', prior['sha'])
        np.testing.assert_array_equal(srgb, frozen)
        rows = score(before, raw, srgb, author, active)
        supports = (all(accepted(rows[key]['raw']) for key in ('fold0', 'fold1'))
                    and all(rows[key]['pipeline_mae'] - rows[key]['external_mae'] >= 1
                            for key in ('full', 'fold0', 'fold1')))
        trials.append(dict(
            rows=rows, supports_pipeline=supports, raw=raw_integrity, srgb=srgb_integrity,
            raw_sha=save(output / f'{i}-raw.png', raw), srgb_sha=save(output / f'{i}-srgb.png', srgb),
            panel_sha=panel(output / f'{i}-comparison.png',
                            {'input': {'original': before, 'author local': author},
                             'same mask': {'RAW full edit': raw, 'sRGB full edit': srgb}})))
    json_to(output / 'results.json', dict(
        sources=[SOURCE_SHA, MASK_SHA, SRGB_SHA], script_sha=digest(Path(__file__)),
        protocol_sha=digest(Path(__file__).with_name('playraw-pipeline-mask-attribution-protocol.md')),
        trials=trials, supports_pipeline=all(row['supports_pipeline'] for row in trials)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'output'):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
