"""Test whether local color repair needs distinct shadow and highlight offsets."""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from audit_bg_real_skin_regions import LUMA
from probe_beyond_rgb_transfer import save, write_json
from probe_local_manual_color_scope import measure
from probe_local_manual_luma_lock import locked_adjustment, montage, read_cases


def to_chroma(rgb):
    return rgb - (rgb @ LUMA)[..., None]


def median_offset(pixels):
    center = np.median(pixels, axis=0)
    return center - center @ LUMA


def fit(tones, differences):
    bounds = np.quantile(tones, [1/3, 2/3])
    bins = np.searchsorted(bounds, tones, side='right')
    groups = [bins == level for level in range(3)]
    counts = [int(group.sum()) for group in groups]
    supported = min(counts) >= 64
    centers = [float(np.median(tones[group])) for group in groups] if supported else []
    supported = supported and bool(np.all(np.diff(centers) > 0))
    offsets = [median_offset(differences[group]).tolist() for group in groups] if supported else []
    return dict(supported=supported, split_luma=bounds.tolist(), centers=centers, offsets=offsets, counts=counts)


def adjust(active, scope, anchor, original, off):
    depth = cv2.distanceTransform(np.pad(scope.astype(np.uint8), 1), cv2.DIST_L2,
                                  cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    weight = np.clip(depth / 8, 0, 1)
    yy, xx = np.indices(scope.shape)
    train = (weight == 1) & ((xx//16 + yy//16) % 2 == 0)
    held = (weight == 1) & ~train
    if min(train.sum(), held.sum()) < 128:
        raise ValueError('Insufficient frozen training geometry')
    luma = active @ LUMA
    residual = to_chroma(anchor.astype(float)) - to_chroma(active.astype(float))
    constant = median_offset(residual[train])
    band = fit(luma[train], residual[train])
    if band['supported']:
        offsets = np.array(band['offsets'])
        flexible = np.stack([np.interp(luma, band['centers'], values) for values in offsets.T], axis=-1)
    else:
        flexible = constant

    first = measure(active[held].astype(np.int16) - anchor[held])
    held_bands = np.searchsorted(band['split_luma'], luma[held], side='right')
    frames, scores = {}, {}
    for key, values in (('constant', constant), ('tonal', flexible)):
        raw, gains = locked_adjustment(active, active + weight[..., None] * values)
        output = np.rint(raw).astype(np.uint8)
        output[~scope] = active[~scope]
        np.testing.assert_array_equal(output[~scope], active[~scope])
        luma_drift = float(np.abs((output.astype(np.int16) - active) @ LUMA).max())
        if luma_drift > .5001:
            raise ValueError('Brightness drift exceeds quantization')
        delta = output[held].astype(np.int16) - anchor[held]
        report = measure(delta)
        bands = [measure(delta[held_bands == k]) if np.any(held_bands == k) else None for k in range(3)]
        row = dict(held=report, held_bands=bands, luma_drift=luma_drift,
                   compressed=float((gains[scope] < 1).mean()),
                   passed=report['chroma'] <= min(1.5, first['chroma'] * .85))
        if original is not None:
            expected = anchor[held].astype(float) - original[held]
            observed = output[held].astype(float) - off[held]
            row['style_projection'] = float(np.sum(expected * observed)/np.square(expected).sum())
        frames[key], scores[key] = output, row
    return frames, dict(held_pixels=int(held.sum()), before=first, constant_offset=constant.tolist(),
                        band_fit=band, scores=scores)


def run(out):
    out.mkdir()
    cases = {}
    for key, scope, before, anchor, original, off in read_cases():
        frames, audit = adjust(before, scope, anchor, original, off)
        hashes = {name: save(out / f'{key}-{name}.png', image) for name, image in frames.items()}
        panel = montage(out / f'{key}-comparison.png',
                        [[('Existing grade', before), ('Offline anchor', anchor)],
                         [('One color offset', frames['constant']), ('By brightness bands', frames['tonal'])]])
        cases[key] = dict(audit, outputs_sha256=hashes, panel_sha256=panel)
    write_json(out/'report.json', cases)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
