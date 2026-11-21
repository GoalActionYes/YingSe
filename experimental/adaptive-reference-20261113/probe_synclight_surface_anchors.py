"""Test color inference with approximate correspondences fixed on ungraded photographs."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from probe_arri_sdr_renderability import transforms
from probe_arri_shared_look import converted, metrics, quantize
from probe_gros_film_b_cross import FILM
from probe_synclight_cross_view_coverage import BASE, LAMP_REPORT, PIN, ROOT, checked_json, load
from probe_synclight_lamp_edit_order import regions
from probe_target_mirror_chroma import digest, save
from spatial_render import finish_spatial


ANCHORS = Path(__file__).with_name('synclight-surface-anchors.json')
STRENGTH = .8
JITTER = (-2, 0, 2)


def patch(rgb, point, offset):
    x, y = (position + offset for position in point)
    sample = rgb[y-2:y+3, x-2:x+3]
    if sample.shape != (5, 5, 3):
        raise ValueError(f'Anchor outside image: {point}')
    return np.median(sample, axis=(0, 1))


def fitted(original, source, destination):
    source_mean = source.mean(axis=0)
    target_mean = destination.mean(axis=0)
    centered = source - source_mean
    energy = np.square(centered).sum(axis=0)
    if np.any(energy == 0):
        raise ValueError('Flat color channel in calibration')
    slope = (centered * (destination - target_mean)).sum(axis=0) / energy
    intercept = target_mean - slope * source_mean
    graded = np.rint(np.clip(original * slope + intercept, 0, 255)).astype(np.uint8)
    return finish_spatial(original, graded, STRENGTH), {
        'slope': slope.tolist(), 'intercept': intercept.tolist(),
    }


def score(original, answer, grades, masks):
    result = {}
    for name, mask in masks.items():
        row = metrics(original, answer, *grades, mask)
        row['passed'] = (row['passed'] and row['off_matches_original_mae'] <= 3
                         and row['on_matches_authored_mae'] <= 3)
        result[name] = row
    return result


def run(out):
    pinned = ROOT / 'docs/research/2026-11-18-synclight-ambient-balance-evidence.json'
    ambient = checked_json(pinned, PIN)['records']
    lamp_dir = BASE / 'synclight-lamp-edit-order-v1'
    lamp = checked_json(lamp_dir / 'results.json', LAMP_REPORT)['records']
    anchor_text = ANCHORS.read_text(encoding='utf-8')
    anchors = json.loads(anchor_text)
    cube = BASE / 'arri-second-film-assets-v1/second-film.cube'
    if digest(cube) != FILM:
        raise ValueError('Changed Film B')
    view, inverse, film = transforms(cube)

    out.mkdir()
    report = {'anchors_sha256': sha256(anchor_text.encode('utf-8')).hexdigest(), 'film_sha256': FILM, 'records': {}}
    for scene, entries in anchors.items():
        source_dir = BASE / 'synclight-ambient-balance' / scene
        target = load(source_dir / 'camera2-base.png', ambient[scene]['output_sha256']['camera2-base.png'])
        native = lamp_dir / scene
        hashes = lamp[scene]['hashes']
        references = [load(native / f'std-{state}-ref.png', hashes[f'std-{state}-ref.png'])
                      for state in ('off', 'on')]
        baseline = [load(native / f'std-{state}-output.png', hashes[f'std-{state}-output.png'])
                    for state in ('off', 'on')]
        variants = {}
        for offset in JITTER:
            colors = [[patch(frame, ref, offset) for _, ref, _ in entries] for frame in references]
            target_colors = [patch(target, point, offset) for _, _, point in entries]
            for key, source in [('cross_view', target_colors), ('same_view', colors[0])]:
                grades, fits = zip(*(fitted(target, np.asarray(source), np.asarray(values))
                                     for values in colors), strict=True)
                variants[f'{key}_{offset:+d}'] = (grades, fits)

        decoded = target.astype(np.float32) / 255
        authored = quantize(np.clip(converted(converted(converted(decoded, inverse), film), view), 0, 1))
        answer = finish_spatial(target, authored, STRENGTH)
        fraction_path = source_dir / 'lamp-fraction.npy'
        if digest(fraction_path) != ambient[scene]['fraction_sha256']:
            raise ValueError('Changed scoring regions')
        fraction = np.load(fraction_path, allow_pickle=False)
        masks = {key: mask for key, mask in regions(target, fraction).items() if mask.sum() >= 256}
        folder = out / scene
        folder.mkdir()
        outputs = {'authored.png': save(folder / 'authored.png', answer)}
        scores = {'baseline_13': score(target, answer, baseline, masks)}
        coefficients = {}
        for key, (grades, fits) in variants.items():
            scores[key] = score(target, answer, grades, masks)
            coefficients[key] = fits
            for state, grade in zip(('off', 'on'), grades, strict=True):
                name = f'{key}-{state}.png'
                outputs[name] = save(folder / name, grade)
        report['records'][scene] = {'scores': scores, 'fits': coefficients, 'outputs': outputs}
    (out / 'results.json').write_bytes((json.dumps(report, indent=2, allow_nan=False) + '\n').encode())


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
