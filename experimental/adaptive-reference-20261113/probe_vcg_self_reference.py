"""Subtract the reference image's self-grade from a frozen ordinary two-image VCG LUT."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from safetensors import safe_open

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, finish_spatial, json_to, save
from probe_vcg_first_grade import EXPECTED_SHA, prepare_models, digest_file, identity_lut, rendered


PARENT_SHA = 'e794f411b9f7bed0b8dc879c0fe59aa222ea99d414ec0f025ec88341eed2c7c9'


def subtract_bias(pair_lut, self_lut, identity):
    delta = identity + (pair_lut - self_lut)
    clipping = float(((delta < 0) | (delta > 1)).mean())
    return np.clip(delta, 0., 1.), clipping


def apply(target, lut):
    result = Image.fromarray(target).filter(ImageFilter.Color3DLUT(16, lut.ravel().tolist()))
    return finish_spatial(target, np.asarray(result), STRENGTH)


def run(source, author, checkpoint, out):
    prior = BASE / 'gros-film-b-cross-v2'
    manifest = checked_json(prior / 'results.json', BASELINE_SHA)
    groups = checked_json(Path(__file__).with_name('gros-portrait-light-regions.json'), REGIONS_SHA)
    frozen = BASE / 'vcg-first-grade-v2'
    parent = checked_json(frozen / 'results.json', PARENT_SHA)
    if parent['checkpoint_sha256'] != EXPECTED_SHA:
        raise ValueError('Baseline uses a different checkpoint')

    models, aliases, processor, scheduler, hook, official = prepare_models(source, author, checkpoint)
    direction = '01-from-02'
    target = checked_rgb(prior / '01-cold-off.png', manifest['inputs']['01-cold']['hashes']['off'])
    areas = masks(groups['01']['cold'], groups['screen'], target.shape[:2])
    identity = identity_lut()
    scores, inputs, media = {}, {}, {}
    out.mkdir()
    with safe_open(checkpoint, framework='pt', device='cpu') as weights:
        for light in LIGHTS:
            scores[light], inputs[light] = {}, {}
            for state in ('off', 'on'):
                name = f'02-{light}'
                reference = checked_rgb(prior / f'{name}-{state}.png', manifest['inputs'][name]['hashes'][state])
                stem = f'{light}-{state}'
                archived = f'ncc-{stem}'
                baseline = parent['conditions']['ncc']['hashes'][archived]
                input_lut = frozen / f'{archived}.npy'
                if digest_file(input_lut) != baseline['lut']:
                    raise ValueError('Changed base table')
                fixed = np.load(input_lut, allow_pickle=False)
                inputs[light][state] = checked_rgb(frozen / f'{archived}.png', baseline['image'])
                np.testing.assert_array_equal(apply(target, fixed), inputs[light][state])
                _, self_lut = rendered(weights, models, aliases, processor, scheduler, hook,
                                       official, reference, reference, True)
                control, _ = subtract_bias(self_lut, self_lut, identity)
                np.testing.assert_array_equal(apply(reference, control), reference)
                corrected, clipping = subtract_bias(fixed, self_lut, identity)
                result = apply(target, corrected)
                scores[light][state] = result
                hashes = {'image': save(out / f'{stem}.png', result)}
                for label, lut in [('self', self_lut), ('corrected', corrected)]:
                    path = out / f'{stem}-{label}.npy'
                    np.save(path, lut, allow_pickle=False)
                    hashes[label] = digest_file(path)
                media[stem] = {'hashes': hashes, 'clipped_fraction': clipping,
                               'self_nonidentity_mae': float(np.abs(self_lut - identity).mean() * 255)}
                print(stem, media[stem], flush=True)

    truth = checked_rgb(prior / f'{direction}-expected.png', manifest['cases'][direction]['expected_sha'])
    baseline_scores = score(target, truth, inputs, areas)
    if baseline_scores != parent['conditions']['ncc']['scores']:
        raise ValueError('Frozen baseline unexpectedly changed')
    json_to(out / 'results.json', {'parent_sha256': PARENT_SHA, 'checkpoint_sha256': EXPECTED_SHA,
                                   'baseline': baseline_scores, 'corrected': score(target, truth, scores, areas),
                                   'media': media})


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for item in ('source', 'author', 'checkpoint', 'out'):
        parser.add_argument(item, type=Path)
    run(**vars(parser.parse_args()))
