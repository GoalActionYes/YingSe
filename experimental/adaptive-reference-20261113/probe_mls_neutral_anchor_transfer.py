"""Test whether a white-background light estimate transfers to other neutrals."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_mls_material_identity import checked
from background_intervention import target_regions
from fetch_mls_linear_pilot import LIGHTS, SCENES
from independent_coco import ROOT, digest
from probe_mls_light_response import evaluate
from probe_mls_linear_pilot import valid_pixels
from probe_mls_neutral_surface import SURFACE
from probe_mls_single_reference_material import render
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


def run(cache, materials, out):
    docs = ROOT/'docs/research'
    evidence = docs/'2026-11-18-mls-neutral-surface-evidence.json'
    baseline = docs/'2026-11-18-mls-linear-pilot-evidence.json'
    anchor, previous = (json.loads(path.read_bytes()) for path in (evidence, baseline))
    assert anchor['proceeded'] and anchor['baseline_evidence_sha256'] == digest(baseline)
    with checked(materials, 'surfaces.zip') as spectra:
        rows = list(csv.DictReader(io.StringIO(spectra.read('surfaces/mask_to_surface.csv').decode())))
    out.mkdir()
    scenes, pending = {}, {}
    with checked(materials, 'masks_preview.zip') as masks:
        for scene in SCENES:
            file = cache/f'{scene}_{LIGHTS[0]}.png'
            assert digest(file) == previous['scenes'][scene]['inputs_sha256'][file.name]
            target = decode_image(file)
            color, = [row['mask_colour_rgb'] for row in rows
                      if int(row['scene']) == int(scene) and row['surface'] == SURFACE]
            with Image.open(io.BytesIO(masks.read(f'masks_preview/{scene}.png'))) as mask:
                palette = np.asarray(mask.resize((1157,950), Image.Resampling.NEAREST))
            white = (palette == np.array(color.split(), dtype=int)).all(axis=-1)
            area = target_regions(target, white)['neutral_proxy'] & ~white
            assert not (area & white).any()
            np.testing.assert_array_equal(render(target, {'neutral':area}, {'neutral':np.zeros(2)}), target)
            arms = {}
            for arm in ('same_setting', 'other_swap'):
                vector = np.array(anchor['scenes'][scene]['arms'][arm]['offset'])
                result = render(target, {'neutral':area}, {'neutral':vector})
                path = out/f'{scene}-{arm}.png'
                Image.fromarray(result).save(path)
                arms[arm] = dict(sha256=digest(path), offset=vector.tolist())
            pending[scene] = target, area
            scenes[scene] = dict(area=int(area.sum()), arms=arms)
    for scene, row in scenes.items():
        target, area = pending[scene]
        file = cache/f'{scene}_{LIGHTS[1]}.png'
        assert digest(file) == previous['scenes'][scene]['inputs_sha256'][file.name]
        switched = decode_image(file)
        valid = valid_pixels(target, switched)
        aligned = previous['scenes'][scene]['registration'] >= .5
        expected = finish_spatial(target, switched, .8)
        for arm, scores in row['arms'].items():
            path = cache/f'{scene}-{arm}-13.png'
            assert digest(path) == previous['scenes'][scene]['baseline'][arm]['sha256']
            scores['candidate'] = evaluate(target, decode_image(out/f'{scene}-{arm}.png'), expected,
                                           area, valid, aligned)
            scores['current13'] = evaluate(target, decode_image(path), expected, area, valid, aligned)
        same = row['arms']['same_setting']['candidate']
        positive = row['arms']['other_swap']
        candidate, original = positive['candidate'], positive['current13']
        gain = 1-candidate['error']/original['error'] if candidate['eligible'] and original['error'] else None
        row['gate'] = dict(gain=gain, passed=bool(gain is not None and gain>=.15
                                                  and same['same_setting_pass'] and candidate['lamp_pass']))
    report = dict(scenes=scenes, proceeded=all(row['gate']['passed'] for row in scenes.values()),
                  source_sha256=digest(Path(__file__)), anchor_sha256=digest(evidence),
                  baseline_sha256=digest(baseline))
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({scene:row['gate'] for scene,row in scenes.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'materials', 'out'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.cache, args.materials, args.out)
