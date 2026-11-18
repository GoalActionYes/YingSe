"""Render a mask-free anchor and check a luminance-matched colored background."""
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
from probe_fau_illuminant_replay import LUMA, replay
from probe_mls_border_surface import propose
from probe_mls_light_response import GROUPS, evaluate
from probe_mls_linear_pilot import valid_pixels
from probe_mls_material_response import light_gain
from probe_mls_neutral_surface import SURFACE
from probe_mls_single_reference_material import mean_colors, render
from probe_multilum_chromaticity import differences, linear_rgb
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


def automatic(target, reference):
    if np.array_equal(target, reference):
        return target.copy(), dict(anchor=False, identity=True)
    proposals = []
    means = []
    for rgb in (target,reference):
        thumbnail, _ = propose(rgb)
        mask = np.asarray(Image.fromarray(thumbnail).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST))
        proposals.append(mask)
        if (valid_pixels(rgb, rgb)&mask).sum() < 256:
            return target.copy(), dict(anchor=False, identity=False)
        means.append(mean_colors(rgb, {'background':mask})['background'])
    proposed_target = proposals[0]
    neutral = target_regions(target, proposed_target)['neutral_proxy'] & ~proposed_target
    delta = means[1]-means[0]
    output = render(target, {'neutral':neutral}, {'neutral':delta})
    assert np.array_equal(output[~neutral], target[~neutral])
    return output, dict(anchor=True, identity=False, offset=delta.tolist(),
                        background_pixels=[int(mask.sum()) for mask in proposals], edited_pixels=int(neutral.sum()))


def run(base, out):
    docs = ROOT/'docs/research'
    previous_path = docs/'2026-11-18-mls-linear-pilot-evidence.json'
    auto_path = docs/'2026-11-18-mls-border-surface-evidence.json'
    previous, earlier = (json.loads(path.read_bytes()) for path in (previous_path,auto_path))
    assert earlier['proceeded'] and earlier['linear_sha256'] == digest(previous_path)
    cache = base/'mls-linear-pilot'
    material = base/'mls-material-identity'
    with checked(material, 'surfaces.zip') as spectra:
        table = list(csv.DictReader(io.StringIO(spectra.read('surfaces/mask_to_surface.csv').decode())))
    frames, masks_by_scene = {}, {}
    with checked(material, 'masks_preview.zip') as masks:
        for scene in SCENES:
            key, = [row['mask_colour_rgb'] for row in table
                    if int(row['scene'])==int(scene) and row['surface']==SURFACE]
            with Image.open(io.BytesIO(masks.read(f'masks_preview/{scene}.png'))) as image:
                values = np.asarray(image.resize((1157,950), Image.Resampling.NEAREST))
            masks_by_scene[scene] = (values == np.array(key.split(), dtype=int)).all(-1)
            frames[scene] = {}
            for light in LIGHTS:
                file = cache/f'{scene}_{light}.png'
                assert digest(file) == previous['scenes'][scene]['inputs_sha256'][file.name]
                frames[scene][light] = decode_image(file)

    old, new = LIGHTS
    out.mkdir()
    report = {}
    for scene in SCENES:
        donor, _ = GROUPS[scene]
        target = frames[scene][old]
        original = frames[donor][old]
        physical = masks_by_scene[donor]
        same, _ = automatic(target, target)
        np.testing.assert_array_equal(same,target)
        tinted, _ = replay(original, light_gain([.22,-.18]), physical)
        np.testing.assert_array_equal(tinted[~physical], original[~physical])
        luma = linear_rgb(original[physical]) @ LUMA
        after_luma = linear_rgb(tinted[physical]) @ LUMA
        relative_luminance_mae = float(np.abs(after_luma-luma).mean()/luma.mean())
        assert relative_luminance_mae<=.01
        ref_file = out/f'{scene}-tinted-reference.png'
        Image.fromarray(tinted).save(ref_file)
        arms = {}
        for arm, photo in (('same_setting',original), ('other_swap',frames[donor][new]), ('surface_tint',tinted)):
            result, details = automatic(target, photo)
            path = out/f'{scene}-{arm}.png'
            Image.fromarray(result).save(path)
            arms[arm] = dict(sha256=digest(path), **details)
        report[scene] = dict(arms=arms, tinted_reference_sha256=digest(ref_file),
                             reference_relative_luminance_mae=relative_luminance_mae)

    for scene, row in report.items():
        target, switched = (frames[scene][light] for light in LIGHTS)
        region = target_regions(target, masks_by_scene[scene])['neutral_proxy'] & ~masks_by_scene[scene]
        valid = valid_pixels(target, switched)
        truth = finish_spatial(target, switched, .8)
        aligned = previous['scenes'][scene]['registration'] >= .5
        for arm, result in row['arms'].items():
            predicted = decode_image(out/f'{scene}-{arm}.png')
            result['score'] = evaluate(target, predicted, truth, region, valid, aligned)
            if arm != 'surface_tint':
                path = cache/f'{scene}-{arm}-13.png'
                assert digest(path) == previous['scenes'][scene]['baseline'][arm]['sha256']
                result['baseline'] = evaluate(target, decode_image(path), truth, region, valid, aligned)
        ordinary = decode_image(out/f'{scene}-same_setting.png')
        colored = decode_image(out/f'{scene}-surface_tint.png')
        selection = valid & region
        change, _ = differences(ordinary[selection], colored[selection])
        row['extra_tint_rms'] = float(np.mean(change**2)**.5)
        pos = row['arms']['other_swap']
        neg = row['arms']['same_setting']
        row['swap_gain'] = 1-pos['score']['error']/pos['baseline']['error']
        row['ordinary_pass'] = bool(pos['score']['lamp_pass'] and neg['score']['same_setting_pass']
                                    and row['swap_gain']>=.15)
        row['tint_pass'] = bool(row['extra_tint_rms']<=.05)

    result = dict(scenes=report, proceeded=all(row['ordinary_pass'] and row['tint_pass'] for row in report.values()),
                  checkpoint_evidence_sha256=digest(previous_path), anchor_evidence_sha256=digest(auto_path),
                  source_sha256=digest(Path(__file__)))
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({scene:[row['ordinary_pass'],row['tint_pass'],row['extra_tint_rms']] for scene,row in report.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.base,args.out)
