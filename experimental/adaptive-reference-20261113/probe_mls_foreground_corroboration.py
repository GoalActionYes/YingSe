"""Check foreground support for an automatic background-based grade."""
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
from probe_mls_auto_anchor_replay import automatic
from probe_mls_light_response import GROUPS, evaluate
from probe_mls_linear_pilot import valid_pixels
from probe_mls_material_response import light_gain
from probe_mls_neutral_surface import SURFACE
from probe_mls_single_reference_material import mean_colors
from probe_multilum_chromaticity import linear_rgb
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


def supported(vector, observed):
    length = float(np.linalg.norm(vector))
    magnitude = float(np.linalg.norm(observed))
    projection = float(vector @ observed / length) if length else 0.
    cosine = projection/magnitude if magnitude else 0.
    return dict(projection=projection, cosine=cosine, offset=observed.tolist(),
                supports=bool(cosine>=.5 and projection>=max(.05, .25*length)))


def run(base, out):
    docs = ROOT/'docs/research'
    previous_file = docs/'2026-11-18-mls-linear-pilot-evidence.json'
    auto_file = docs/'2026-11-18-mls-auto-anchor-replay-evidence.json'
    previous, prior_auto = (json.loads(path.read_bytes()) for path in (previous_file,auto_file))
    assert prior_auto['checkpoint_evidence_sha256'] == digest(previous_file)
    cache, materials, prior = (base/name for name in ('mls-linear-pilot','mls-material-identity','mls-auto-anchor-replay'))
    frames, shapes = {}, {}
    with checked(materials, 'surfaces.zip') as spectra:
        labels = list(csv.DictReader(io.StringIO(spectra.read('surfaces/mask_to_surface.csv').decode())))
    with checked(materials, 'masks_preview.zip') as masks:
        for scene in SCENES:
            with Image.open(io.BytesIO(masks.read(f'masks_preview/{scene}.png'))) as image:
                palette = np.asarray(image.resize((1157,950), Image.Resampling.NEAREST))
            shapes[scene] = {}
            for name in (*GROUPS[scene][1],SURFACE):
                code, = [row['mask_colour_rgb'] for row in labels if row['surface']==name and int(row['scene'])==int(scene)]
                shapes[scene][name] = (palette==np.array(code.split(),dtype=int)).all(-1)
            frames[scene] = {}
            for light in LIGHTS:
                path = cache/f'{scene}_{light}.png'
                assert digest(path) == previous['scenes'][scene]['inputs_sha256'][path.name]
                frames[scene][light] = decode_image(path)

    out.mkdir()
    original_light, switched_light = LIGHTS
    scenes = {}
    for scene in SCENES:
        donor, names = GROUPS[scene]
        target = frames[scene][original_light]
        donor_start = frames[donor][original_light]
        warm_path = prior/f'{scene}-tinted-reference.png'
        assert digest(warm_path) == prior_auto['scenes'][scene]['tinted_reference_sha256']
        donor_surface = shapes[donor][SURFACE]
        cool, _ = replay(donor_start, light_gain([0,.22]), donor_surface)
        path = out/f'{scene}-blue-reference.png'
        Image.fromarray(cool).save(path)
        donor_luma = linear_rgb(donor_start[donor_surface]) @ LUMA
        colors = dict(same_setting=donor_start, other_swap=frames[donor][switched_light],
                      surface_tint=decode_image(warm_path), blue_tint=cool)
        target_means = mean_colors(target, {name:shapes[scene][name] for name in names})
        outputs, counterfactuals = {}, {}
        for arm, ref in colors.items():
            if arm.endswith('tint'):
                np.testing.assert_array_equal(ref[~donor_surface],donor_start[~donor_surface])
                luma = linear_rgb(ref[donor_surface]) @ LUMA
                delta = float(np.abs(luma-donor_luma).mean()/donor_luma.mean())
                assert delta<=.01
                counterfactuals[arm] = delta
            before, automatic_details = automatic(target, ref)
            assert automatic_details['anchor']
            vector = np.array(automatic_details['offset'])
            reference_means = mean_colors(ref, {name:shapes[donor][name] for name in names})
            checks = {name:supported(vector, reference_means[name]-target_means[name]) for name in names}
            agreed = all(item['supports'] for item in checks.values())
            if arm!='blue_tint':
                prior_path = prior/f'{scene}-{arm}.png'
                assert digest(prior_path)==prior_auto['scenes'][scene]['arms'][arm]['sha256']
                np.testing.assert_array_equal(before,decode_image(prior_path))
            after = before if agreed else target
            file = out/f'{scene}-{arm}.png'
            Image.fromarray(after).save(file)
            outputs[arm] = dict(accepted=agreed, checks=checks, sha256=digest(file), background=vector.tolist(),
                                ungated=before)
        scenes[scene] = dict(outputs=outputs, tinted_luminance_mae=counterfactuals,
                             blue_reference_sha256=digest(path))

    for scene, item in scenes.items():
        target, other = (frames[scene][light] for light in LIGHTS)
        truth = finish_spatial(target, other, .8)
        mask = target_regions(target, shapes[scene][SURFACE])['neutral_proxy'] & ~shapes[scene][SURFACE]
        valid = valid_pixels(target,other)
        aligned = previous['scenes'][scene]['registration']>=.5
        for arm, record in item['outputs'].items():
            output = decode_image(out/f'{scene}-{arm}.png')
            record['score'] = evaluate(target,output,truth,mask,valid,aligned)
            record['uncorroborated_score'] = evaluate(target,record.pop('ungated'),truth,mask,valid,aligned)
        baseline = prior_auto['scenes'][scene]['arms']['other_swap']['baseline']['error']
        positive = item['outputs']['other_swap']['score']
        gain = 1-positive['error']/baseline
        item['gain'] = gain
        item['passed'] = bool(positive['lamp_pass'] and gain>=.15 and all(
            record['score']['same_setting_pass'] for arm,record in item['outputs'].items() if arm!='other_swap'))
    result = dict(scenes=scenes, proceeded=all(scene['passed'] for scene in scenes.values()),
                  auto_evidence_sha256=digest(auto_file), baseline_sha256=digest(previous_file),
                  source_sha256=digest(Path(__file__)))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({scene:{arm:row['accepted'] for arm,row in item['outputs'].items()}
                     for scene,item in scenes.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    run(args.base,args.out)
