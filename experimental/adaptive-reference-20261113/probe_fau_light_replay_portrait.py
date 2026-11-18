"""Replay the frozen lab intervention on references with no lab light swap."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import region_delta, target_regions
from independent_coco import ROOT, digest
from probe_fau_estimated_light import predictor
from probe_fau_reference_light_setting import SETTINGS, left_right
from probe_fau_target_light_response import design, render
from probe_intrinsic_intervention import OUT as SHADING, SOURCE
from reference_color_app.media import decode_image


VARIANTS = ('identity', 'identity_repeat', 'background_green_100', 'background_magenta_100',
            'whole_green', 'whole_magenta')


def locked(folder, file, expected):
    path = folder / file
    assert digest(path) == expected
    return path


def run(lab, out):
    docs = ROOT / 'docs/research'
    files = {name: docs / f'2026-11-18-fau-{name}-evidence.json'
             for name in ('automatic-light-replay', 'reference-light-setting')}
    previous, settings = (json.loads(files[name].read_bytes()) for name in files)
    assert previous['proceeded'] and settings['proceed']
    assert previous['evidence_sha256']['reference-light-setting'] == digest(files['reference-light-setting'])
    centers = np.array(settings['centers']['illumination'])
    coefficients = np.array(previous['calibration_coefficients'])

    def classify(shade):
        distances = np.linalg.norm(centers - left_right(shade), axis=1)
        return SETTINGS[int(np.argmin(distances))]

    def predict(target, target_light, reference_setting):
        decision = classify(target_light), reference_setting
        activated = decision == SETTINGS
        image = render(target, design(target_light) @ coefficients) if activated else target
        return image, dict(settings=list(decision), activated=activated)

    lab_case = previous['cases']['bymug']
    lab_input = decode_image(locked(lab, 'bymug-target.png', lab_case['target_sha256']))
    lab_light = np.load(locked(lab, 'bymug-shade.npy', lab_case['shade_sha256']), allow_pickle=False)
    for setting in SETTINGS:
        key = f'diff_{setting}'
        pair = lab_case['pairs'][key]
        img, decision = predict(lab_input, lab_light, settings['cases'][key]['variants']['illumination']['predicted'])
        expected = locked(lab, f'bymug-{key}-automatic.png', pair['outputs']['automatic']['sha256'])
        assert decision['settings'] == pair['predicted_settings']
        assert np.array_equal(img, decode_image(expected))

    source_path, shades_path = SOURCE / 'results.json', SHADING / 'results.json'
    source, shading = (json.loads(path.read_bytes()) for path in (source_path, shades_path))
    assert source['finished'] and shading['finished'] and shading['source_results_sha256'] == digest(source_path)
    shade_cases = {case['pair']: {row['name']: row for row in case['rows']} for case in shading['cases']}
    infer = predictor()
    out.mkdir()
    predictions, targets = {}, {}
    for case in source['cases']:
        key = case['pair']
        target_info = case['input_file']
        target = decode_image(locked(SOURCE, target_info['path'], target_info['sha256']))
        target_light = infer(target)[1]
        folder = out / key
        folder.mkdir()
        np.save(folder/'target-shade.npy', target_light, allow_pickle=False)
        targets[key] = target
        variants = {row['name']: row['reference'] for row in case['variants']}
        records = {}
        for name in VARIANTS:
            reference = variants[name]
            locked(SOURCE, reference['path'], reference['sha256'])
            prior = shade_cases[key][name]
            assert prior['input_sha256'] == reference['sha256']
            record = prior['prediction']
            shade = np.load(locked(SHADING, record['file'], record['sha256']), allow_pickle=False)[1]
            image, decision = predict(target, target_light, classify(shade))
            file = folder / f'{name}.png'
            Image.fromarray(image).save(file)
            records[name] = dict(**decision, image_sha256=digest(file), reference_sha256=reference['sha256'])
        predictions[key] = dict(variants=records, target_sha256=target_info['sha256'],
                                estimated_target_sha256=digest(folder/'target-shade.npy'))

    for key, case in predictions.items():
        mask_path = SOURCE / key / 'target-mask.png'
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert('L')) > 0
        target = targets[key]
        regions = target_regions(target, mask)
        baseline = decode_image(out / key / 'identity.png')
        for name, row in case['variants'].items():
            image = decode_image(out / key / f'{name}.png')
            flagged = row['activated']
            assert flagged or np.array_equal(target, image)
            row['regions_vs_identity'] = {area: region_delta(image, baseline, regions[area])
                                          for area in ('whole', 'person', 'neutral_proxy')}
        case['target_mask_sha256'] = digest(mask_path)
        case['repeat_exact'] = case['variants']['identity']['image_sha256'] == case['variants']['identity_repeat']['image_sha256']
        assert case['repeat_exact']

    activations = sum(row['activated'] for case in predictions.values() for row in case['variants'].values())
    switches = sum(case['variants'][name]['activated'] != case['variants']['identity']['activated']
                   for case in predictions.values() for name in VARIANTS[2:4])
    report = dict(cases=predictions, invalid_activations=activations, background_switches=switches,
                  proceeded=activations == 0, source_sha256=digest(Path(__file__)), lab_bridge_passed=True,
                  source_inputs_sha256=digest(source_path), reference_shading_sha256=digest(shades_path),
                  calibration_sha256={key: digest(path) for key, path in files.items()})
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(invalid_activations=activations, background_switches=switches)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lab', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.lab, args.out)
