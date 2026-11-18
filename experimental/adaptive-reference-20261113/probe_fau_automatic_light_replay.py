"""Run the frozen two-image lab-light decision and render on new target objects."""
import argparse
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from audit_fau_two_light_intake import ARCHIVE_SHA, native_rgb
from background_intervention import target_regions
from independent_coco import ROOT, digest
from probe_fau_estimated_light import predictor
from probe_fau_reference_light_setting import SETTINGS, left_right
from probe_fau_target_light_response import design, labeled_area, render
from probe_fau_two_light_response import encode, evaluate, member
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


TARGETS = ('bymug', 'pmug', 'ymug')


def capture(archive, scene, setting):
    return native_rgb(member(archive, 'img', scene, setting), 260, 452)


def encode_capture(image):
    return encode((image.astype(np.uint32)*4).clip(0, 65535))


def run(root, out):
    archive_path = root / 'benchmark_mirf.tar.gz'
    assert digest(archive_path) == ARCHIVE_SHA
    docs = ROOT / 'docs/research'
    ids = ('exposure', 'reference-light-setting', 'target-light-response')
    sources = {name: docs / f'2026-11-18-fau-{name}-evidence.json' for name in ids}
    prior = {name: json.loads(path.read_bytes()) for name, path in sources.items()}
    assert all(record['archive_sha256'] == ARCHIVE_SHA for record in prior.values())
    response = prior['target-light-response']
    settings = prior['reference-light-setting']
    assert response['proceeded'] and settings['proceed']
    assert response['exposure_evidence_sha256'] == settings['exposure_sha256'] == digest(sources['exposure'])
    assert response['lighting_evidence_sha256'] == settings['illumination_sha256']
    coeff = np.mean([case['coefficients'] for case in response['cases'].values()], axis=0)
    centers = np.array(settings['centers']['illumination'])

    def classify(shade):
        return SETTINGS[int(np.argmin(np.linalg.norm(centers - left_right(shade), axis=1)))]

    references = {}
    source_folder = root / 'reference-light-setting'
    for key, row in settings['cases'].items():
        paths = [source_folder / f"{row['index']:02d}-{suffix}" for suffix in ('rgb.png', 'shade.npy')]
        assert [digest(path) for path in paths] == [row['rgb_sha256'], row['shade_sha256']]
        references[key] = (decode_image(paths[0]), classify(np.load(paths[1], allow_pickle=False)))
    assert len(references) == 10
    infer = predictor()
    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha == prior['exposure']['checkpoint_sha256']
    out.mkdir()

    cases, original = {}, {}
    with tarfile.open(archive_path, 'r:gz') as archive:
        for scene in TARGETS:
            raw = capture(archive, scene, SETTINGS[0])
            target = encode_capture(raw)
            shade = infer(target)[1]
            target_guess = classify(shade)
            target_file, shade_file = out / f'{scene}-target.png', out / f'{scene}-shade.npy'
            Image.fromarray(target).save(target_file)
            np.save(shade_file, shade, allow_pickle=False)
            original[scene] = raw, target
            swapped = render(target, design(shade) @ coeff)
            cases[scene] = dict(target_predicted_setting=target_guess, target_sha256=digest(target_file),
                                shade_sha256=digest(shade_file), pairs={})
            for key, (reference, reference_guess) in references.items():
                prediction = target_guess, reference_guess
                automatic = swapped if prediction == SETTINGS else target
                baseline = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
                outputs = {}
                for name, image in (('automatic', automatic), ('current13', baseline)):
                    file = out / f'{scene}-{key}-{name}.png'
                    Image.fromarray(image).save(file)
                    outputs[name] = dict(sha256=digest(file))
                cases[scene]['pairs'][key] = dict(predicted_settings=list(prediction), outputs=outputs)

        eligible_scenes = []
        for scene, row in cases.items():
            first, target = original[scene]
            second = capture(archive, scene, SETTINGS[1])
            reference = encode_capture(second)
            labels, valid = labeled_area(archive, scene, (target, reference))
            desired = finish_spatial(target, reference, .8)
            regions = target_regions(target, np.zeros(valid.shape, bool))
            for key, pair in row['pairs'].items():
                setting = key[-3:]
                correct = pair['predicted_settings'] == [SETTINGS[0], setting]
                for name, record in pair['outputs'].items():
                    image = decode_image(out / f'{scene}-{key}-{name}.png')
                    if setting == SETTINGS[0]:
                        delta = image.astype(float) - target
                        record.update(identity=bool(not delta.any()), rgb_mae=float(np.abs(delta).mean()))
                    else:
                        record['regions'] = {area: evaluate(regions[area], valid, (first, second), labels,
                                                            target, desired, image)
                                             for area in ('whole', 'neutral_proxy')}
                auto, base = (pair['outputs'][name] for name in ('automatic', 'current13'))
                if setting == SETTINGS[0]:
                    pair['passed'] = correct and auto['identity']
                    continue
                scores = auto['regions']
                neutral = scores['neutral_proxy']
                neutral_ok = True
                if neutral['eligible']:
                    metric = 'output_chroma_rms_error'
                    neutral_ok = neutral[metric] <= .85*base['regions']['neutral_proxy'][metric]
                if all(result['eligible'] for result in scores.values()) and scene not in eligible_scenes:
                    eligible_scenes.append(scene)
                pair['passed'] = correct and neutral_ok and all(score['passed'] for score in scores.values()
                                                              if score['eligible'])

    report = dict(cases=cases, eligible_scenes=eligible_scenes,
                  proceeded=len(eligible_scenes) >= 2 and all(pair['passed'] for row in cases.values()
                                                            for pair in row['pairs'].values()),
                  calibration_coefficients=coeff.tolist(), archive_sha256=ARCHIVE_SHA,
                  checkpoint_sha256=engine.checkpoint_sha, source_sha256=digest(Path(__file__)),
                  evidence_sha256={name: digest(path) for name, path in sources.items()})
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(eligible=eligible_scenes, proceeded=report['proceeded'],
                          passed={scene: sum(pair['passed'] for pair in row['pairs'].values())
                                  for scene, row in cases.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.root, args.out)
