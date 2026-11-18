"""Test cross-content reuse of the frozen RGB-only illumination estimates."""
import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import torch

from audit_fau_two_light_intake import ARCHIVE_SHA, native_rgb
from background_intervention import target_regions
from independent_coco import ROOT, digest
from probe_fau_estimated_light import estimated_ratio
from probe_fau_exposure_control import bounds_mask
from probe_fau_illuminant_replay import replay
from probe_fau_two_light_response import SCENES, evaluate, member
from probe_multilum_chromaticity import differences
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


def run(path, exposure, estimated, out):
    assert digest(path) == ARCHIVE_SHA
    docs = ROOT / 'docs/research'
    input_evidence = docs / '2026-11-18-fau-exposure-evidence.json'
    model_evidence = docs / '2026-11-18-fau-estimated-light-evidence.json'
    earlier = json.loads(input_evidence.read_bytes())
    previous = json.loads(model_evidence.read_bytes())
    assert previous['proceeded'] and previous['exposure_sha256'] == digest(input_evidence)
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    assert engine.checkpoint_sha == earlier['checkpoint_sha256']
    out.mkdir()
    photos, shades = {}, {}
    for scene in SCENES:
        photos[scene], shades[scene] = {}, {}
        for role in ('target', 'reference'):
            rgb = exposure / f'{scene}-{role}.png'
            shade = estimated / f'{scene}-{role}-shade.npy'
            assert digest(rgb) == earlier['predictions'][scene][role]
            assert digest(shade) == previous['cases'][scene]['shading'][role]
            photos[scene][role] = decode_image(rgb)
            shades[scene][role] = np.load(shade, allow_pickle=False)
    cases, generated = {}, {}
    for target_id, ref_id in (SCENES, SCENES[::-1]):
        target = photos[target_id]['target']
        mask = np.ones(target.shape[:2], bool)
        assert np.array_equal(replay(target, estimated_ratio(
            shades[target_id]['target'], shades[target_id]['target']).reshape(-1, 3), mask)[0], target)
        cases[target_id] = dict(reference_scene=ref_id)
        generated[target_id] = {}
        for scenario, role in (('swap', 'reference'), ('same_lights', 'target')):
            ratio = estimated_ratio(shades[target_id]['target'], shades[ref_id][role]).reshape(-1, 3)
            reference = photos[ref_id][role]
            outputs = dict(
                global_light=replay(target, np.exp(np.log(ratio).mean(axis=0)), mask)[0],
                spatial_light=replay(target, ratio, mask)[0],
                current13=finish_spatial(target, engine.prepare(target, reference).apply(target), .8),
            )
            generated[target_id][scenario] = outputs
            records = {}
            for method, image in outputs.items():
                filename = out / f'{target_id}-{scenario}-{method}.png'
                Image.fromarray(image).save(filename)
                records[method] = {'sha256': digest(filename)}
            cases[target_id][scenario] = records

    with tarfile.open(path, 'r:gz') as archive:
        for scene, row in cases.items():
            target, reference = (photos[scene][role] for role in ('target', 'reference'))
            data = {}
            for role in ('img', 'groundtruth', 'masks'):
                buffers = [member(archive, role, scene, direction) for direction in ('b_r', 'r_b')]
                data[role] = [native_rgb(content, 260, 452) if role == 'img'
                              else np.array(Image.open(io.BytesIO(content))) for content in buffers]
            labels = data['groundtruth']
            valid = bounds_mask((target, reference))
            for mask, label in zip(data['masks'], labels):
                valid &= (mask == 255) & (label > 0).all(axis=-1)
            regions = target_regions(target, np.zeros(valid.shape, bool))
            desired = finish_spatial(target, reference, .8)
            for name, scores in row['swap'].items():
                scores['regions'] = {
                    area: evaluate(regions[area], valid, data['img'], labels, target, desired,
                                   generated[scene]['swap'][name])
                    for area in ('whole', 'neutral_proxy')
                }
            for name, scores in row['same_lights'].items():
                scores['regions'] = {}
                for area in ('whole', 'neutral_proxy'):
                    selected = valid & regions[area]
                    change, _ = differences(target[selected], generated[scene]['same_lights'][name][selected])
                    scores['regions'][area] = dict(
                        chroma_rms=float(np.sqrt(np.mean(change**2))), pixels=int(selected.sum()))
            other = row['reference_scene']
            other_label = np.array(Image.open(io.BytesIO(member(archive, 'groundtruth', other, 'b_r'))))
            common = valid & (other_label > 0).all(axis=-1)
            ratios = np.log(other_label[common].astype(float) / labels[0][common])
            row['same_setting_label_rms'] = float(np.sqrt(np.mean((ratios[:, [0, 2]] - ratios[:, 1:2])**2)))
            candidate = row['swap']['spatial_light']['regions']
            global_neutral = row['swap']['global_light']['regions']['neutral_proxy']['output_chroma_rms_error']
            row['neutral_gain'] = 1 - candidate['neutral_proxy']['output_chroma_rms_error'] / global_neutral
            row['passed'] = row['neutral_gain'] >= .15 and all(x['passed'] for x in candidate.values()) and all(
                value['chroma_rms'] <= .05 for value in row['same_lights']['spatial_light']['regions'].values())
    report = dict(cases=cases, proceeded=all(row['passed'] for row in cases.values()),
                  archive_sha256=ARCHIVE_SHA, input_evidence_sha256=digest(input_evidence),
                  model_evidence_sha256=digest(model_evidence), checkpoint_sha256=engine.checkpoint_sha,
                  source_sha256=digest(Path(__file__)))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: dict(passed=value['passed'], neutral_gain=value['neutral_gain'],
                               same_setting_label_rms=value['same_setting_label_rms'])
                      for key, value in cases.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('path', 'exposure', 'estimated', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
