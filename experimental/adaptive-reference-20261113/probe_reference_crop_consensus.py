"""Average fixed mild reference reframings through one frozen color model."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from appearance_pairwise import appearance_forward
from appearance_replication import initialize_replication_model
from audit_consistent_reference import score_outputs
from background_intervention import target_regions
from independent_coco import digest
from pairwise_gate import response_development_gate
from probe_deep_preset_two_image import summarize
from probe_expanded_background import PARENT_SHA256, WEIGHT, WEIGHT_SHA256
from probe_internal_chroma_compression import BACKGROUND_SHA
from probe_local_edit_toggle import mask_for, save
from probe_reference_chroma_spread import BACKGROUND, BASE, GLOBAL, PARENT, decision, read
from probe_roundtrip_grade_choice import AUDIT_SHA, COHORT_SHA, ENDPOINTS, checked_json, read_rgb
from spatial_render import SpatialEngine, finish_spatial


CROPS = ((0, 0), (24, 0), (0, 24), (24, 24), (12, 12))
ARMS = ('resize', 'consensus')


def round_divide(numerator, divisor):
    base, rest = np.divmod(numerator, divisor)
    return (base + ((rest * 2 > divisor) | ((rest * 2 == divisor) & (base % 2 == 1)))).astype(np.uint8)


@torch.inference_mode()
def outputs(engine, target, reference, tensor, folder, key):
    content = tensor(target)

    def prediction(image):
        result, _ = appearance_forward(engine.model, engine.api, content, tensor(image))
        return np.rint(result[0].cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)

    original = prediction(reference)
    ref_image = Image.fromarray(reference)
    kernel = Image.Resampling.BILINEAR
    resize = prediction(np.asarray(ref_image.resize((232, 232), kernel).resize((256, 256), kernel)))
    total = np.zeros_like(original, dtype=np.int16)
    changes = []
    for x, y in CROPS:
        region = ref_image.crop((x, y, x + 232, y + 232)).resize((256, 256), kernel)
        view = prediction(np.asarray(region))
        total += view
        changes.append(float(np.abs(view.astype(np.int16) - original).mean()))
    path = folder / f'{key}-parts.npz'
    np.savez(path, total=total, resize=resize)
    hard = original.astype(np.int16)
    return original, dict(resize=round_divide(hard + resize, 2), consensus=round_divide(5 * hard + total, 10)), dict(
        view_rgb_delta=changes, crop_sensitivity=float(np.mean(changes)),
        resize_sensitivity=float(np.abs(resize.astype(np.int16) - hard).mean()), parts_sha256=digest(path))


def run(out):
    start = time.monotonic()
    np.testing.assert_array_equal(round_divide(np.array([1, 3, 5, 7]), 2), [0, 2, 2, 4])
    torch.set_num_threads(4)
    base = BASE / 'tst100k'
    cohort_dir = base / 'cohort-training-coverage-v2'
    audit = checked_json(base / 'training-coverage-audit-v2/results.json', AUDIT_SHA)
    cohort = checked_json(cohort_dir / 'cohort.json', COHORT_SHA)
    prepared = checked_json(cohort_dir / 'srgb-v1/manifest.json', audit['prepared_manifest_sha256'])
    assets = {row['original_path']: row for row in prepared['files']}
    rows = cohort['groups']['validation']
    standards = {name: checked_json(base / folder / 'scores.json', sha) for name, (folder, sha) in ENDPOINTS.items()}
    recorded = {r['index']: r for r in standards['expanded']['cases']}
    parent = checked_json(PARENT / 'results.json', PARENT_SHA256)
    old_dir = BASE / 'expanded-background-v1'
    old_record = checked_json(old_dir / 'results.json', BACKGROUND_SHA)
    old = {row['pair']: row['variants'] for row in old_record['cases']}
    if digest(WEIGHT) != WEIGHT_SHA256 or len(rows) != 64 or len(old) != 6:
        raise ValueError('Frozen model or samples changed')
    engine = SpatialEngine('multiref-v1')
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    out.mkdir(exist_ok=False)

    def load(name):
        item = assets[name]
        return read_rgb(cohort_dir / item['prepared_path'], item['sha256'], resize=True)

    def tensor(rgb):
        return (torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255).cuda()

    predictions = {name: {} for name in ARMS}
    cases = {}
    for row in rows:
        index = row['index']
        original, generated, meta = outputs(engine, load(row['content']), load(row['reference']), tensor, out, index)
        path = base / ENDPOINTS['expanded'][0] / f'{index}.png'
        if not np.array_equal(original, read_rgb(path, recorded[index]['sha256'])):
            raise ValueError('Frozen original development output changed')
        files = {}
        for name, rgb in generated.items():
            predictions[name][index] = rgb
            files[name] = save(out / f'{index}-{name}.png', rgb)
        cases[index] = dict(**meta, images_sha256=files)
    control_pictures, control_files = {}, {}
    for case in parent['cases']:
        pair = case['pair']
        target = read(case['input_file'])
        versions = {v['name']: v for v in case['variants']}
        pictures, metadata = {}, {}
        for name in ('identity', *BACKGROUND, *GLOBAL):
            source = versions[name]
            key = f'{pair}-{name}'
            baseline, results, info = outputs(engine, target, read(source['reference']), engine.analysis, out, key)
            original = finish_spatial(target, baseline, .8)
            frozen = read_rgb(old_dir / pair / f'{name}-candidate.png', old[pair][name]['output_sha256'])
            if not np.array_equal(original, frozen):
                raise ValueError('Frozen original background output changed')
            final = {name: finish_spatial(target, rgb, .8) for name, rgb in results.items()}
            pictures[name] = dict(current=read(source['methods']['adaptive']['file']), **final)
            metadata[name] = dict(**info, images_sha256={name: save(out / f'{key}-{name}.png', rgb)
                                                         for name, rgb in final.items()})
        control_pictures[pair], control_files[pair] = pictures, metadata
    frozen = out / 'predictions.json'
    frozen.write_text(json.dumps(dict(development=cases, controlled=control_files), indent=2), encoding='utf-8')

    truth = {row['index']: load(row['gt']) for row in rows}
    validation = {}
    for name, values in predictions.items():
        scored = score_outputs(rows, values, truth)
        for item in scored['cases']:
            cases[item['index']].setdefault('mae', {})[name] = item['mae']
        validation[name] = dict(mean=scored['mean'], delta=scored['delta_mean'],
                                controls={k: response_development_gate(standards['current13'], scored, standards[k])
                                          for k in ('C10', 'control')})
    by_perturbation = sorted(cases, key=lambda key: -cases[key]['crop_sensitivity'])
    ranks = {str(index): by_perturbation.index(index) + 1 for index in (3026, 27824)}
    results = {}
    for name in ARMS:
        groups = []
        for case in parent['cases']:
            pair = case['pair']
            target = read(case['input_file'])
            mask = target_regions(target, mask_for(case, pair.split('-')[0]))
            values = {key: dict(baseline=row['current'], candidate=row[name])
                      for key, row in control_pictures[pair].items()}
            groups.append(dict(pair=pair, variants=summarize(values['identity'], values, mask)))
        results[name] = decision(groups)
    elapsed = time.monotonic() - start
    if elapsed >= 120:
        raise TimeoutError('Reframing screen exceeded frozen budget')
    source = Path(__file__)
    report = dict(validation=validation, background=results, severe_sensitivity_rank=ranks, development=cases,
                  controlled=control_files, predictions_sha256=digest(frozen), seconds=elapsed,
                  cohort_sha256=COHORT_SHA, original_control_sha256=BACKGROUND_SHA, model_sha256=WEIGHT_SHA256,
                  source_sha256=digest(source), protocol_sha256=digest(source.with_name(
                      'reference-crop-consensus-protocol.md')))
    report['passed'] = (validation['consensus']['mean'] < min(standards['expanded']['mean'], validation['resize']['mean'])
                        and all(v['proceed_to_independent_evaluation']
                                for v in validation['consensus']['controls'].values())
                        and results['consensus']['proceed'])
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(mae={k: v['mean'] for k, v in validation.items()}, ranks=ranks,
                          controls=results, passed=report['passed'], seconds=elapsed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
