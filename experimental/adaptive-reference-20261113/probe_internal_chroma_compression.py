"""Replace channel clipping by a neutral-axis contraction before post-guide."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from appearance_pairwise import appearance_grid
from appearance_replication import initialize_replication_model
from audit_consistent_reference import score_outputs
from background_intervention import target_regions
from independent_coco import ROOT, digest
from internal_affine_stage import internal_affine
from pairwise_gate import response_development_gate
from probe_deep_preset_two_image import summarize
from probe_expanded_background import PARENT_SHA256, WEIGHT, WEIGHT_SHA256
from probe_local_edit_toggle import mask_for, save
from probe_reference_chroma_spread import BACKGROUND, BASE, GLOBAL, PARENT, decision, read
from probe_roundtrip_grade_choice import AUDIT_SHA, COHORT_SHA, ENDPOINTS, checked_json, read_rgb
from spatial_render import SpatialEngine, finish_spatial


BACKGROUND_SHA = 'ec7e90fc2559904347b06ad71f55744903ef924887f1b136caf2ed63c5785045'


def compress(rgb):
    outside = (rgb.abs() > 1).any(dim=1, keepdim=True)
    mean = rgb.mean(dim=1, keepdim=True)
    center = mean.clamp(-1, 1)
    offset = rgb - mean
    headroom = torch.where(offset > 0, 1 - center, center + 1)
    limits = headroom / torch.where(offset == 0, torch.ones_like(offset), offset.abs())
    limits = torch.where(offset == 0, torch.ones_like(limits), limits)
    factor = limits.amin(dim=1, keepdim=True).clamp(0, 1)
    adjusted = (center + factor * offset).clamp(-1, 1)
    return torch.where(outside, adjusted, rgb)


def synthetic_check():
    image = torch.tensor([[[-1., 0., 1.], [2., 0., -1.5], [3., 3., 3.], [-3., -3., -3.], [1.5, 0, .3]]],
                         dtype=torch.float64).transpose(1, 2).unsqueeze(-1)
    output = compress(image)
    assert torch.equal(image[:, :, :1], output[:, :, :1])
    assert torch.isfinite(output).all() and output.abs().max() <= 1
    torch.testing.assert_close(output[0, :, 2, 0], torch.ones(3, dtype=image.dtype))
    torch.testing.assert_close(output[0, :, 3, 0], -torch.ones(3, dtype=image.dtype))
    torch.testing.assert_close(output.mean(1), image.mean(1).clamp(-1, 1), rtol=0, atol=1e-12)
    d = image - image.mean(dim=1, keepdim=True)
    after = output - output.mean(dim=1, keepdim=True)
    torch.testing.assert_close(
        torch.linalg.cross(d.transpose(1, -1), after.transpose(1, -1)), torch.zeros_like(d.transpose(1, -1)),
        atol=1e-12, rtol=0)


@torch.inference_mode()
def apply(engine, content, reference):
    model = engine.model
    grid = appearance_grid(model, engine.api, content, reference)
    coefficients = model.sampler(grid, content)
    before = internal_affine(model, coefficients, content)
    hard = model.post_guide(before.clamp(-1, 1))
    native = model.apply_coeffs(coefficients, content)
    if not torch.equal(hard, native):
        raise ValueError('Rebuilt clipping path differs from author output')
    out = model.post_guide(compress(before))

    def rgb(tensor):
        return np.rint(tensor[0].cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)

    original, adjusted = rgb(hard), rgb(out)
    return original, adjusted, dict(signed_channels_outside=float((before.abs() > 1).float().mean().item()),
                                    changed=float((original != adjusted).any(axis=-1).mean()))


def run(out):
    start = time.monotonic()
    synthetic_check()
    torch.set_num_threads(4)
    data = BASE / 'tst100k'
    cohort_path = data / 'cohort-training-coverage-v2'
    audit = checked_json(data / 'training-coverage-audit-v2/results.json', AUDIT_SHA)
    cohort = checked_json(cohort_path / 'cohort.json', COHORT_SHA)
    prepared = checked_json(cohort_path / 'srgb-v1/manifest.json', audit['prepared_manifest_sha256'])
    inputs = {row['original_path']: row for row in prepared['files']}
    rows = cohort['groups']['validation']
    fixed = {name: checked_json(data / folder / 'scores.json', sha) for name, (folder, sha) in ENDPOINTS.items()}
    freeze = {row['index']: row['sha256'] for row in fixed['expanded']['cases']}
    parent = checked_json(PARENT / 'results.json', PARENT_SHA256)
    old_dir = BASE / 'expanded-background-v1'
    old_control = checked_json(old_dir / 'results.json', BACKGROUND_SHA)
    old_cases = {row['pair']: row['variants'] for row in old_control['cases']}
    if digest(WEIGHT) != WEIGHT_SHA256 or len(rows) != 64 or len(parent['cases']) != 6:
        raise ValueError('Unexpected model or development group')
    engine = SpatialEngine('multiref-v1')
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    out.mkdir(exist_ok=False)

    def image(name):
        item = inputs[name]
        return read_rgb(cohort_path / item['prepared_path'], item['sha256'], resize=True)

    def tensor(rgb):
        return (torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255).cuda()

    real_pixels, real = {}, {}
    for row in rows:
        idx = row['index']
        orig, candidate, info = apply(engine, tensor(image(row['content'])), tensor(image(row['reference'])))
        previous = read_rgb(data / ENDPOINTS['expanded'][0] / f'{idx}.png', freeze[idx])
        if not np.array_equal(orig, previous):
            raise ValueError(f'Hard clipping changed on development case {idx}')
        real_pixels[idx] = candidate
        real[idx] = dict(**info, bytes_changed=not np.array_equal(candidate, orig),
                         sha256=save(out / f'{idx}.png', candidate))

    pictures, controlled = {}, {}
    for case in parent['cases']:
        pair = case['pair']
        target = read(case['input_file'])
        analyses, stats = {}, {}
        records = {v['name']: v for v in case['variants']}
        for name in ('identity', *BACKGROUND, *GLOBAL):
            row = records[name]
            old, new, info = apply(engine, engine.analysis(target), engine.analysis(read(row['reference'])))
            old, new = (finish_spatial(target, value, .8) for value in (old, new))
            previous = read_rgb(old_dir / pair / f'{name}-candidate.png', old_cases[pair][name]['output_sha256'])
            if row['certified_aligned'] or not np.array_equal(old, previous):
                raise ValueError('Hard clipping changed on background control')
            original = read(row['methods']['adaptive']['file'])
            analyses[name] = dict(baseline=original, candidate=new, old=old)
            stats[name] = dict(**info, changed_at_80=float((old != new).any(axis=-1).mean()),
                               sha256=save(out / f'{pair}-{name}.png', new))
        controlled[pair] = stats
        pictures[pair] = analyses
    snapshot = out / 'renders.json'
    snapshot.write_text(json.dumps(dict(development=real, controlled=controlled), indent=2), encoding='utf-8')
    reference_truth = {row['index']: image(row['gt']) for row in rows}
    qualities = score_outputs(rows, real_pixels, reference_truth)
    for row in qualities['cases']:
        real[row['index']]['mae'] = row['mae']
    gates = {carrier: response_development_gate(fixed['current13'], qualities, fixed[carrier])
             for carrier in ('C10', 'control')}
    converted, old_compare = [], []
    for case in parent['cases']:
        pair = case['pair']
        target = read(case['input_file'])
        masks = target_regions(target, mask_for(case, pair.split('-')[0]))
        images = pictures[pair]
        for label, destination in (('candidate', converted), ('old', old_compare)):
            values = {name: dict(baseline=row['baseline'], candidate=row[label]) for name, row in images.items()}
            destination.append(dict(pair=pair, variants=summarize(values['identity'], values, masks)))
    elapsed = time.monotonic() - start
    if elapsed >= 120:
        raise TimeoutError('Frozen compression test budget exceeded')
    source = Path(__file__)
    result = dict(development=real, mean=qualities['mean'], delta=qualities['delta_mean'], gates=gates,
                  render_sha256=digest(snapshot), controlled=controlled, controls=decision(converted),
                  old_controls=decision(old_compare), control_metrics=converted, seconds=elapsed,
                  cohort_sha256=COHORT_SHA, audit_sha256=AUDIT_SHA, model_sha256=WEIGHT_SHA256,
                  background_sha256=BACKGROUND_SHA, source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('internal-chroma-compression-protocol.md')))
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(mae=result['mean'], delta=result['delta'], controls=result['controls'], seconds=elapsed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
