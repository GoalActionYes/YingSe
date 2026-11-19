"""Isolate the reference-side top-class gate using the original color inputs."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from probe_reference_skin_chroma_attribution import CORE, OLD_SHA, enough, regions, score
from probe_reference_skin_chroma_isolation import (
    BASE, BG_SHA, CHECKPOINT, LUMA, ROOT, SpatialEngine, checked_json, checked_rgb,
    chroma, json_to, protector, render_branch, weight_for,
)
from probe_target_mirror_chroma import save


def predictions(model, rgb):
    engine = model._get_engine()
    h, w = rgb.shape[:2]
    scale = min(1.0, model.MAX_EDGE / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = model._cv.resize(rgb, size, interpolation=model._cv.INTER_AREA) if scale < 1 else rgb
    result = engine.segment(model._mp.Image(image_format=model._mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)))
    category = np.asarray(result.category_mask.numpy_view()).squeeze().copy()
    body, face = (np.asarray(result.confidence_masks[i].numpy_view()).squeeze().copy() for i in (2, 3))
    if category.shape != small.shape[:2] or body.shape != category.shape or face.shape != category.shape:
        raise ValueError('Invalid model output shape')
    if not np.isfinite(body + face).all():
        raise ValueError('Invalid confidence')
    return category, body, face


def smooth(model, weight, shape):
    if np.count_nonzero(weight > .5) < max(8, int(np.ceil(weight.size * .001))):
        return None
    weight = model._cv.GaussianBlur(weight, (0, 0), 1.2)
    return model._cv.resize(weight, shape[::-1], interpolation=model._cv.INTER_LINEAR).clip(0, 1)


def compare_weights(model, reference):
    category, body, face = predictions(model, reference)
    ramp = np.clip(((body + face) - .45) / .40, 0, 1)
    raw = (ramp * ramp * (3 - 2 * ramp)).astype(np.float32)
    old = smooth(model, raw * np.isin(category, (2, 3)), reference.shape[:2])
    actual = weight_for(model, reference)
    if actual is None:
        if old is not None:
            raise ValueError('Old fallback changed')
    else:
        np.testing.assert_array_equal(old, actual)
    return old, smooth(model, raw, reference.shape[:2])


def candidate(engine, target, reference, baseline, target_weight, guide_weight):
    if np.array_equal(target, reference):
        return target.copy()
    if target_weight is None or guide_weight is None:
        return baseline.copy()
    y = reference @ LUMA
    return render_branch(engine, target, baseline, target_weight, chroma(reference, y) * guide_weight[..., None], y)[0]


def measure_pair(engine, model, source, old_dir, out, receipts, old_case, old_scores, areas):
    target = checked_rgb(source / 'target.png', receipts['target.png'])
    target_weight = weight_for(model, target)
    category, body, face = predictions(model, target)
    if category.shape != target.shape[:2]:
        raise ValueError('Measurement areas require native canvas')
    target_regions = {name: {'class_counts': np.bincount(category[area].ravel(), minlength=6).tolist(),
                             'mean_body': float(body[area].mean()), 'mean_face': float(face[area].mean())}
                      for name, area in areas.items()}
    out.mkdir()
    states = {}
    for state in ('off', 'on'):
        frames, controls, old_masks, new_masks = [], [], [], []
        artifacts = {}
        for role in ('C1', 'F1'):
            tag = f'{role}-{state}'
            ref_name, base_name = f'{tag}-reference.png', f'{tag}-13.png'
            ref = checked_rgb(source / ref_name, receipts[ref_name])
            baseline = checked_rgb(source / base_name, receipts[base_name])
            control = checked_rgb(old_dir / f'{tag}.png', old_case['outputs'][tag]['sha256'])
            old, new = compare_weights(model, ref)
            np.testing.assert_array_equal(candidate(engine, target, ref, baseline, target_weight, old), control)
            frame = candidate(engine, target, ref, baseline, target_weight, new)
            frames.append(frame)
            controls.append(control)
            old_masks.append(old)
            new_masks.append(new)
            artifacts[tag] = {'sha256': save(out / f'{tag}.png', frame),
                              'changed_output_pixels': int(np.any(frame != control, axis=-1).sum()),
                              'changed_mask_pixels': int(np.count_nonzero(new != old))}
        scores = {}
        for name, area in areas.items():
            before = score(*controls, area)
            if before != old_scores[name][state]['candidate']:
                raise ValueError('Frozen score changed')
            scores[name] = {'old': before, 'new': score(*frames, area)}
        states[state] = {'scores': scores, 'artifacts': artifacts,
                         'old_mae': float(np.abs(old_masks[0] - old_masks[1]).mean()),
                         'new_mae': float(np.abs(new_masks[0] - new_masks[1]).mean())}
    return {'states': states, 'target_regions': target_regions}


@torch.inference_mode()
def run(out):
    torch.set_num_threads(4)
    anchor = BASE / 'bg-real-editor-look'
    previous = BASE / 'reference-skin-chroma-isolation-v1'
    receipt = checked_json(anchor / 'results.json', BG_SHA)
    old = checked_json(previous / 'results.json', OLD_SHA)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Changed 13 weights')
    model = protector(ROOT / 'experimental/skin-protection-20261019/reference_color_app/skin.py')
    flat = np.full((64, 64, 3), 128, np.uint8)
    other = 255 - flat
    if compare_weights(model, other) != (None, None):
        raise ValueError('Uniform negative unexpectedly received a reference skin selection')
    np.testing.assert_array_equal(candidate(engine, flat, flat.copy(), other, None, None), flat)
    np.testing.assert_array_equal(candidate(engine, flat, other, other, None, None), other)
    boxes = regions()
    out.mkdir()
    cases = {pair: measure_pair(engine, model, anchor / pair, previous / 'background' / pair, out / pair,
                                item['artifacts'], old['suites']['background']['cases'][pair],
                                old['registered_regions'][pair], boxes[pair])
             for pair, item in receipt['cases'].items()}
    core, regressions = [], []
    for pair, item in cases.items():
        for state, row in item['states'].items():
            for name, values in row['scores'].items():
                before, after = values['old'], values['new']
                if name in CORE.get(pair, ()):
                    core.append(enough(before, after))
                elif after - before > .25:
                    regressions.append(dict(pair=pair, state=state, region=name, delta=after - before))
    if len(core) != 6:
        raise ValueError('Expected six core measures')
    json_to(out / 'results.json', dict(cases=cases, baseline_sha256=CHECKPOINT, old_sha256=OLD_SHA,
                                      primary_improved=sum(core), regressions=regressions,
                                      passed=sum(core) >= 5 and not regressions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('locator_dependencies', type=Path)
    args = parser.parse_args()
    sys.path.append(str(args.locator_dependencies.resolve(strict=True)))
    run(args.out)
