"""Replay reference color and appearance changes independently on frozen weights."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from appearance_pairwise import appearance_forward
from appearance_replication import initialize_replication_model
from background_intervention import edit_weight, target_regions
from independent_coco import digest
from probe_deep_preset_two_image import summarize
from probe_expanded_background import PARENT_SHA256, WEIGHT, WEIGHT_SHA256
from probe_internal_chroma_compression import BACKGROUND_SHA
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import ROLES, compare, gates_for, mask_for, save
from probe_reference_chroma_spread import BACKGROUND, BASE, GLOBAL, PARENT, decision, read
from probe_roundtrip_grade_choice import checked_json, read_rgb
from spatial_render import SpatialEngine, finish_spatial


ARMS = ('full', 'stats', 'appearance')


@torch.inference_mode()
def render(engine, target, content, stats, appearance=None):
    raw, _ = appearance_forward(engine.model, engine.api, content, stats, appearance_reference=appearance)
    full = np.rint(raw[0].cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    return finish_spatial(target, full, .8)


def replay(engine, out, prefix, target, pictures, anchor):
    content = engine.analysis(target)
    encoded = {name: engine.analysis(rgb) for name, rgb in pictures.items()}
    root = encoded[anchor]
    anchor_rgb = render(engine, target, content, root)
    predictions = {}
    hashes = {anchor: dict(anchor=save(out / f'{prefix}-{anchor}.png', anchor_rgb))}
    for name, value in encoded.items():
        if name == anchor:
            predictions[name] = dict.fromkeys(ARMS, anchor_rgb)
            continue
        candidates = dict(full=render(engine, target, content, value),
                          stats=render(engine, target, content, value, root),
                          appearance=render(engine, target, content, root, value))
        predictions[name] = candidates
        hashes[name] = {arm: save(out / f'{prefix}-{name}-{arm}.png', rgb) for arm, rgb in candidates.items()}
    return predictions, hashes


def interactions(prediction, anchor, regions):
    root = prediction[anchor]['full'].astype(np.int16)
    records = {}
    for name, row in prediction.items():
        if name == anchor:
            continue
        full, stats, appearance = (row[key].astype(np.int16) - root for key in ARMS)
        residual = full - stats - appearance
        records[name] = {
            r: (dict(**{a: float(np.abs(d[m]).mean()) for a, d in zip(ARMS, (full, stats, appearance), strict=True)},
                     interaction=float(np.abs(residual[m]).mean()),
                     interaction_mean_rgb=residual[m].mean(axis=0).tolist()) if m.any() else None)
            for r, m in regions.items() if r in ('whole', 'person', 'neutral_proxy')
        }
    return records


def run(out):
    start = time.monotonic()
    torch.set_num_threads(4)
    docs = Path(__file__).resolve().parents[2] / 'docs/research'
    legacy = checked_json(PARENT / 'results.json', PARENT_SHA256)
    local = checked_json(docs / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    original = BASE / 'expanded-background-v1'
    original_report = checked_json(original / 'results.json', BACKGROUND_SHA)
    previous = {row['pair']: row['variants'] for row in original_report['cases']}
    if digest(WEIGHT) != WEIGHT_SHA256 or len(legacy['cases']) != 6 or len(local['cases']) != 6:
        raise ValueError('Frozen weight or cohort changed')
    engine = SpatialEngine('multiref-v1')
    initialize_replication_model(engine.model, engine.model.state_dict())
    engine.model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
    engine.model.eval().requires_grad_(False)
    out.mkdir(exist_ok=False)
    background, desired, identities = {}, {}, {'background': {}, 'local': {}}
    for case in legacy['cases']:
        pair = case['pair']
        target = read(case['input_file'])
        baseline_rows = {v['name']: v for v in case['variants']}
        photos = {name: read(baseline_rows[name]['reference']) for name in ('identity', *BACKGROUND, *GLOBAL)}
        prediction, hashes = replay(engine, out, f'background-{pair}', target, photos, 'identity')
        for name, result in prediction.items():
            expected = previous[pair][name]['output_sha256']
            stored = read_rgb(original / pair / f'{name}-candidate.png', expected)
            if not np.array_equal(result['full'], stored):
                raise ValueError('Frozen full background rendering changed')
        background[pair] = prediction
        identities['background'][pair] = hashes

        folder = BASE / 'local-edit-toggle-v1' / pair
        files = local['cases'][pair]['files']
        if not np.array_equal(target, read_rgb(folder / 'target.png', files['target.png'])):
            raise ValueError('Target difference between controls')
        photos = {name: read_rgb(folder / f'{name}-reference.png', files[f'{name}-reference.png']) for name in ROLES}
        prediction, hashes = replay(engine, out, f'local-{pair}', target, photos, 'base')
        desired[pair] = prediction
        identities['local'][pair] = hashes
    snapshot = out / 'predictions.json'
    snapshot.write_text(json.dumps(identities, indent=2), encoding='utf-8')

    results = {key: [] for key in ARMS}
    authored = {key: [] for key in ARMS}
    interaction_summary = {'background': {}, 'local': {}}
    for case in legacy['cases']:
        pair = case['pair']
        target = read(case['input_file'])
        person, ref_person = (mask_for(case, photo) for photo in pair.split('-'))
        regions = target_regions(target, person)
        expected_13 = {row['name']: read(row['methods']['adaptive']['file'])
                       for row in case['variants'] if row['name'] in background[pair]}
        for label in ARMS:
            values = {name: dict(baseline=expected_13[name], candidate=row[label]) for name, row in background[pair].items()}
            results[label].append(dict(pair=pair, variants=summarize(values['identity'], values, regions)))
        interaction_summary['background'][pair] = interactions(background[pair], 'identity', regions)

        weights = tuple(edit_weight(mask, 1)[0] if mask.any() else np.zeros(mask.shape) for mask in (person, ref_person))
        local_masks = dict(regions, edited_core=weights[0] == 1, untouched=weights[0] == 0,
                           neutral_person=regions['neutral_proxy'] & person,
                           neutral_edited=regions['neutral_proxy'] & (weights[0] == 1))
        files = local['cases'][pair]['files']
        folder = BASE / 'local-edit-toggle-v1' / pair
        expected = {name: read_rgb(folder / f'{name}-expected.png', files[f'{name}-expected.png']) for name in ROLES}
        for label in ARMS:
            rendered = {name: values[label] for name, values in desired[pair].items()}
            scores = compare(rendered, expected, local_masks)
            authored[label].append(dict(pair=pair, gates=gates_for(scores, rendered, (person, ref_person), weights),
                                        metrics=scores))
        interaction_summary['local'][pair] = interactions(desired[pair], 'base', regions)
    elapsed = time.monotonic() - start
    if elapsed >= 100:
        raise TimeoutError('Frozen branch replay budget exceeded')
    source = Path(__file__)
    report = dict(background={key: decision(rows) for key, rows in results.items()}, local=authored,
                  interference_regions=results, interaction=interaction_summary, seconds=elapsed,
                  model_sha256=WEIGHT_SHA256, source_sha256=digest(source), predictions_sha256=digest(snapshot),
                  protocol_sha256=digest(source.with_name('reference-path-response-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(background=report['background'], seconds=elapsed)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
