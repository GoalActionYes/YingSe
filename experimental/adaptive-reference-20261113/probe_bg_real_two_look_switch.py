"""Add a controlled style switch to the qualified Film B background suite."""
import argparse
from pathlib import Path

import numpy as np
import torch

from appearance_replication import initialize_replication_model
from background_intervention import target_regions
from probe_arri_shared_look import metrics
from probe_bg_real_editor_look import PARENT, PARENT_SHA, evaluate, mask_for, split_signal
from probe_bg_real_film_b_qualification import ROLES, transform
from probe_capture_invariant_look import style_transforms
from probe_frozen_candidate_development_crosscheck import CHECKPOINT, WEIGHT, WEIGHT_SHA256, grade
from probe_reference_skin_chroma_attribution import regions
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save
from spatial_render import SpatialEngine, finish_spatial
from verify_development_inputs import manifest

FILM_B_REPORT_SHA = '881bf466bd4456ba5b6a4ffe5a3126ddb7fb3de25912291a5790b9fb83813e16'


def prepare(out, capture):
    original = BASE / 'bg-real-editor-look'
    b_root = BASE / 'bg-real-film-b-qualification-v1'
    a_style = style_transforms(BASE)['film_a']
    parent = checked_json(PARENT / 'results.json', PARENT_SHA)
    masks = {x['pair']: x for x in parent['cases']}
    boxes = regions()
    cases = {}
    for key, row in manifest(BASE, 'bg-real-editor-look')['cases'].items():
        folder = out / key
        folder.mkdir()
        artifacts = row['artifacts']
        target = checked_rgb(original / key / 'target.png', artifacts['target.png'])
        raw_a, audit = transform(target, a_style)
        if audit['roundtrip_max_rgb8'] != 0:
            raise ValueError('Changed Film A off-state display round trip')
        a = finish_spatial(target, raw_a, STRENGTH)
        b = checked_rgb(b_root / f'{key}-expected.png', capture['qualification'][key]['expected_sha256'])
        hashes = {'expected-a': save(folder / 'expected-a.png', a)}
        references = {}
        for role in ROLES:
            name = f'{role}-off-reference.png'
            off = checked_rgb(original / key / name, artifacts[name])
            references[role], debug = transform(off, a_style)
            if debug['roundtrip_max_rgb8'] != 0:
                raise ValueError('Changed background-specific display round trip')
            hashes[role] = save(folder / f'{role}-a-reference.png', references[role])
        person = mask_for(masks[key], key.split('-')[0])
        neutral = np.logical_and.reduce([target_regions(frame, person)['neutral_proxy'] for frame in (target, a, b)])
        areas = {'whole': np.s_[:, :], 'neutral': neutral, **boxes[key]}
        if neutral.sum() < 256:
            raise ValueError('Lost neutral support')
        strength = {channel: float(np.abs(values).mean()) for channel, values in split_signal(b, a).items()}
        cases[key] = (target, a, b, person, areas, references, hashes, strength)
    return cases


def run_arm(engine, out, cases, frozen, arm):
    b_root = BASE / 'bg-real-film-b-qualification-v1'
    results = {}
    for key, (target, a, b, person, areas, references, assets, strength) in cases.items():
        previous = frozen['cases'][key]
        photos, response, hashes, board = {}, {}, {}, {}
        for role, guide in references.items():
            off, existing = [checked_rgb(b_root / f'{arm}-{key}-{role}-{state}.png', previous['outputs'][f'{role}-{state}'])
                             for state in ('off', 'on')]
            if arm == 'expanded':
                current = grade(engine, target, guide, target)[0]
            else:
                current = finish_spatial(target, engine.prepare(target, guide).apply(target), STRENGTH)
            hashes[role] = save(out / key / f'{arm}-{role}-a.png', current)
            photos[role] = {'off': off, 'on': current}
            response[role] = {label: metrics(a[where], b[where], current[where], existing[where],
                                             np.ones(a[where].shape[:2],dtype=bool))
                              if isinstance(where, tuple) else metrics(a,b,current,existing,where)
                              for label, where in areas.items()}
            board[role] = {'reference A': guide, 'A candidate': current, 'B candidate': existing, 'known A': a, 'known B': b}
        qualification = min(strength.values()) >= 1.
        pass_switch = qualification and all(row[region]['passed'] for row in response.values() for region in ('whole','neutral'))
        scores_a = evaluate(target, {'off':target,'on':a}, photos, person)
        results[key] = dict(film_a=scores_a, film_b=previous['scores'], look_switch=response,
                            qualified=qualification, expected_switch_strength=strength, asset_hashes=assets,
                            frame_hashes=hashes, passed=pass_switch and scores_a['passed'] and previous['scores']['passed'],
                            switch_passed=pass_switch, panel_sha256=panel(out / key / f'{arm}-comparison.png', board))
    return results


@torch.inference_mode()
def run(out):
    parent = checked_json(BASE / 'bg-real-film-b-qualification-v1/results.json', FILM_B_REPORT_SHA)
    torch.set_num_threads(4)
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT or digest(WEIGHT) != WEIGHT_SHA256:
        raise ValueError('Changed frozen checkpoints')
    out.mkdir()
    cases = prepare(out, parent)
    results = {}
    for arm in ('13','expanded'):
        if arm == 'expanded':
            initialize_replication_model(engine.model, engine.model.state_dict())
            engine.model.load_state_dict(torch.load(WEIGHT, map_location='cpu', weights_only=True), strict=True)
            engine.model.eval().requires_grad_(False)
        results[arm] = run_arm(engine, out, cases, parent['comparison'][arm], arm)
    counts = {arm: sum(row['passed'] for row in cases.values()) for arm, cases in results.items()}
    json_to(out / 'results.json', dict(film_b_report_sha256=FILM_B_REPORT_SHA, counts=counts, results=results))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    run(p.parse_args().out)
