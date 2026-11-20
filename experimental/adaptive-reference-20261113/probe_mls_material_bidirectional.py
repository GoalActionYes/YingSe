"""Test whether bidirectional ranks recover cross-scene material matches."""
import argparse
from pathlib import Path

import numpy as np
import torch

from probe_mls_material_retrieval import (
    BASE, HISTORY_SHA, LIGHTS, SCENES, SOURCE_SHA, DenseAnchorModel, checked_json, checked_rgb,
    digest, grid_labels, json_to, load_material_regions, match,
)

PRIOR = '68e0ce014d03cb8c6bf942f024148bfb173541100985b1ce216e98fb578d90c7'


def ranks(similarity, axis):
    order = np.argsort(-similarity, axis=axis, kind='stable')
    return np.argsort(order, axis=axis, kind='stable')


def compare(query, reference):
    similarity = query[0] @ reference[0].T
    forward, backward = ranks(similarity, 1), ranks(similarity, 0)
    combined = (forward + backward) * similarity.shape[1] + forward
    picked = combined.argmin(axis=1)
    old = forward.argmin(axis=1)
    mutual = backward[np.arange(len(old)), old] == 0
    measured = []
    for label in (1, 2):
        area = query[2] == label
        ready = int(area.sum())
        labels = reference[2][picked]
        correct = int((area & (labels == label)).sum())
        consistent = area & mutual
        measured.append(dict(
            available=ready, selected_counts=[int((area & (labels == code)).sum()) for code in (0, 1, 2)],
            recall=correct / ready, strict_accepted=int(consistent.sum()),
            strict_correct=int((consistent & (reference[2][old] == label)).sum()),
            changed=int((area & (picked != old)).sum())))
    return dict(materials=measured, balanced=float(np.mean([row['recall'] for row in measured])))


def run(out):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    history = checked_json(BASE / 'mls-material-retrieval-v1/results.json', PRIOR)
    source = BASE / 'mls-linear-pilot'
    edits = BASE / 'mls-film-light-cross-v1'
    raw = checked_json(source / 'results.json', SOURCE_SHA)
    authored = checked_json(edits / 'results.json', HISTORY_SHA)
    model = DenseAnchorModel()
    if model.provenance != history['model']:
        raise ValueError('Changed descriptor')
    regions = load_material_regions()
    data = {}
    for scene in SCENES:
        data[scene] = {}
        for light, suffix in LIGHTS.items():
            file = f'{scene}_2HAL_DESK{suffix}.png'
            inputs = {'off': (source / file, raw['scenes'][scene]['inputs_sha256'][file]),
                      'on': (edits / f'{scene}-{light}-reference-film.png', authored['references'][scene][light])}
            for state, (path, key) in inputs.items():
                label = f'{light}:{state}'
                tokens, colors, grid, _ = model.extract(checked_rgb(path, key))
                if dict(sha=key, grid=list(grid)) != history['inputs'][scene][label]:
                    raise ValueError('Changed source image or grid')
                tokens = torch.nn.functional.normalize(tokens, dim=1).cpu().numpy()
                data[scene][label] = (tokens, colors.cpu().numpy(), grid_labels(regions[scene], grid))
    results = {}
    for target, donor in (SCENES, SCENES[::-1]):
        query = data[target]['standard:off']
        results[target] = {}
        for label, reference in data[donor].items():
            frozen = history['cases'][target][label]
            if frozen != match(query, reference):
                raise ValueError('Changed original matching results')
            results[target][label] = dict(single=frozen['dino']['balanced'],
                                          rgb=frozen['rgb']['balanced'], bidirectional=compare(query, reference))
    all_rows = [row for states in results.values() for row in states.values()]
    gain = float(np.mean([row['bidirectional']['balanced'] - row['rgb'] for row in all_rows]))
    out.mkdir()
    json_to(out / 'results.json', dict(prior=PRIOR, cases=results, gain_vs_rgb=gain,
            script_sha=digest(Path(__file__)), protocol_sha=digest(Path(__file__).with_name(
                'mls-material-bidirectional-protocol.md')),
            passed=gain >= .10 and all(row['bidirectional']['balanced'] >= .75 for row in all_rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
