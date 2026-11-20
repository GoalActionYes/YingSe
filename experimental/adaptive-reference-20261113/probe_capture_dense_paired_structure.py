"""Contrast frozen DINO spatial descriptors with capture and digital pairs."""
import argparse
from pathlib import Path

import numpy as np
import torch

from dense_anchor_model import DenseAnchorModel
from probe_capture_tile_tone_color import MID_SHA, STATES, acquire, render
from probe_target_mirror_chroma import BASE, checked_json, digest, json_to

PARENT = 'e5ea6d809e495ab3cd9a7b0ea30e11b00a17292491961bb921f6bd8b6e0ee550'


def describe(model, rgb):
    token, _, shape, _ = model.extract(rgb)
    values = torch.nn.functional.normalize(token, dim=1).reshape(*shape, -1).cpu().numpy()
    if min(shape) < 2 or not np.isfinite(values).all():
        raise ValueError('Invalid spatial feature map')
    return (np.abs(np.diff(values, axis=0)).mean(axis=(0, 1), dtype=np.float64)
            + np.abs(np.diff(values, axis=1)).mean(axis=(0, 1), dtype=np.float64)) / 2


def fit(features, excluded):
    local = [np.asarray([[states[role] for role in STATES] for states in scenes.values()])
             for family, scenes in features.items() if family != excluded]
    means = np.stack([rows.mean(axis=0) for rows in local]).mean(axis=0)
    other = (means[0] + means[2]) / 2
    direction = means[1] - other
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length <= 1e-12:
        return None
    direction /= length
    positive, negative = float(means[1] @ direction), float(other @ direction)
    if positive <= negative:
        return None
    return direction, (positive + negative) / 2


def decide(features, family):
    result = {}
    fitted = fit(features, family)
    for scene, states in features[family].items():
        row = {}
        for state, vector in states.items():
            if fitted is None:
                row[state] = dict(capture=None, margin=None)
                continue
            direction, limit = fitted
            margin = float(vector @ direction - limit)
            row[state] = dict(capture=margin > 0, margin=margin)
        result[scene] = row
    return result


def run(out):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = checked_json(BASE / 'capture-tile-tone-color-v1/results.json', PARENT)
    groups, mid = acquire()
    model = DenseAnchorModel()
    features = {family: {scene: {label: describe(model, image) for label, image in trio.items()}
                         for scene, trio in cases.items()} for family, cases in groups.items()}
    for family, data in features.items():
        if set(data) != set(reference['choices'][family]):
            raise ValueError('Changed capture cohort')
    decisions = {family: decide(features, family) for family in features}
    summaries = {}
    for family, choices in decisions.items():
        counts = {}
        for key in STATES:
            readings = [row[key]['capture'] for row in choices.values()]
            counts[key] = dict(detected=readings.count(True), missing=readings.count(None), total=len(readings))
        summaries[family] = counts
    out.mkdir()
    evidence = render(mid, decisions['mid']['copyroom'], out)
    passed = all(
        row['captured']['detected'] >= .75 * row['captured']['total']
        and all(row[key]['missing'] == 0 for key in STATES)
        and row['base']['detected'] == row['edited']['detected'] == 0
        for row in summaries.values()
    ) and all(value['passed'] for value in evidence.values())
    json_to(out / 'results.json', dict(prior=PARENT, render_parent=MID_SHA, model=model.provenance,
            script_sha=digest(Path(__file__)), protocol_sha=digest(Path(__file__).with_name(
                'capture-dense-paired-structure-protocol.md')), decisions=decisions, summaries=summaries,
            pictures=evidence, passed=passed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
