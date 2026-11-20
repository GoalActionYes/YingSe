"""Test whether frozen image features track measured material identity across scenes."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dense_anchor_model import DenseAnchorModel
from probe_mls_material_relative_look import HISTORY_SHA, LIGHTS, SCENES, load_material_regions
from probe_mls_film_light_cross import SOURCE_SHA
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to


def grid_labels(regions, grid):
    labels = np.zeros(grid, dtype=np.int32)
    for index, mask in enumerate(regions, start=1):
        image = Image.fromarray(mask.astype(np.float32))
        coverage = np.asarray(image.resize(grid[::-1], Image.Resampling.BOX))
        use = coverage >= .75
        if np.any(labels[use]):
            raise ValueError('Overlapping material labels')
        labels[use] = index
    return labels.ravel()


def match(target, donor):
    feature, color, labels = target
    ref_feature, ref_color, ref_labels = donor
    dino = (feature @ ref_feature.T).argmax(axis=1)
    rgb = np.square(color[:, None, :] - ref_color[None, :, :]).sum(axis=2).argmin(axis=1)
    measures = {}
    for name, indices in (('dino', dino), ('rgb', rgb)):
        predicted = ref_labels[indices]
        by_material = []
        for label in (1, 2):
            local = labels == label
            total = int(local.sum())
            available = int((ref_labels == label).sum())
            rows = [int(((predicted == code) & local).sum()) for code in (0, 1, 2)]
            by_material.append(dict(query=total, reference=available, matches=rows,
                                    recall=rows[label] / total if min(total, available) >= 4 else None))
        recalls = [row['recall'] for row in by_material]
        measures[name] = dict(materials=by_material,
                              balanced=float(np.mean(recalls)) if None not in recalls else None)
    return measures


def run(out):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = BASE / 'mls-linear-pilot'
    raw = checked_json(source / 'results.json', SOURCE_SHA)
    edits = BASE / 'mls-film-light-cross-v1'
    authored = checked_json(edits / 'results.json', HISTORY_SHA)
    model = DenseAnchorModel()
    regions = load_material_regions()
    features = {}
    inputs = {}
    for scene in SCENES:
        features[scene], inputs[scene] = {}, {}
        for light, suffix in LIGHTS.items():
            file = f'{scene}_2HAL_DESK{suffix}.png'
            paths = {'off': (source / file, raw['scenes'][scene]['inputs_sha256'][file]),
                     'on': (edits / f'{scene}-{light}-reference-film.png', authored['references'][scene][light])}
            for state, (path, expected) in paths.items():
                image = checked_rgb(path, expected)
                token, color, grid, _ = model.extract(image)
                vector = torch.nn.functional.normalize(token, dim=1).cpu().numpy()
                key = f'{light}:{state}'
                features[scene][key] = (vector, color.cpu().numpy(), grid_labels(regions[scene], grid))
                inputs[scene][key] = dict(sha=expected, grid=grid)
    cases = {target: {state: match(features[target]['standard:off'], value)
                      for state, value in features[donor].items()}
             for target, donor in (SCENES, SCENES[::-1])}
    rows = [record for states in cases.values() for record in states.values()]
    complete = all(row[name]['balanced'] is not None for row in rows for name in ('dino', 'rgb'))
    improvement = (float(np.mean([row['dino']['balanced'] - row['rgb']['balanced'] for row in rows]))
                   if complete else None)
    passed = complete and improvement >= .10 and all(row['dino']['balanced'] >= .75 for row in rows)
    out.mkdir()
    json_to(out / 'results.json', dict(model=model.provenance, source=[SOURCE_SHA, HISTORY_SHA], inputs=inputs,
             materials=['paper_prism_blue', 'paper_cube_green'], cases=cases, complete=complete,
             improvement=improvement, passed=passed, script_sha=digest(Path(__file__)),
             protocol_sha=digest(Path(__file__).with_name('mls-material-retrieval-protocol.md'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
