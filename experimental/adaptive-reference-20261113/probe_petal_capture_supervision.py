"""Compare matched ridge controls on frozen photographic style features."""

import argparse
from pathlib import Path

import numpy as np
import torch

from petal_model import PetalModel
from probe_capture_invariant_look import ARMS, evaluate, fit, natural_scenes, renderings, samples, style_transforms
from probe_petal_known_look_selection import features
from probe_target_mirror_chroma import BASE, json_to
from probe_target_patch_inverse_look import compressed


def save_features(out, encodings):
    vectors = {}
    for encoding, domains in encodings.items():
        for domain, groups in domains.items():
            for group, views in groups.items():
                for view, variants in views.items():
                    for label, vector in variants.items():
                        vectors['/'.join((encoding, domain, group, view, label))] = vector
    np.savez_compressed(out / 'features.npz', **vectors)


def run(out):
    torch.set_num_threads(4)
    out.mkdir()
    transforms = style_transforms(BASE)
    scenes = natural_scenes(BASE)
    prepared = renderings(scenes, transforms)
    model = PetalModel()
    identity = model.state_digest()
    encodings = {
        'original': samples(BASE, scenes, transforms, lambda rgb: features(model, rgb)),
        'jpeg85': samples(BASE, scenes, transforms, lambda rgb: features(model, compressed(rgb))),
    }
    if identity != model.state_digest():
        raise ValueError('Frozen model state changed')
    save_features(out, encodings)
    original = encodings['original']
    results = {}
    for arm, penalty in ARMS.items():
        models = {name: fit([other for key, other in original.items() if key != name], penalty)
                  for name in original}
        results[arm] = {}
        for encoding, domains in encodings.items():
            folds = {}
            for domain, groups in domains.items():
                folder = out / arm / encoding / domain
                folder.mkdir(parents=True)
                model = models[domain]
                folds[domain] = evaluate(groups, domain, model, prepared, folder) | {
                    'fit_rows': model[-2], 'paired_rows': model[-1]}
            if sum(row['states'] for row in folds.values()) != 72:
                raise ValueError('Changed capture cohort')
            results[arm][encoding] = {'folds': folds, 'passing_domains': sum(r['passed'] for r in folds.values())}
            print(arm, encoding, {name: row['correct'] for name, row in folds.items()}, flush=True)
    json_to(out / 'results.json', {'arms': results, 'model_state_sha256': identity,
                                   'passed': all(r['passing_domains'] >= 3 for r in results['paired'].values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
