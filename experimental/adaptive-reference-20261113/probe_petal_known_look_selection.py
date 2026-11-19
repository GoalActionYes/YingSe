"""Rank fixed target look previews against a reference with frozen PETAL features."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from petal_encoder import encode
from petal_model import PetalModel
from probe_capture_invariant_look import LABELS, evaluate, natural_scenes, render_style, renderings, samples, style_transforms, target_scene
from probe_target_mirror_chroma import BASE, json_to
from probe_target_patch_inverse_look import compressed


def features(model, rgb):
    image = np.asarray(Image.fromarray(rgb).resize((256, 256), Image.Resampling.BILINEAR))
    z = encode(model, [image])[0].cpu().numpy().astype(np.float64)
    return z / np.linalg.norm(z)


def decide(reference, alternatives):
    values = np.clip(1 - alternatives @ reference, 0, 2)
    best, second = np.argsort(values, kind='stable')[:2]
    accepted = best != 0 and values[second] - values[best] > .001
    return LABELS[best] if accepted else 'off', values.tolist()


def run(out):
    torch.set_num_threads(4)
    out.mkdir()
    transforms = style_transforms(BASE)
    scenes = natural_scenes(BASE)
    prepared = renderings(scenes, transforms)
    model = PetalModel()
    identity = model.state_digest()
    alternatives = {}
    for name, scene in scenes.items():
        original = scene['off']
        previews = [original, *(render_style(original, transforms[style]) for style in LABELS[1:])]
        alternatives[name] = np.stack([features(model, rgb) for rgb in previews])
    arms = {}
    for encoding in ('original', 'jpeg85'):
        def describe(rgb):
            return features(model, compressed(rgb) if encoding == 'jpeg85' else rgb)

        data = samples(BASE, scenes, transforms, describe)
        folds = {}
        for domain, groups in data.items():
            folder = out / encoding / domain
            folder.mkdir(parents=True)
            cohort = {
                group: {view: {label: (ref, alternatives[target_scene(domain, group)])
                               for label, ref in variants.items()}
                        for view, variants in views.items()}
                for group, views in groups.items()
            }
            folds[domain] = evaluate(cohort, domain, None, prepared, folder,
                                     lambda feature, _: decide(*feature))
        if sum(row['states'] for row in folds.values()) != 72:
            raise ValueError('Changed capture cohort')
        arms[encoding] = {'folds': folds, 'passing_domains': sum(row['passed'] for row in folds.values())}
        print(encoding, {domain: fold['correct'] for domain, fold in folds.items()}, flush=True)
    if identity != model.state_digest():
        raise ValueError('Frozen model state changed')
    json_to(out / 'results.json', {'arms': arms, 'model_state_sha256': identity,
                                   'passed': all(x['passing_domains'] >= 3 for x in arms.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
