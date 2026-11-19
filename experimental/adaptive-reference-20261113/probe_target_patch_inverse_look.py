"""Use target/reference structural correspondences to score candidate inverse looks."""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from probe_capture_invariant_look import LABELS, evaluate, natural_scenes, renderings, samples, style_transforms, target_scene
from probe_capture_look_inverse_prior import inverse_tables, observe
from probe_target_mirror_chroma import BASE, json_to


def resize(rgb):
    image = Image.fromarray(rgb)
    image.thumbnail((256, 256), Image.Resampling.BILINEAR)
    return np.asarray(image)


def blocks(rgb):
    h, w = (size // 8 for size in rgb.shape[:2])
    return rgb[:h * 8, :w * 8].reshape(h, 8, w, 8, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 8, 8, 3)


def structure(rgb):
    light = blocks(rgb) @ np.array([.2126, .7152, .0722])
    micro = light.reshape(-1, 4, 2, 4, 2).mean(axis=(2, 4)).reshape(-1, 16)
    center = micro.mean(axis=1)
    return micro - center[:, None], center


def chroma(rgb):
    color = blocks(rgb).mean(axis=(1, 2))
    return color[:, (0, 2)] - color[:, 1, None]


def patch_evidence(hypotheses, target):
    guide, guide_light = structure(hypotheses['off'][2])
    desired, desired_light = structure(target)
    if not guide.size or not desired.size:
        return 0, None, None
    distance, nearest = cKDTree(desired).query(guide, workers=1)
    reverse = cKDTree(guide).query(desired, workers=1)[1]
    mask = ((reverse[nearest] == np.arange(len(guide))) & (distance / 4 <= 12)
            & (np.abs(guide_light - desired_light[nearest]) <= 16))
    count = int(mask.sum())
    if count < 16:
        return count, None, None
    wanted = chroma(target)[nearest[mask]]
    cost = [float(np.median(np.abs(chroma(hypotheses[label][2])[mask] - wanted).mean(axis=1)))
            for label in LABELS]
    return count, cost, [hypotheses[label][1] for label in LABELS]


def decide(feature, _):
    count, costs, errors = feature
    evidence = {'matches': count, 'costs': costs, 'inverse_mae': errors}
    if costs is None:
        return 'off', evidence
    best, second = np.argsort(costs, kind='stable')[:2]
    accepted = best != 0 and costs[second] - costs[best] >= 1 and errors[best] <= 2
    return LABELS[best] if accepted else 'off', evidence


def compressed(rgb):
    with BytesIO() as stream:
        Image.fromarray(rgb).save(stream, 'JPEG', quality=85)
        stream.seek(0)
        with Image.open(stream) as image:
            return np.array(image.convert('RGB'))


def run(out):
    transforms = style_transforms(BASE)
    lattice, tables = inverse_tables(transforms)
    scenes = natural_scenes(BASE)
    prepared = renderings(scenes, transforms)
    targets = {name: resize(scene['off']) for name, scene in scenes.items()}
    out.mkdir()
    arms = {}
    for encoding in ('original', 'jpeg85'):
        data = samples(BASE, scenes, transforms, lambda rgb: observe(
            compressed(rgb) if encoding == 'jpeg85' else rgb, lattice, tables))
        folds = {}
        for domain, groups in data.items():
            cohort = {
                group: {view: {label: patch_evidence(hypotheses, targets[target_scene(domain, group)])
                               for label, hypotheses in variants.items()}
                        for view, variants in views.items()}
                for group, views in groups.items()
            }
            folder = out / encoding / domain
            folder.mkdir(parents=True)
            folds[domain] = evaluate(cohort, domain, None, prepared, folder, decide)
        if sum(row['states'] for row in folds.values()) != 72:
            raise ValueError('Changed capture cohort')
        arms[encoding] = {'folds': folds, 'passing_domains': sum(row['passed'] for row in folds.values())}
    json_to(out / 'results.json', {'arms': arms, 'passed': all(x['passing_domains'] >= 3 for x in arms.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
