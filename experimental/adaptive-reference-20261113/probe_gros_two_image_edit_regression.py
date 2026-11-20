"""Regress continuous color coefficients using an ordinary target and reference."""
from argparse import ArgumentParser
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_capture_invariant_look import capture_groups, descriptor, natural_scenes, render_style, samples, style_transforms
from probe_gros_film_b_cross import ARCHIVE, GROUPS, LIGHTS, masks, score
from probe_gros_local_tone_coupling import describe
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, digest, finish_spatial, json_to, panel, save


def small(rgb):
    view = Image.fromarray(rgb)
    view.thumbnail((256, 256), Image.Resampling.BILINEAR)
    return np.array(view)


def features(rgb):
    return np.r_[descriptor(small(rgb)) / 255, describe(rgb)]


def poly(rgb):
    r, g, b = (rgb.reshape(-1, 3).astype(float) / 255).T
    return np.stack((np.ones_like(r), r, g, b, r*r, g*g, b*b, r*g, r*b, g*b), axis=1)


def coefficients(start, changed):
    x = poly(start)
    y = (changed.astype(float) - start).reshape(-1, 3)
    return np.linalg.lstsq(x, y, rcond=1e-4)[0]


def render(target, coef):
    change = (poly(target) @ coef).reshape(target.shape)
    image = np.rint(np.clip(target + change, 0, 255)).astype(np.uint8)
    return finish_spatial(target, image, STRENGTH)


def feature_pair(target, reference):
    return np.r_[reference, reference - target]


def model(rows, truth):
    x, y = np.asarray(rows), np.asarray(truth)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    x = (x - center) / scale
    intercept = y.mean(axis=0)
    regularizer = .05 * np.eye(x.shape[1])
    weights = np.linalg.solve((x.T @ x) / len(x) + regularizer, x.T @ (y - intercept) / len(x))
    return center, scale, intercept, weights


def train(ops):
    natural = natural_scenes(BASE)
    reference = samples(BASE, natural, ops, describe=features)
    captures = capture_groups(BASE)
    targets = {}
    for domain, scenes in captures.items():
        for scene, views in scenes.items():
            src = small(next(iter(views.values())))
            states = {key: render_style(src, transform) for key, transform in ops.items()}
            targets[domain, scene] = src, states
    for scene, views in natural.items():
        targets["arri", scene] = views["off"], {label: views[label] for label in ops}
    desired = {key: (features(rgb), {"off": np.zeros((10, 3))} |
                     {style: coefficients(rgb, after) for style, after in states.items()})
               for key, (rgb, states) in targets.items()}
    by_ref, by_pair, outputs = [], [], []
    for domain, groups in reference.items():
        for group, variants in groups.items():
            for state in variants.values():
                for (domain_id, target_id), (x, labels) in desired.items():
                    if (domain, group) == (domain_id, target_id):
                        continue
                    for role, ref in state.items():
                        by_ref.append(ref)
                        by_pair.append(feature_pair(x, ref))
                        outputs.append(labels[role].ravel())
    return {"reference": model(by_ref, outputs), "two_image": model(by_pair, outputs)}, len(outputs)


def predict(target, ref, digest_target, digest_ref, model_data, paired):
    if np.array_equal(target, ref):
        return target.copy()
    x = feature_pair(digest_target, digest_ref) if paired else digest_ref
    center, scale, intercept, weights = model_data
    coef = (intercept + ((x - center) / scale) @ weights).reshape(10, 3)
    return render(target, coef)


def run(out, parent_sha):
    root = BASE / "gros-film-b-cross-v2"
    source = checked_json(root / "results.json", parent_sha)
    roster = checked_json(BASE / "gros-portrait-light-transfer-v1/results.json", ARCHIVE)
    regions_path = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(regions_path) != roster["regions_sha256"]:
        raise ValueError("Changed region ledger")
    coords = json.loads(regions_path.read_bytes())
    ops = style_transforms(BASE)
    fitted, rows = train(ops)
    people = {}
    for p in GROUPS:
        people[p] = {}
        for light in LIGHTS:
            hashes = source["inputs"][f"{p}-{light}"]["hashes"]
            off, film_b = [checked_rgb(root / f"{p}-{light}-{key}.png", hashes[key]) for key in ("off", "on")]
            states = dict(off=off, film_a=render_style(off, ops["film_a"]), film_b=film_b)
            people[p][light] = {k: (img, features(img)) for k, img in states.items()}
    out.mkdir()
    cases = {}
    for person, donor in permutations(GROUPS, 2):
        target, x = people[person]["cold"]["off"]
        references = people[donor]
        expected = {k: finish_spatial(target, a[0], STRENGTH)
                    for k, a in people[person]["cold"].items()}
        geometry = masks(coords[person]["cold"], coords["screen"], target.shape[:2])
        arms = {}
        for k, params in fitted.items():
            arms[k] = {light: {state: predict(target, rgb, x, ref, params, k == "two_image")
                              for state, (rgb, ref) in values.items()} for light, values in references.items()}
            if not np.array_equal(predict(target, target, x, x, params, k == "two_image"), target):
                raise ValueError("Same-input identity failed")
        diagnostics = {"off": target}
        for k in ops:
            diagnostics[k] = render(target, coefficients(small(target), small(people[person]["cold"][k][0])))
        arms["capacity"] = {light: diagnostics for light in LIGHTS}
        summary = {arm: {k: score(target, expected[k],
                                  {light: {"off": group["off"], "on": group[k]} for light, group in views.items()},
                                  geometry) for k in ops} for arm, views in arms.items()}
        name = f"{person}-from-{donor}"
        hashes = {f"{arm}-{light}-{state}": save(out / f"{name}-{arm}-{light}-{state}.png", rgb)
                  for arm, views in arms.items() if arm != "capacity" for light, group in views.items()
                  for state, rgb in group.items()}
        cases[name] = dict(scores=summary, hashes=hashes,
            panel_sha=panel(out / f"{name}-comparison.png", {"expected": expected} |
                {f"{arm} {light}": views[light] for arm, views in arms.items() if arm != "capacity" for light in LIGHTS}))
    script = Path(__file__)
    json_to(out / "results.json", dict(parent=parent_sha, training_pairs=rows, cases=cases,
        script_sha=digest(script), protocol_sha=digest(script.with_name("gros-two-image-edit-regression-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("parent_sha")
    run(**vars(parser.parse_args()))
