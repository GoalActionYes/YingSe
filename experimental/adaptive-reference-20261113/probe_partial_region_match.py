"""Compare balanced and reference-relaxed region matching on two-image color transfer."""

from argparse import ArgumentParser
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dense_anchor_model import DenseAnchorModel
from probe_bg_real_editor_look import PARENT, PARENT_SHA, evaluate, mask_for
from probe_gros_film_b_cross import GROUPS, LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_sparks_shared_chroma import render
from probe_target_mirror_chroma import BASE, CHECKPOINT, STRENGTH, checked_json, checked_rgb, digest, json_to, panel, save

BG_SHA = "11b72e0865bad7f3d20a4b3b9a8120c5e8df0de8fd2bbc78549625d67fa1d94d"
STATES = ("off", "on")
ARMS = {"equal_reference": 1., "flexible_reference": .5}


def transport(kernel, power):
    rows, columns = kernel.shape
    # Target weights are fixed. Reference weights are normalized exactly (power=1)
    # or softly penalized. The last update restores the target marginal.
    right = np.ones(columns)
    for _ in range(160):
        left = (1 / rows) / (kernel @ right)
        right = ((1 / columns) / (kernel.T @ left)) ** power
    left = (1 / rows) / (kernel @ right)
    flow = left[:, None] * kernel * right[None, :]
    if not np.isfinite(flow).all():
        raise ValueError("Nonfinite transport")
    np.testing.assert_allclose(flow.sum(axis=1), 1 / rows, rtol=1e-9, atol=0)
    if power == 1:
        np.testing.assert_allclose(flow.sum(axis=0), 1 / columns, rtol=.005, atol=0)
    return flow


def describe(model, rgb):
    small = Image.fromarray(rgb)
    small.thumbnail((224, 224), Image.Resampling.BILINEAR)
    features, pixels, shape, _ = model.extract(np.asarray(small))
    tokens = F.normalize(features, dim=1).cpu().numpy().astype(np.float64)
    colors = chroma(pixels.cpu().numpy() * 255)
    return tokens, colors, shape


def recolor(target, target_data, guide_data, original):
    tokens, colors, shape = target_data
    guide, palette, _ = guide_data
    cost = np.maximum(0, 1 - tokens @ guide.T)
    kernel = np.exp(-cost / .08)
    results, audits = {}, {}
    for arm, power in ARMS.items():
        flow = transport(kernel, power)
        shift = (flow @ palette * len(colors) - colors).reshape(*shape, 2)
        block = torch.from_numpy(shift).permute(2, 0, 1)[None]
        full = F.interpolate(block, size=target.shape[:2], mode="bilinear", align_corners=False)
        full = full[0].permute(1, 2, 0).numpy()
        grade, gamut = render(chroma(target) + STRENGTH * full, original @ LUMA)
        results[arm] = grade
        supply = flow.sum(axis=0) * len(guide)
        audits[arm] = dict(relative_supply_min=float(supply.min()), relative_supply_max=float(supply.max()),
                           below_half=int(np.count_nonzero(supply < .5)), color_clip_fraction=gamut,
                           luma_max=float(np.abs((grade.astype(float) - original) @ LUMA).max()))
    return results, audits


def predict(target, reference, model, targets, guides, key, original):
    if np.array_equal(target, reference):
        return {arm: target.copy() for arm in ARMS}, {arm: {"identity": True} for arm in ARMS}
    if key not in guides:
        guides[key] = describe(model, reference)
    return recolor(target, targets, guides[key], original)


def experiment(model, out, target, expected, refs, original, score_arms, cache):
    target_data = describe(model, target)
    same, _ = predict(target, target, model, target_data, cache, None, original[next(iter(original))]["off"])
    if not all(np.array_equal(image, target) for image in same.values()):
        raise ValueError("Same-image control failed")
    images = {arm: {name: {} for name in refs} for arm in ARMS}
    details, hashes = {}, {}
    for name, versions in refs.items():
        for state, (key, reference) in versions.items():
            result, details[f"{name}-{state}"] = predict(target, reference, model, target_data, cache, key,
                                                          original[name][state])
            for arm, grade in result.items():
                images[arm][name][state] = grade
                hashes[f"{arm}-{name}-{state}"] = save(out / f"{arm}-{name}-{state}.png", grade)
    board = {"target": {"original": target, "expected": expected}}
    board |= {f"{arm} {name}": states for arm, repeats in ({"frozen13": original} | images).items()
              for name, states in repeats.items()}
    return dict(scores={arm: score_arms(frames) for arm, frames in ({"frozen13": original} | images).items()},
                transport=details, hashes=hashes, board=panel(out / "comparison.png", board))


def bg_suite(model, out):
    directory = BASE / "bg-real-editor-look"
    frozen = checked_json(directory / "results.json", BG_SHA)
    if frozen["model_sha256"] != CHECKPOINT or frozen["strength"] != STRENGTH:
        raise ValueError("Changed BG-REAL checkpoint")
    registry = checked_json(PARENT / "results.json", PARENT_SHA)
    entries = {row["pair"]: row for row in registry["cases"]}
    results = {}
    for key, row in frozen["cases"].items():
        here = directory / key
        artifacts = row["artifacts"]

        def load(name):
            return checked_rgb(here / f"{name}.png", artifacts[f"{name}.png"])

        target, expected = load("target"), load("expected-on")
        names = ("C1", "F1")
        refs = {name: {state: (f"{name}-{state}", load(f"{name}-{state}-reference")) for state in STATES}
                for name in names}
        original = {name: {state: load(f"{name}-{state}-13") for state in STATES} for name in names}
        person = mask_for(entries[key], key.split("-")[0])
        dest = out / key
        dest.mkdir()
        results[key] = experiment(model, dest, target, expected, refs, original,
                                  lambda frames: evaluate(target, {"off": target, "on": expected}, frames, person), {})
        np.testing.assert_allclose(results[key]["scores"]["frozen13"]["max_protected_swap"], row["max_protected_swap"])
    return results


def gros_suite(model, out):
    directory = BASE / "gros-film-b-cross-v2"
    frozen = checked_json(directory / "results.json", BASELINE_SHA)
    if frozen["spatial"] != CHECKPOINT or frozen["strength"] != STRENGTH:
        raise ValueError("Changed Gros checkpoint")
    zones = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    photos = {key: {state: checked_rgb(directory / f"{key}-{state}.png", row["hashes"][state]) for state in STATES}
              for key, row in frozen["inputs"].items()}
    results, cache = {}, {}
    for person, donor in permutations(GROUPS, 2):
        key = f"{person}-from-{donor}"
        row = frozen["cases"][key]
        target = photos[f"{person}-cold"]["off"]
        expected = checked_rgb(directory / f"{key}-expected.png", row["expected_sha"])
        refs = {light: {state: (f"{donor}-{light}-{state}", photos[f"{donor}-{light}"][state]) for state in STATES}
                for light in LIGHTS}
        original = {light: {state: checked_rgb(directory / f"{key}-13-{light}-{state}.png",
                                              row["hashes"][f"13-{light}-{state}"]) for state in STATES}
                    for light in LIGHTS}
        area = masks(zones[person]["cold"], zones["screen"], target.shape[:2])
        dest = out / key
        dest.mkdir()
        results[key] = experiment(model, dest, target, expected, refs, original,
                                  lambda frames: score(target, expected, frames, area), cache)
        if results[key]["scores"]["frozen13"] != row["scores"]["13"]:
            raise ValueError("Changed Gros baseline score")
    return results


def run(out):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = DenseAnchorModel()
    example = np.tile(np.exp(-np.array([0., .1, .8, 1.]) / .08), (3, 1))
    equal, flexible = (transport(example, power).sum(axis=0) for power in ARMS.values())
    if not flexible[2:].sum() < equal[2:].sum():
        raise ValueError("Flexible reference weighting control failed")
    out.mkdir()
    bg = bg_suite(model, out)
    gros = gros_suite(model, out)
    protocol = Path(__file__).with_name("partial-region-match-protocol.md")
    json_to(out / "results.json", dict(bg_source=BG_SHA, gros_source=BASELINE_SHA, checkpoint=CHECKPOINT,
                                       descriptor=model.provenance, protocol=digest(protocol), strength=STRENGTH,
                                       examples=dict(equal=equal.tolist(), flexible=flexible.tolist()), bg=bg, gros=gros))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
