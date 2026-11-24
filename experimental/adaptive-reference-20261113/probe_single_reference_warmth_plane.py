"""Estimate and replay a signed local warmth change from one reference photo."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluate_positional_warmth_trial import difference, heldout_scores
from probe_matched_wall_warmth_grid_capacity import nearby_region
from probe_matched_wall_warmth_workbench import target_areas
from probe_paired_wall_warmth_coordinate import lab, replay
from probe_playraw_flash_room import read
from probe_playraw_flash_room_workbench import added_endpoints
from probe_shared_authored_tone_admission import support
from probe_target_mirror_chroma import STRENGTH, checked_json, digest, json_to, panel, save
from run_positional_warmth_trial import MANIFEST_SHA, ROLES, read_manifest, read_photo
from spatial_render import finish_spatial


PAIRED_EVIDENCE = "2ee188d6e9d592a09edb9b34a999a4ea87ced08bd71341395faaaeb2ca0de44d"
ARMS = ("constant", "plane")
SURFACES = {
    "mit": ((144, 335), (485, 317), (498, 411), (472, 460), (447, 519), (441, 655),
            (416, 777), (141, 803), (137, 575)),
    "beyond_rgb": ((757, 670), (829, 687), (913, 705), (1024, 730), (1024, 768),
                   (818, 768), (785, 741), (768, 710)),
}


def region_mask(shape, polygon):
    height, width = shape[:2]
    image = Image.new("1", (width, height))
    ImageDraw.Draw(image).polygon(polygon, fill=1)
    return np.array(image, dtype=bool)


def read_weight(directory, receipt, core, outside):
    views = {}
    for state, expected in receipt["samples"]["gray128"]["files"].items():
        path = directory / f"gray128-{state}.png"
        if digest(path) != expected:
            raise ValueError(f"Changed calibration frame: {path}")
        views[state], _, _ = read(path, False)
    delta = views["on"].astype(np.int16) - views["off"]
    direction = np.median(delta[core], axis=0)
    if not np.array_equal(direction, receipt["samples"]["gray128"]["delta"]):
        raise ValueError("Calibrated core color direction changed")
    weight = np.clip(delta @ direction / (direction @ direction), 0, 1)
    weight[outside] = 0
    return weight


def fit_ab(x, y, weight, values):
    constant = np.ones_like(weight)
    plane = np.column_stack((constant, x, y))
    explained, _, rank, _ = np.linalg.lstsq(plane, weight, rcond=None)
    residual = weight - plane @ explained
    spread = float(np.sqrt(np.mean(residual ** 2)))
    support = dict(weak=int(np.count_nonzero(weight <= .15)),
                   strong=int(np.count_nonzero(weight >= .80)), template_residual_rms=spread,
                   plane_rank=int(rank), points=int(len(weight)))
    if min(support["weak"], support["strong"]) < 24 or rank != 3 or spread < .10:
        return {}, support
    fits = {}
    for arm, backdrop in zip(ARMS, (constant[:, None], plane), strict=True):
        design = np.column_stack((backdrop, weight))
        params, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        fits[arm] = dict(lab_offset=[0., *params[-1].tolist()],
                         fit_residual_rms=float(np.sqrt(np.mean((values - design @ params) ** 2))))
    return fits, support


def estimate(first, mask, weight, after=None):
    valid = mask[::4, ::4] & support(first[::4, ::4])
    if after is not None:
        valid &= support(after[::4, ::4])
    y, x = np.nonzero(valid)
    if not len(y):
        return {}, dict(points=0, weak=0, strong=0)
    x, y = x * 4, y * 4
    height, width = first.shape[:2]
    values = lab(first[y, x])[:, 1:]
    if after is not None:
        values = lab(after[y, x])[:, 1:] - values
    fitted, counts = fit_ab(x / width, y / height, weight[y, x], values)
    return fitted, dict(sampled_valid_fraction=len(y) / int(mask[::4, ::4].sum()), **counts)


def prepare(base, calibration, family, specification, geometry, receipt):
    target = specification["target"]
    original = read_photo(base, target["photos"]["off"], family)
    target_area = target_areas(original, target, False)
    target_weight = read_weight(calibration / family / "target-calibration", receipt["target"],
                                target_area["wall_left"], target_area["outside"])
    anonymous = {}
    roles = {}
    for index, role in enumerate(("captured_edited", "base", "edited", "captured")):
        key = f"image-{index}"
        roles[role] = key
        anonymous[key] = read_photo(base, specification["reference"][role], family)
    image = anonymous["image-0"]
    reference_areas = target_areas(image, geometry, False)
    region = read_weight(calibration / family / "ref-calibration", receipt["reference"],
                         reference_areas["wall_left"], reference_areas["outside"])
    mask = region_mask(image.shape, SURFACES[family])
    return (original, target_weight, anonymous, mask, region), target_area, roles


def make_exports(out, original, target_weight, anonymous, mask, region):
    outputs = {}
    for key, image in anonymous.items():
        fits, statistics = estimate(image, mask, region)
        outputs[key] = dict(support=statistics, arms=fits)
        for arm, result in fits.items():
            shifted = replay(original, target_weight, np.asarray(result["lab_offset"]), "lab_d50")
            image_path = out / f"{arm}-{key}.png"
            result["sha256"] = save(image_path, finish_spatial(original, shifted, STRENGTH))
    return outputs


def score_exports(base, out, family, spec, saved):
    prepared, areas, roles, anonymous_predictions = saved
    original, _, anonymous, mask, region = prepared
    sources = {role: anonymous[roles[role]] for role in ROLES}
    outputs = {role: anonymous_predictions[roles[role]] for role in ROLES}
    expected = finish_spatial(original, read_photo(base, spec["photos"]["on"], family), STRENGTH)
    ring = nearby_region(original.shape, spec["extent"])
    images, scores = {}, {}
    for arm in ARMS:
        images[arm] = {}
        for role in ROLES:
            entry = outputs[role]["arms"].get(arm)
            if entry is None:
                continue
            path = out / f"{arm}-{roles[role]}.png"
            if digest(path) != entry["sha256"]:
                raise ValueError("Primary output was changed before scoring")
            with Image.open(path) as image:
                images[arm][role] = np.array(image)
            authored = expected if role in {"edited", "captured_edited"} else original
            value = images[arm][role]
            entry.update(nearby_guard_mae=difference(value, authored, ring),
                         added_endpoint_fraction=added_endpoints(original, value))
        if len(images[arm]) != len(ROLES):
            scores[arm] = dict(complete=False, reason="incomplete support")
            continue
        measured = heldout_scores(original, expected, images[arm], areas)
        passed = all(value["passed"] for value in measured.values()) and all(
            outputs[role]["arms"][arm]["added_endpoint_fraction"] <= .02 for role in ROLES
        )
        scores[arm] = dict(complete=passed, measured=measured)
    identity = dict.fromkeys(ROLES, original)
    pairs = (("base", "edited"), ("captured", "captured_edited"), ("base", "captured"))
    diagnostics = {}
    for first, second in pairs:
        fitted, counts = estimate(sources[first], mask, region, after=sources[second])
        diagnostics[f"{first}_to_{second}"] = dict(arms=fitted, support=counts)
    board = panel(out / "comparison.png", {
        "target": {"original": original, "authored": expected}, "reference": sources, **images
    })
    return dict(frames=outputs, labels=roles, scores=scores, paired_diagnostics=diagnostics,
                identity_control=heldout_scores(original, expected, identity, areas), panel=board)


def run(base, calibration, out):
    ledger = checked_json(calibration / "results.json", PAIRED_EVIDENCE)
    manifest = read_manifest()
    paths = manifest["receipts"]
    mit_path = "shared-local-warmth-v2/results.json"
    beyond_path = "beyond-rgb-near-white-v2/warm-verified-first/results.json"
    geometries = {
        "mit": checked_json(base / mit_path, paths[mit_path])["results"]["mit"]["14n_copyroom1"],
        "beyond_rgb": checked_json(base / beyond_path, paths[beyond_path]),
    }
    out.mkdir()
    saved = {}
    for family, spec in manifest["families"].items():
        location = out / family
        location.mkdir()
        prepared, area, labels = prepare(base, calibration, family, spec, geometries[family],
                                         ledger["cases"][family]["calibration"])
        outputs = make_exports(location, *prepared)
        saved[family] = prepared, area, labels, outputs
    receipt = out / "prediction-receipt.json"
    json_to(receipt, {family: item[-1] for family, item in saved.items()})
    results = {family: score_exports(base, out / family, family, spec["target"], saved[family])
               for family, spec in manifest["families"].items()}
    json_to(out / "results.json", dict(manifest=MANIFEST_SHA, paired_evidence=PAIRED_EVIDENCE, strength=STRENGTH,
                                      single_reference=True, fixed_offline_geometry=SURFACES,
                                      receipt=digest(receipt), cases=results))
    for name, item in results.items():
        print(name, {arm: value["complete"] for arm, value in item["scores"].items()}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "calibration", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
