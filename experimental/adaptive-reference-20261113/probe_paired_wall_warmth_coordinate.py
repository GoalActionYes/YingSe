"""Replay reference-only local warmth with an offline, calibrated region weight."""
from argparse import ArgumentParser
import os
from pathlib import Path

import colour
import numpy as np
from PIL import Image

from evaluate_positional_warmth_trial import difference, heldout_scores
from probe_matched_wall_warmth_grid_capacity import nearby_region
from probe_matched_wall_warmth_workbench import target_areas
from probe_playraw_flash_room import RT_CLI, read
from probe_playraw_flash_room_workbench import added_endpoints
from probe_shared_authored_tone_admission import support
from probe_target_mirror_chroma import STRENGTH, checked_json, digest, json_to, panel, save
from render_rawtherapee_sky_pair import check, disabled, render
from run_positional_warmth_trial import MANIFEST_SHA, read_manifest, read_photo
from spatial_render import finish_spatial
from warm_capture_preflight import RT_PROFILE, RT_RUNTIME


D50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
GRADED_ROLES = {"base": "edited", "captured": "captured_edited"}
ARMS = ("rgb8", "lab_d50")


def calibration(cli, icc, out, profile, spec, original):
    out.mkdir()
    enabled = profile / "on.pp3"
    off = profile / "off.pp3"
    on_data = check(enabled, spec["profiles"]["on"])
    if check(off, spec["profiles"]["off"]) != disabled(on_data):
        raise ValueError("Unexpected difference between editor profiles")
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "cache"), OMP_NUM_THREADS="1")
    areas = target_areas(original, spec, False)
    core, outside = areas["wall_left"], areas["outside"]
    data, weights = {}, {}
    colors = {"gray128": (128, 128, 128), "gray192": (192, 192, 192), "pale": (160, 150, 140)}
    for name, color in colors.items():
        source, first, second = (out / f"{name}-{tag}.png" for tag in ("input", "off", "on"))
        solid = np.empty_like(original)
        solid[...] = color
        Image.fromarray(solid).save(source, icc_profile=icc)
        render(cli, source, off, first, env)
        render(cli, source, enabled, second, env)
        before, _, a = read(first, False)
        after, _, b = read(second, False)
        if a["icc_sha256"] != RT_PROFILE or b["icc_sha256"] != RT_PROFILE or before.shape != original.shape:
            raise ValueError("Calibration changed size or output color profile")
        if before.shape != after.shape or not np.array_equal(before, solid):
            raise ValueError("Calibration off state changed the uniform input")
        delta = after.astype(np.int16) - before
        direction = np.median(delta[core], axis=0)
        outside_max = float(np.abs(delta[outside]).mean(axis=-1).max())
        if np.abs(direction).mean() < 3 or outside_max > 1:
            raise ValueError(f"Invalid flat-field reference for {name}: {direction}; outside {outside_max}")
        raw = delta @ direction / (direction @ direction)
        weight = np.clip(raw, 0, 1)
        weight[outside] = 0
        weights[name] = weight
        local = ~outside
        residual = np.abs(delta[local] - raw[local, None] * direction).mean(axis=-1)
        data[name] = dict(delta=direction.tolist(), outside_max=outside_max,
                               orthogonal_residual_p95_rgb8=float(np.quantile(residual, .95)),
                               clipped_weight_fraction=float(np.mean((raw[local] < 0) | (raw[local] > 1))),
                               core_weight_p05=float(np.quantile(weight[core], .05)),
                               files={tag: digest(path) for tag, path in (("off", first), ("on", second))})
    differences = {name: float(np.quantile(np.abs(value[~outside] - weights["gray128"][~outside]), .95))
                   for name, value in weights.items() if name != "gray128"}
    receipt = dict(samples=data, weight_difference_p95=differences, profiles=spec["profiles"],
                   passed=differences["gray192"] <= .10)
    json_to(out / "calibration.json", receipt)
    if not receipt["passed"]:
        raise ValueError(f"Flat-field weights change with brightness: P95={differences['gray192']:.5f}")
    return weights["gray128"], areas, receipt


def lab(rgb):
    xyz = colour.sRGB_to_XYZ(rgb / 255, illuminant=D50, chromatic_adaptation_transform="Bradford")
    return colour.XYZ_to_Lab(xyz, illuminant=D50)


def rgb_from_lab(values):
    xyz = colour.Lab_to_XYZ(values, illuminant=D50)
    return colour.XYZ_to_sRGB(xyz, illuminant=D50, chromatic_adaptation_transform="Bradford") * 255


def shift(before, after, weight, core):
    valid = core & support(before) & support(after) & (weight >= .8)
    coverage = float(valid.sum() / core.sum())
    if coverage < .8:
        raise ValueError(f"Low edit support in reference: {coverage:.2%}")
    a, b, w = before[valid], after[valid], weight[valid, None]
    offsets = {
        "rgb8": np.median((b.astype(np.int16) - a) / w, axis=0),
        "lab_d50": np.median((lab(b) - lab(a)) / w, axis=0),
    }
    return offsets, coverage, valid


def replay(image, weight, change, arm):
    active = weight > 0
    values = image[active]
    offset = weight[active, None] * change
    if arm == "lab_d50":
        moved = rgb_from_lab(lab(values) + offset)
    else:
        moved = values + offset
    adjusted = image.copy()
    adjusted[active] = np.rint(np.clip(moved, 0, 255)).astype(np.uint8)
    return adjusted


def prepare_scene(base, cli, icc, out, family, spec, locations):
    source_spec, target_spec = locations
    references = {role: read_photo(base, entry, family) for role, entry in spec["reference"].items()}
    original = read_photo(base, spec["target"]["photos"]["off"], family)
    source_dir = base / spec["reference"]["base"]["path"]
    target_dir = base / spec["target"]["photos"]["off"]["path"]
    if spec["target"]["photos"]["off"]["scene"] == spec["reference"]["base"]["scene"]:
        raise ValueError("Comparison requires distinct scenes within each capture family")
    source_weight, source_areas, source_cal = calibration(
        cli, icc, out / "ref-calibration", source_dir.parent, source_spec, references["base"])
    target_weight, areas, target_cal = calibration(
        cli, icc, out / "target-calibration", target_dir.parent, target_spec, original)
    return (original, references, source_weight, target_weight, source_areas["wall_left"], areas,
            dict(reference=source_cal, target=target_cal))


def make_exports(out, prepared):
    original, references, source_weight, target_weight, source_core, _, _ = prepared
    hashes, replays = {arm: {} for arm in ARMS}, {}
    control = save(out / "off.png", original)
    for state, edited_role in GRADED_ROLES.items():
        offsets, coverage, valid = shift(references[state], references[edited_role], source_weight, source_core)
        replays[state] = dict(coverage=coverage, offsets={name: value.tolist() for name, value in offsets.items()},
                             source_core_reconstruction_rgb8={})
        for arm, value in offsets.items():
            native = replay(original, target_weight, value, arm)
            change = finish_spatial(original, native, STRENGTH)
            hashes[arm][state] = save(out / f"{arm}-{state}.png", change)
            reconstruction = replay(references[state], source_weight, value, arm)
            replays[state]["source_core_reconstruction_rgb8"][arm] = difference(
                reconstruction, references[edited_role], valid)
    return dict(offset=replays, files=hashes, off_sha256=control)


def measure(original, expected, frames, areas, region):
    states = dict(base=original, captured=original, edited=frames["base"], captured_edited=frames["captured"])
    measured = heldout_scores(original, expected, states, areas)
    endpoints = {name: added_endpoints(original, frame) for name, frame in frames.items()}
    return dict(measured=measured, added_endpoint_fraction=endpoints,
                nearby_guard_mae={name: difference(frame, expected, region) for name, frame in frames.items()},
                complete=all(value["passed"] for value in measured.values()) and max(endpoints.values()) <= .02)


def score_exports(base, out, family, spec, prepared, metadata):
    original, references, _, target_weight, _, areas, calibration_receipt = prepared
    authored = read_photo(base, spec["photos"]["on"], family)
    expected = finish_spatial(original, authored, STRENGTH)
    region = nearby_region(original.shape, spec["extent"])
    off = np.array(Image.open(out / "off.png"))
    if not np.array_equal(original, off):
        raise ValueError("The archived off state differs from the target")
    photos, scores = {}, {}
    for arm, hashes in metadata["files"].items():
        photos[arm] = {}
        for state in GRADED_ROLES:
            path = out / f"{arm}-{state}.png"
            if digest(path) != hashes[state]:
                raise ValueError("Archived output changed before scoring")
            with Image.open(path) as image:
                photos[arm][state] = np.array(image)
        scores[arm] = measure(original, expected, photos[arm], areas, region)
    target_offsets, target_coverage, _ = shift(original, authored, target_weight, areas["wall_left"])
    controls = {}
    for arm, value in target_offsets.items():
        control = finish_spatial(original, replay(original, target_weight, value, arm), STRENGTH)
        path = out / f"target-fit-{arm}.png"
        control_sha = save(path, control)
        with Image.open(path) as image:
            archived = np.array(image)
        controls[arm] = dict(offset=value.tolist(), core_coverage=target_coverage, sha256=control_sha,
                             **measure(original, expected, dict(base=archived, captured=archived), areas, region))
    board = panel(out / "comparison.png", {"target": {"original": original, "authored": expected},
                                          "reference": references, **photos})
    return {**metadata, "calibration": calibration_receipt, "scores": scores, "target_fit": controls, "panel": board}


def run(base, out, replay_only=False):
    base, out = base.absolute(), out.absolute()
    manifest = read_manifest()
    cli = base / RT_RUNTIME / "rawtherapee-cli.exe"
    if not out.as_posix().isascii() or not cli.as_posix().isascii():
        raise ValueError("Map the research and output paths to an ASCII drive")
    check(cli, RT_CLI)
    icc = check(cli.parent / "iccprofiles/output/RTv4_sRGB.icc", RT_PROFILE)
    if colour.__version__ != "0.4.7":
        raise ValueError("Use the pinned Colour Science wheel")
    paths = manifest["receipts"]
    mit_path = "shared-local-warmth-v2/results.json"
    beyond_ref_path = "beyond-rgb-near-white-v2/warm-verified-first/results.json"
    beyond_target_path = "warmth-source-holdout-v1/target-first/target.json"
    mit, beyond_ref, beyond_target = [
        checked_json(base / path, paths[path]) for path in (mit_path, beyond_ref_path, beyond_target_path)
    ]
    mit = mit["results"]["mit"]
    registry = {
        "mit": (mit["14n_copyroom1"], mit["everett_living2"]),
        "beyond_rgb": (beyond_ref, beyond_target),
    }
    out.mkdir()
    prepared, results = {}, {}
    for name, spec in manifest["families"].items():
        (out / name).mkdir()
        prepared[name] = prepare_scene(base, cli, icc, out / name, name, spec, registry[name])
        results[name] = make_exports(out / name, prepared[name])
    receipt = out / "replay-receipt.json"
    json_to(receipt, results)
    if replay_only:
        return
    evidence = {
        name: score_exports(base, out / name, name, spec["target"], prepared[name], results[name])
        for name, spec in manifest["families"].items()
    }
    json_to(out / "results.json", dict(scope="offline with paired reference and relocated editor geometry",
                                      manifest=MANIFEST_SHA, strength=STRENGTH, editor=RT_CLI, profile=RT_PROFILE,
                                      colour=colour.__version__, replay_receipt=digest(receipt), cases=evidence))
    for name, item in evidence.items():
        print(name, {arm: value["complete"] for arm, value in item["scores"].items()}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--replay-only", action="store_true")
    run(**vars(parser.parse_args()))
