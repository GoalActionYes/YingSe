"""Check fixed single-photo color inference under a physical change of one lamp."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

from probe_matched_wall_warmth_grid_capacity import nearby_region
from probe_matched_wall_warmth_workbench import target_areas
from probe_paired_wall_warmth_coordinate import calibration, lab, measure, replay, shift
from probe_playraw_flash_room import RT_CLI, read
from probe_shared_authored_tone_admission import support
from probe_single_reference_warmth_plane import (
    PAIRED_EVIDENCE, estimate, fit_ab, make_exports, read_weight, region_mask, score_exports,
)
from probe_target_mirror_chroma import STRENGTH, checked_json, digest, json_to, save
from render_rawtherapee_sky_pair import check
from run_positional_warmth_trial import MANIFEST_SHA, read_manifest, read_photo
from spatial_render import finish_spatial
from warm_capture_preflight import RT_PROFILE, RT_RUNTIME


FAU_EVIDENCE = "30c35070d787c4bbdce966ff3d01ff2423929b9b4fe8a243bc01894e6f87cd5e"
SCENES = {
    "lion": (((95, 134), (105, 124), (151, 119), (157, 127), (147, 146), (145, 155), (128, 159),
              (112, 154), (104, 147)),
             ((289, 118), (359, 118), (388, 126), (395, 137), (387, 146), (353, 151), (312, 155), (297, 147))),
    "bmug": (((216, 224), (246, 225), (275, 223), (303, 217), (322, 209), (321, 224), (315, 236),
              (300, 244), (272, 252), (248, 255), (218, 249)),),
}
ROLES = {"base": "w_b-off", "captured": "r_b-off", "edited": "w_b-on", "captured_edited": "r_b-on"}
LOAD_ORDER = ("captured_edited", "base", "edited", "captured")


def load_scene(cli, icc, out, directory, spec, polygons):
    anonymous, labels, files = {}, {}, {}
    for index, role in enumerate(LOAD_ORDER):
        name = ROLES[role]
        light, state = name.split("-")
        record = spec["frames"][light]
        if not record["metrics"]["admitted"]:
            raise ValueError(f"The pinned reference {directory.name}/{name} did not qualify")
        path = directory / f"{name}.png"
        frame, _, receipt = read(path, False)
        if frame.shape != (260, 452, 3) or receipt != record["files"][state]:
            raise ValueError(f"Changed FAU render: {path.name}")
        if state == "off" and receipt["pixels_sha256"] != record["source_pixels"]:
            raise ValueError(f"FAU off state changed source pixels: {path.name}")
        key = f"image-{index}"
        labels[role] = key
        anonymous[key] = frame
        files[key] = dict(file_sha256=digest(path), pixels_sha256=receipt["pixels_sha256"])
    anchor = anonymous[labels["base"]]
    weight, areas, sampled = calibration(cli, icc, out, directory, spec, anchor)
    masks = [region_mask(anchor.shape, polygon) for polygon in polygons]
    material = np.logical_or.reduce(masks)
    return (anonymous, material, weight), labels, areas["wall_left"], dict(files=files, calibration=sampled), masks


def load_targets(base, paired, ledger, families):
    targets = {}
    for family, item in families.items():
        spec = item["target"]
        original = read_photo(base, spec["photos"]["off"], family)
        areas = target_areas(original, spec, False)
        weight = read_weight(paired / family / "target-calibration", ledger["cases"][family]["calibration"]["target"],
                             areas["wall_left"], areas["outside"])
        targets[family] = dict(original=original, areas=areas, weight=weight, specification=spec)
    return targets


def paired_control(base, out, family, target, prepared, labels, core):
    original, target_weight, anonymous, _, reference_weight = prepared
    spec = target["specification"]
    expected = finish_spatial(original, read_photo(base, spec["photos"]["on"], family), STRENGTH)
    frames, details = {}, {}
    for prior, edited in (("base", "edited"), ("captured", "captured_edited")):
        before, after = anonymous[labels[prior]], anonymous[labels[edited]]
        valid = support(before) & support(after) & (reference_weight >= .8)
        coverage = float(valid[core].mean())
        details[prior] = dict(core_coverage=coverage)
        if coverage < .8:
            continue
        offsets, _, _ = shift(before, after, reference_weight, core)
        chroma = offsets["lab_d50"]
        chroma[0] = 0
        image = finish_spatial(original, replay(original, target_weight, chroma, "lab_d50"), STRENGTH)
        path = out / f"paired-{prior}.png"
        details[prior].update(offset=chroma.tolist(), sha256=save(path, image))
        with Image.open(path) as decoded:
            frames[prior] = np.array(decoded)
    if len(frames) != 2:
        return dict(complete=False, details=details, reason="incomplete paired support")
    ring = nearby_region(original.shape, spec["extent"])
    result = measure(original, expected, frames, target["areas"], ring)
    return {**result, "details": details}


def light_diagnostics(data, labels, masks):
    anonymous, material, weight = data
    frames = {role: anonymous[key] for role, key in labels.items()}
    by_piece = {role: [estimate(photo, mask, weight)[1] for mask in masks] for role, photo in frames.items()}
    valid = material[::4, ::4].copy()
    for photo in frames.values():
        valid &= support(photo[::4, ::4])
    y, x = np.nonzero(valid)
    if not len(y):
        return dict(piece_support=by_piece, common_support=dict(points=0), projected_fraction={},
                    vectors={}, strong_mixed_light_pressure=False)
    height, width = material.shape
    colors = {role: lab(photo[::4, ::4][valid])[:, 1:] for role, photo in frames.items()}
    pairs = (("base", "captured"), ("base", "edited"), ("captured", "captured_edited"))
    vectors, counts = {}, {}
    for first, second in pairs:
        fits, counts = fit_ab(x * 4 / width, y * 4 / height, weight[::4, ::4][valid],
                              colors[second] - colors[first])
        if fits:
            vectors[f"{first}_to_{second}"] = np.array(fits["plane"]["lab_offset"][1:])
    projections = {}
    if vectors:
        physical = vectors["base_to_captured"]
        for name, vector in vectors.items():
            if name != "base_to_captured":
                energy = vector @ vector
                projections[name] = float(physical @ vector / energy) if energy else None
    return dict(piece_support=by_piece, common_support=counts, projected_fraction=projections,
                vectors={name: value.tolist() for name, value in vectors.items()},
                strong_mixed_light_pressure=any(value is not None and abs(value) >= .5 for value in projections.values()))


def run(base, paired, out):
    base, out = base.absolute(), out.absolute()
    cli = base / RT_RUNTIME / "rawtherapee-cli.exe"
    if not all(path.as_posix().isascii() for path in (cli, out)):
        raise ValueError("Map the runtime and output directory to ASCII drive paths")
    check(cli, RT_CLI)
    icc = check(cli.parent / "iccprofiles/output/RTv4_sRGB.icc", RT_PROFILE)
    source = base / "fau-shared-warmth-v1"
    frozen = checked_json(source / "results.json", FAU_EVIDENCE)
    families = read_manifest()["families"]
    targets = load_targets(base, paired, checked_json(paired / "results.json", PAIRED_EVIDENCE), families)
    out.mkdir()
    scenes, exports, queued, diagnostics = {}, {}, {}, {}
    for name, polygons in SCENES.items():
        scene = out / name
        scene.mkdir()
        data, labels, core, proof, masks = load_scene(
            cli, icc, scene / "calibration", source / "fau" / name, frozen["groups"][name], polygons
        )
        scenes[name] = proof
        diagnostics[name] = (data, labels, masks)
        for family, target in targets.items():
            path = scene / family
            path.mkdir()
            prepared = (target["original"], target["weight"], *data)
            predictions = make_exports(path, *prepared)
            exports[f"{name}/{family}"] = predictions
            queued[f"{name}/{family}"] = path, family, target, prepared, labels, core, predictions
    receipt = out / "prediction-receipt.json"
    json_to(receipt, exports)
    cases = {}
    for key, (path, family, target, prepared, labels, core, predictions) in queued.items():
        scored = score_exports(base, path, family, target["specification"],
                               (prepared, target["areas"], labels, predictions))
        control = paired_control(base, path, family, target, prepared, labels, core)
        cases[key] = dict(**scored, paired_control=control)
        print(key, {arm: value["complete"] for arm, value in scored["scores"].items()},
              "paired", control["complete"], flush=True)
    signals = {name: light_diagnostics(*entry) for name, entry in diagnostics.items()}
    json_to(out / "results.json", dict(manifest=MANIFEST_SHA, evidence=FAU_EVIDENCE, paired_evidence=PAIRED_EVIDENCE,
                                      fixed_offline_geometry=SCENES, sources=scenes, strength=STRENGTH,
                                      receipt=digest(receipt), cases=cases, light_diagnostics=signals,
                                      ready_to_test_localization=any(row["strong_mixed_light_pressure"] for row in signals.values())
                                      and all(row["scores"]["plane"]["complete"] for row in cases.values())))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "paired", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
