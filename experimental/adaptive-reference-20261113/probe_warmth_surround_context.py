"""Measure a fixed local edit while keeping the core pixels identical."""
from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
from PIL import Image

from probe_fau_single_reference_warmth import FAU_EVIDENCE, load_targets
from probe_paired_wall_warmth_coordinate import lab
from probe_playraw_flash_room import RT_CLI, read
from probe_playraw_flash_room_workbench import added_endpoints
from probe_shared_authored_tone_admission import support
from probe_single_reference_warmth_plane import PAIRED_EVIDENCE, read_weight
from probe_target_mirror_chroma import checked_json, digest, json_to
from probe_warmth_lightness_response import AXIS, STRESS_EVIDENCE, source_frames
from render_rawtherapee_sky_pair import check, disabled, render
from run_positional_warmth_trial import MANIFEST_SHA, read_manifest
from warm_capture_preflight import RT_PROFILE, RT_RUNTIME


ARMS = ("original", "far_flat", "near_flat")


class IntakeError(ValueError):
    """The current source cannot be compared under the locked protocol."""


def distance_from_box(shape, box):
    y, x = np.ogrid[:shape[0], :shape[1]]
    left, top, right, bottom = np.ceil(box).astype(int)
    dx = np.maximum(left - x, x - (right - 1))
    dy = np.maximum(top - y, y - (bottom - 1))
    return np.maximum(np.maximum(dx, dy), 0)


def distance_inside_box(shape, box):
    y, x = np.ogrid[:shape[0], :shape[1]]
    left, top, right, bottom = np.ceil(box).astype(int)
    dx = np.minimum(x - left + 1, right - x)
    dy = np.minimum(y - top + 1, bottom - y)
    return np.maximum(np.minimum(dx, dy), 0)


def blend_flat(image, color, amount):
    weight = (1. - np.cos(np.pi * amount)) * .5
    return np.rint(image * (1. - weight[..., None]) + color * weight[..., None]).astype(np.uint8)


def surroundings(image, spec, core):
    left, top, right, bottom = spec["boxes"]["wall_left"]
    rim = max(4, int(min(right - left, bottom - top) / 4))
    color = np.rint(np.median(image[core], axis=0)).astype(np.uint8)
    near_bounds = (left - rim, top - rim, right + rim, bottom + rim)
    far_distance = distance_from_box(image.shape, spec["extent"])
    near_distance = distance_from_box(image.shape, near_bounds)
    clearance = distance_inside_box(image.shape, spec["extent"])
    far = blend_flat(image, color, np.minimum(far_distance / rim, 1.))
    near = blend_flat(far, color, np.minimum(np.minimum(near_distance, clearance) / rim, 1.))
    if not np.array_equal(near[far_distance > 0], far[far_distance > 0]):
        raise IntakeError("The near intervention changed pixels outside the edit extent")
    return {
        "original": (image, np.ones(core.shape, dtype=bool), None),
        "far_flat": (far, far_distance == 0, dict(keep_box=spec["extent"], feather_pixels=rim, fill_rgb=color.tolist())),
        "near_flat": (near, near_distance == 0, dict(keep_box=near_bounds, feather_pixels=rim, fill_rgb=color.tolist())),
    }


def export_case(cli, icc, folder, item):
    folder.mkdir()
    original, spec, areas = item["image"], item["spec"], item["areas"]
    core, outside = areas["wall_left"], areas["outside"]
    if not np.array_equal(outside, distance_from_box(original.shape, spec["extent"]) > 0):
        raise IntakeError("Flat replacement disagrees with the frozen edit extent")
    pp3 = {state: item["directory"] / f"{state}.pp3" for state in ("off", "on")}
    try:
        enabled = check(pp3["on"], item["profiles"]["on"])
        if check(pp3["off"], item["profiles"]["off"]) != disabled(enabled):
            raise ValueError("Unexpected difference between the two frozen editor states")
    except ValueError as error:
        raise IntakeError(str(error)) from error
    env = dict(os.environ, RT_SETTINGS=str(folder / "settings"), RT_CACHE=str(folder / "cache"), OMP_NUM_THREADS="1")
    readings, core_pairs = {}, {}
    previous = original
    for arm, (variant, kept, geometry) in surroundings(original, spec, core).items():
        group = folder / arm
        group.mkdir()
        if not (kept[core].all() and np.array_equal(original[kept], variant[kept])):
            raise IntakeError(f"Replacement changed retained pixels: {arm}")
        path = group / "input.png"
        Image.fromarray(variant).save(path, icc_profile=icc)
        images, files, pixels = {}, {"input": digest(path)}, {}
        for state, profile in pp3.items():
            output = group / f"{state}.png"
            render(cli, path, profile, output, env)
            data, _, evidence = read(output, False)
            if data.shape != variant.shape or evidence["icc_sha256"] != RT_PROFILE:
                raise IntakeError(f"Source and output color profile or shape differ: {arm}/{state}")
            images[state], files[state], pixels[state] = data, digest(output), evidence["pixels_sha256"]
        before, after = images["off"], images["on"]
        difference = np.abs(variant.astype(np.int16) - previous).max(axis=-1)
        zones = dict(in_edit=~outside, outside=outside)
        intervention = {name: dict(changed=int(np.count_nonzero(difference[area])),
                                   substantial_fraction=float((difference[area] >= 8).mean()),
                                   median_max_rgb8=float(np.median(difference[area])))
                        for name, area in zones.items()}
        affected = outside if arm == "far_flat" else ~outside & ~kept
        sufficient = arm == "original" or bool(affected.any()) and (difference[affected] >= 8).mean() >= .10
        endpoints = added_endpoints(before, after)
        outside_identity = bool(np.array_equal(before[outside], after[outside]))
        off_identity = bool(np.array_equal(variant, before))
        same_original = pixels == item["archive_pixels"] if arm == "original" else None
        readings[arm] = dict(files=files, pixels=pixels, geometry=geometry, input_changes_from_parent=intervention,
                             sufficient_intervention=bool(sufficient), off_identity=off_identity,
                             outside_identity=outside_identity, added_endpoint_fraction=endpoints,
                             matches_original_archive=same_original,
                             admitted=(off_identity and outside_identity and endpoints <= .02
                                       and (same_original if arm == "original" else bool(sufficient))))
        core_pairs[arm] = before[core], after[core]
        previous = variant
    weight = item["weight"][core]
    common = weight >= .8
    for before, after in core_pairs.values():
        common &= support(before) & support(after)
    coverage = float(common.mean())
    passed = coverage >= .8 and all(row["admitted"] for row in readings.values())
    receipt = dict(passed=passed, core_pixels=int(core.sum()), common_support=coverage, arms=readings,
                   profile=item["profiles"])
    json_to(folder / "qualification.json", receipt)
    if passed:
        for arm, (before, after) in core_pairs.items():
            shift = np.median((lab(after[common]) - lab(before[common])) / weight[common, None], axis=0)
            readings[arm].update(lab_shift=shift.tolist(), warmth=float(AXIS @ shift[1:]))
        receipt["warmth_change_from_parent"] = {
            newer: readings[newer]["warmth"] - readings[older]["warmth"]
            for older, newer in zip(ARMS, ARMS[1:])
        }
    return receipt


def gap_results(rows):
    all_gaps = {}
    for donor in ("lion", "bmug"):
        for role in ("base", "captured"):
            source = rows[f"{donor}-{role}"]
            for family in ("mit", "beyond_rgb"):
                target = rows[family]
                key = f"{donor}-{role}/{family}"
                if not source["passed"] or not target["passed"]:
                    all_gaps[key] = dict(eligible=False, reason="source or target failed intake")
                    continue
                gaps = {arm: target["arms"][arm]["warmth"] - source["arms"][arm]["warmth"] for arm in ARMS}
                original = abs(gaps["original"])
                all_gaps[key] = dict(
                    eligible=original >= 1., gaps=gaps,
                    closure={arm: 1. - abs(value) / original for arm, value in gaps.items() if arm != "original"}
                    if original >= 1. else {},
                    supported={arm: abs(value) <= .5 * original
                               for arm, value in gaps.items() if arm != "original" and original >= 1.},
                )
    counts = {arm: sum(row.get("supported", {}).get(arm, False) for row in all_gaps.values())
              for arm in ARMS[1:]}
    return dict(cases=all_gaps, explained=counts, universal={arm: count == len(all_gaps) for arm, count in counts.items()})


def build_cases(base, paired, stress_dir, manifest):
    source_root = base / "fau-shared-warmth-v1"
    source_ledger = checked_json(source_root / "results.json", FAU_EVIDENCE)
    stress = checked_json(stress_dir / "results.json", STRESS_EVIDENCE)
    donors = source_frames(source_root, stress, source_ledger)
    targets = load_targets(base, paired, checked_json(paired / "results.json", PAIRED_EVIDENCE), manifest["families"])
    cases = {}
    for name, scene in donors.items():
        spec, areas = scene["spec"], scene["areas"]
        weight = read_weight(stress_dir / name / "calibration", scene["receipt"], areas["wall_left"], areas["outside"])
        for role, light in (("base", "w_b"), ("captured", "r_b")):
            record = spec["frames"][light]["files"]
            cases[f"{name}-{role}"] = dict(image=scene["samples"][role], directory=scene["directory"], spec=spec,
                                          areas=areas, weight=weight, profiles=spec["profiles"],
                                          archive_pixels={state: record[state]["pixels_sha256"] for state in ("off", "on")})
    mit_path = "shared-local-warmth-v2/results.json"
    beyond_path = "warmth-source-holdout-v1/target-first/target.json"
    mit = checked_json(base / mit_path, manifest["receipts"][mit_path])["results"]["mit"]["everett_living2"]
    beyond = checked_json(base / beyond_path, manifest["receipts"][beyond_path])
    for family, profiles in (("mit", mit["profiles"]), ("beyond_rgb", beyond["profiles"])):
        target = targets[family]
        spec = target["specification"]
        paths = spec["photos"]
        cases[family] = dict(image=target["original"], directory=(base / paths["off"]["path"]).parent, spec=spec,
                             areas=target["areas"], weight=target["weight"], profiles=profiles,
                             archive_pixels={state: item["pixels_sha256"] for state, item in paths.items()})
    return cases


def run(base, paired, stress_dir, out):
    base, out = base.absolute(), out.absolute()
    cli = base / RT_RUNTIME / "rawtherapee-cli.exe"
    if not all(path.as_posix().isascii() for path in (cli, out)):
        raise ValueError("Map the runtime and output directory to ASCII drive paths")
    check(cli, RT_CLI)
    icc = check(cli.parent / "iccprofiles/output/RTv4_sRGB.icc", RT_PROFILE)
    cases = build_cases(base, paired, stress_dir, read_manifest())
    out.mkdir()
    results = {}
    for name, item in cases.items():
        try:
            results[name] = export_case(cli, icc, out / name, item)
        except IntakeError as error:
            results[name] = dict(passed=False, reason=str(error))
            json_to(out / name / "qualification.json", results[name])
    gaps = gap_results(results)
    json_to(out / "results.json", dict(manifest=MANIFEST_SHA, fau_archive=FAU_EVIDENCE, paired=PAIRED_EVIDENCE,
                                      stress=STRESS_EVIDENCE, rows=results, gap_closure=gaps))
    print({name: row["passed"] for name, row in results.items()}, flush=True)
    print("Explained directions:", gaps["explained"], flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "paired", "stress_dir", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
