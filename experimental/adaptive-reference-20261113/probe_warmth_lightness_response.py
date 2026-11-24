"""Test a lightness-only correction for replaying the fixed local warmth edit."""
from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
from PIL import Image

from probe_fau_single_reference_warmth import FAU_EVIDENCE, load_targets
from probe_matched_wall_warmth_grid_capacity import nearby_region
from probe_matched_wall_warmth_workbench import target_areas
from probe_paired_wall_warmth_coordinate import lab, measure, replay, rgb_from_lab
from probe_playraw_flash_room import RT_CLI, read
from probe_playraw_flash_room_workbench import added_endpoints
from probe_shared_authored_tone_admission import support
from probe_single_reference_warmth_plane import PAIRED_EVIDENCE, read_weight
from probe_target_mirror_chroma import STRENGTH, checked_json, digest, json_to, save
from render_rawtherapee_sky_pair import check, disabled, render
from run_positional_warmth_trial import MANIFEST_SHA, read_manifest, read_photo
from spatial_render import finish_spatial
from warm_capture_preflight import RT_PROFILE, RT_RUNTIME


STRESS_EVIDENCE = "a912dc48d74a61f43e4db3d434c0561d9eedf28967602448d24d5ff754e8164f"
LEVELS = (30, 40, 50, 60, 70, 80, 90)
CHROMA = (3, 9)
AXIS = np.array((1600., 2800.)) / np.hypot(1600., 2800.)


def source_frames(base, stress, frozen):
    scenes = {}
    for scene in ("lion", "bmug"):
        directory = base / "fau" / scene
        spec = frozen["groups"][scene]
        samples = {}
        for role, light in (("base", "w_b"), ("captured", "r_b")):
            path = directory / f"{light}-off.png"
            photo, _, receipt = read(path, False)
            if receipt != spec["frames"][light]["files"]["off"]:
                raise ValueError(f"Changed source pixels or color profile: {path}")
            samples[role] = photo
        areas = target_areas(samples["base"], spec, False)
        calibration = stress["sources"][scene]["calibration"]
        scenes[scene] = dict(samples=samples, directory=directory, spec=spec, areas=areas, receipt=calibration)
    return scenes


def test_scope(cli, icc, out, profile, profiles, areas, weight, levels=LEVELS, chroma=CHROMA):
    out.mkdir()
    pp3 = {state: profile / f"{state}.pp3" for state in ("off", "on")}
    on_data = check(pp3["on"], profiles["on"])
    if check(pp3["off"], profiles["off"]) != disabled(on_data):
        raise ValueError("Profiles differ beyond the local switch")
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "cache"), OMP_NUM_THREADS="1")
    core, outside = areas["wall_left"], areas["outside"]
    results, samples = {}, {}
    for level in levels:
        srgb = rgb_from_lab(np.array((level, *chroma), dtype=float))
        if np.any((srgb < 0) | (srgb > 255)):
            raise ValueError(f"Registered uniform input outside display gamut at L*={level}")
        pixel = np.rint(srgb).astype(np.uint8)
        flat = np.broadcast_to(pixel, (*core.shape, 3))
        source = out / f"{level}-input.png"
        Image.fromarray(flat).save(source, icc_profile=icc)
        images, files = {}, {}
        for role, settings in pp3.items():
            path = out / f"{level}-{role}.png"
            render(cli, source, settings, path, env)
            photo, _, receipt = read(path, False)
            if photo.shape != flat.shape or receipt["icc_sha256"] != RT_PROFILE:
                raise ValueError("Uniform export changed dimensions or color profile")
            images[role] = photo
            files[role] = digest(path)
        before, after = images["off"], images["on"]
        valid = core & support(before) & support(after) & (weight >= .8)
        coverage = float(valid.sum() / core.sum())
        measured = lab(pixel)
        tone_error = float(np.abs(measured[1:] - chroma).max())
        protect = bool(np.array_equal(before[outside], after[outside]))
        endpoints = added_endpoints(before, after)
        delta = np.median((lab(after[valid]) - lab(before[valid])) / weight[valid, None], axis=0) if valid.any() else None
        passes = dict(input_chroma=tone_error <= .65, off_identity=bool(np.array_equal(before, flat)),
                      core_coverage=coverage >= .8, outside_identity=protect, endpoints=endpoints <= .02)
        results[level] = dict(off_lab=measured.tolist(), on_core_lab_shift=delta.tolist() if delta is not None else None,
                              axis_strength=float(AXIS @ delta[1:]) if delta is not None else None,
                              core_coverage=coverage, added_endpoint_fraction=endpoints, checks=passes,
                              passed=all(passes.values()), files=files)
        samples[level] = valid[core], lab(after[core]) - lab(before[core])
    return results, (weight[core], samples)


def response_curve(rows, observations):
    levels = [level for level in LEVELS if all(group[level]["passed"] for group in rows.values())]
    if len(levels) < 6 or 60 not in levels:
        return dict(passed=False, valid_levels=levels, reason="insufficient common lightness points")
    strengths, coverage = [], {}
    for name, (weight, samples) in observations.items():
        common = np.logical_and.reduce([samples[level][0] for level in levels])
        coverage[name] = float(common.mean())
        if coverage[name] < .8:
            return dict(passed=False, valid_levels=levels, common_support=coverage, reason="insufficient common pixels")
        strengths.append([float(AXIS @ np.median(samples[level][1][common, 1:] / weight[common, None], axis=0))
                          for level in levels])
    strengths = np.asarray(strengths)
    if (strengths <= 0).any():
        return dict(passed=False, valid_levels=levels, reason="nonpositive response along the registered direction")
    normalized = strengths / strengths[:, [levels.index(60)]]
    spread = float(np.ptp(normalized, axis=0).max())
    axis = np.mean([[group[level]["off_lab"][0] for level in levels] for group in rows.values()], axis=0)
    if not np.all(np.diff(axis) > 0):
        raise ValueError("Calibrated lightness axis is not ordered")
    return dict(passed=spread <= .10, valid_levels=levels, common_support=coverage,
                block_ratio_spread=spread, real_lightness=axis.tolist(),
                relative_strength=normalized.mean(axis=0).tolist(), absolute_block_strength=strengths.tolist())


def core_color(photo, weight, core):
    valid = core & support(photo) & (weight >= .8)
    if valid.sum() < .8 * core.sum():
        return None
    return np.median(lab(photo[valid]), axis=0).tolist()


def compensated_offset(signal, source_l, target_l, curve):
    if not curve["passed"] or source_l is None or target_l is None:
        return None, None
    axis, gain = curve["real_lightness"], curve["relative_strength"]
    if not (axis[0] <= min(source_l, target_l) and max(source_l, target_l) <= axis[-1]):
        return None, None
    source_gain, target_gain = np.interp([source_l, target_l], axis, gain)
    factor = float(target_gain / source_gain)
    return np.asarray(signal) * factor, factor


def replay_cases(out, archived, stored, scenes, targets, curve):
    exports = {}
    for key, control in stored.items():
        scene, family = key.split("/")
        donor, target = scenes[scene], targets[family]
        directory = out / scene / family
        directory.mkdir(parents=True)
        core = donor["areas"]["wall_left"]
        weight = read_weight(archived / scene / "calibration", donor["receipt"], core, donor["areas"]["outside"])
        target_lab = core_color(target["original"], target["weight"], target["areas"]["wall_left"])
        target_l = target_lab[0] if target_lab is not None else None
        details = {}
        for role, record in control.items():
            source_lab = core_color(donor["samples"][role], weight, core)
            source_l = source_lab[0] if source_lab is not None else None
            prior = archived / scene / family / f"paired-{role}.png"
            if digest(prior) != record["sha256"]:
                raise ValueError("Archived paired rendering changed")
            change, factor = compensated_offset(record["offset"], source_l, target_l, curve)
            entry = dict(source_lab=source_lab, target_lab=target_lab, factor=factor, before=record["offset"],
                         archived_output=record["sha256"])
            if change is not None:
                frame = replay(target["original"], target["weight"], change, "lab_d50")
                entry.update(after=change.tolist(), sha256=save(directory / f"compensated-{role}.png",
                                                                finish_spatial(target["original"], frame, STRENGTH)))
            details[role] = entry
        exports[key] = details
    return exports


def score_cases(base, out, archived, cases, controls, targets):
    results = {}
    for key, roles in cases.items():
        _, family = key.split("/")
        target = targets[family]
        original, spec, areas = target["original"], target["specification"], target["areas"]
        authored = read_photo(base, spec["photos"]["on"], family)
        goal = finish_spatial(original, authored, STRENGTH)
        ring = nearby_region(original.shape, spec["extent"])
        new, old = {}, {}
        for role, entry in roles.items():
            with Image.open(archived / key / f"paired-{role}.png") as photo:
                old[role] = np.array(photo)
            if "sha256" in entry:
                path = out / key / f"compensated-{role}.png"
                if entry["sha256"] != digest(path):
                    raise ValueError("Compensated output changed before scoring")
                with Image.open(path) as photo:
                    new[role] = np.array(photo)
        prior = measure(original, goal, old, areas, ring)
        expected = controls[key]["paired_control"]
        if any(expected[field] != value for field, value in prior.items()):
            raise ValueError("Archived metrics no longer reproduce")
        result = measure(original, goal, new, areas, ring) if len(new) == 2 else dict(complete=False, reason="missing curve support")
        lost = [name for name, row in prior["measured"].items()
                if row["passed"] and not result.get("measured", {}).get(name, {}).get("passed", False)]
        old_projection = prior["measured"]["wall_left"]["projection"]
        new_projection = result["measured"]["wall_left"]["projection"]
        levels = {role: dict(prior_projection=a, gained_projection=b, needs_at_least=(.75 / a if a > 0 else None))
                  for role, a, b in zip(("base", "captured"), old_projection, new_projection, strict=True)}
        reason = ("pass" if result["complete"] else
                  "below minimum warmth" if min(new_projection) < .75 else "other image gate")
        results[key] = dict(inputs=roles, baseline=prior, compensated=result, lost_previously_passing_areas=lost,
                            response_by_light=levels, decision=reason)
        print(key, new_projection, reason, flush=True)
    return results


def run(base, paired, stress_dir, out, replay_only=False):
    base, out = base.absolute(), out.absolute()
    cli = base / RT_RUNTIME / "rawtherapee-cli.exe"
    if not all(path.as_posix().isascii() for path in (cli, out)):
        raise ValueError("Map the runtime and output directory to ASCII drive paths")
    check(cli, RT_CLI)
    icc = check(cli.parent / "iccprofiles/output/RTv4_sRGB.icc", RT_PROFILE)
    prior = checked_json(stress_dir / "results.json", STRESS_EVIDENCE)
    source = base / "fau-shared-warmth-v1"
    frozen = checked_json(source / "results.json", FAU_EVIDENCE)
    manifest = read_manifest()
    targets = load_targets(base, paired, checked_json(paired / "results.json", PAIRED_EVIDENCE), manifest["families"])
    scenes = source_frames(source, prior, frozen)
    target = targets["mit"]
    target_path = base / target["specification"]["photos"]["off"]["path"]
    parent = checked_json(base / "shared-local-warmth-v2/results.json",
                          manifest["receipts"]["shared-local-warmth-v2/results.json"])
    lion = scenes["lion"]
    near = read_weight(stress_dir / "lion" / "calibration", lion["receipt"],
                       lion["areas"]["wall_left"], lion["areas"]["outside"])
    out.mkdir()
    blocks = {
        "lion": (lion["directory"], lion["spec"]["profiles"], lion["areas"], near),
        "mit": (target_path.parent, parent["results"]["mit"]["everett_living2"]["profiles"],
                target["areas"], target["weight"]),
    }
    calibrated = {name: test_scope(cli, icc, out / f"level-{name}", *block) for name, block in blocks.items()}
    rows = {name: row for name, (row, _) in calibrated.items()}
    curve = response_curve(rows, {name: data for name, (_, data) in calibrated.items()})
    controls = {}
    if curve["passed"] and np.ptp(curve["relative_strength"]) > .05:
        for name, block in blocks.items():
            for label, chroma in (("lower", (2, 8)), ("upper", (4, 10))):
                sample, _ = test_scope(cli, icc, out / f"chroma-{name}-{label}", *block, levels=(60,), chroma=chroma)
                controls[f"{name}/{label}"] = sample[60]
    fixed_pairs = {
        key: {role: {field: record[field] for field in ("sha256", "offset")}
              for role, record in case["paired_control"]["details"].items()}
        for key, case in prior["cases"].items()
    }
    cases = replay_cases(out, stress_dir, fixed_pairs, scenes, targets, curve)
    receipt = out / "prediction-receipt.json"
    json_to(receipt, cases)
    ready = len(cases) == 4 and all(len(roles) == 2 and all("sha256" in item for item in roles.values())
                                    for roles in cases.values())
    scores = score_cases(base, out, stress_dir, cases, prior["cases"], targets) if ready and not replay_only else {}
    json_to(out / "results.json", dict(manifest=MANIFEST_SHA, stress=STRESS_EVIDENCE, strength=STRENGTH,
                                      calibration=rows, curve=curve, quantization_controls=controls,
                                      receipt=digest(receipt), complete_replay=ready, cases=scores,
                                      sufficient=bool(scores) and all(result["compensated"]["complete"]
                                                                     and not result["lost_previously_passing_areas"]
                                                                     for result in scores.values())))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "paired", "stress_dir", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--replay-only", action="store_true")
    run(**vars(parser.parse_args()))
