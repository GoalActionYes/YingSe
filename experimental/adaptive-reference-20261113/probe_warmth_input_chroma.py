"""Test whether baseline chroma, at fixed lightness, explains weak warmth replay."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_fau_single_reference_warmth import FAU_EVIDENCE, load_targets
from probe_paired_wall_warmth_coordinate import replay, rgb_from_lab
from probe_playraw_flash_room import RT_CLI
from probe_single_reference_warmth_plane import PAIRED_EVIDENCE, read_weight
from probe_target_mirror_chroma import STRENGTH, checked_json, digest, json_to, save
from probe_warmth_lightness_response import AXIS, STRESS_EVIDENCE, core_color, score_cases, source_frames, test_scope
from render_rawtherapee_sky_pair import check
from run_positional_warmth_trial import MANIFEST_SHA, read_manifest
from spatial_render import finish_spatial
from warm_capture_preflight import RT_PROFILE, RT_RUNTIME


FIXED_L = 60
ANCHOR = "anchor"


def prepare_palette(stress_dir, scenes, targets):
    palette = {ANCHOR: [3., 9.]}
    donor_weights = {}
    for scene, spec in scenes.items():
        core = spec["areas"]["wall_left"]
        weight = read_weight(stress_dir / scene / "calibration", spec["receipt"], core, spec["areas"]["outside"])
        donor_weights[scene] = weight
        for role, photo in spec["samples"].items():
            median = core_color(photo, weight, core)
            palette[f"{scene}-{role}"] = median[1:] if median is not None else None
    for name, target in targets.items():
        median = core_color(target["original"], target["weight"], target["areas"]["wall_left"])
        palette[name] = median[1:] if median is not None else None
    return palette, donor_weights


def calibrate(cli, icc, out, blocks, palette):
    proof, profiles = {}, {}
    for block_name, block in blocks.items():
        holder = out / f"block-{block_name}"
        holder.mkdir()
        rows, observed = {}, {}
        rendered_codes = {}
        for key, chroma in palette.items():
            if chroma is None:
                rows[key] = dict(passed=False, reason="missing measured core chroma")
                continue
            rgb = rgb_from_lab(np.array((FIXED_L, *chroma)))
            if np.any((rgb < 0) | (rgb > 255)):
                rows[key] = dict(passed=False, reason="requested chroma outside sRGB", requested_rgb=rgb.tolist())
                continue
            code = tuple(np.rint(rgb).astype(int).tolist())
            if code in rendered_codes:
                original = rendered_codes[code]
                record = {**rows[original], "checks": dict(rows[original]["checks"]), "same_render_as": original}
                observed[key] = observed[original]
            else:
                values, (weight, samples) = test_scope(cli, icc, holder / key, *block, levels=(FIXED_L,), chroma=chroma)
                record = values[FIXED_L]
                observed[key] = samples[FIXED_L]
                rendered_codes[code] = key
            record["requested_rgb"] = rgb.tolist()
            record["rgb8"] = code
            record["checks"]["input_chroma"] = float(np.abs(np.asarray(record["off_lab"][1:]) - chroma).max()) <= .65
            record["checks"]["fixed_lightness"] = abs(record["off_lab"][0] - FIXED_L) <= .30
            record["passed"] = all(record["checks"].values())
            rows[key] = record
        proof[block_name] = rows
        if not all(row["passed"] for row in rows.values()):
            profiles[block_name] = dict(passed=False, reason="one or more palette colors failed intake")
            continue
        common = np.logical_and.reduce([flag for flag, _ in observed.values()])
        coverage = float(common.mean())
        measured_l = [row["off_lab"][0] for row in rows.values()]
        entry = profiles[block_name] = dict(common_support=coverage, passed=coverage >= .8,
                                            lightness_spread=max(measured_l) - min(measured_l))
        if not entry["passed"]:
            continue
        responses = {name: np.median(delta[common, 1:] / weight[common, None], axis=0) for name, (_, delta) in observed.items()}
        strengths = {name: float(vector @ AXIS) for name, vector in responses.items()}
        valid = all(value > 0 for value in strengths.values())
        entry.update(passed=valid, vector={name: value.tolist() for name, value in responses.items()},
                     strength=strengths,
                     cosine={name: strengths[name] / float(np.linalg.norm(vector)) if vector.any() else None
                             for name, vector in responses.items()})
        if valid:
            entry["relative"] = {name: value / strengths[ANCHOR] for name, value in strengths.items()}
    shared = dict(passed=False, profiles=profiles)
    if all(block["passed"] for block in profiles.values()):
        gains = np.array([[measure["relative"][name] for name in palette] for measure in profiles.values()])
        spread = float(np.ptp(gains, axis=0).max())
        shared.update(passed=spread <= .10, block_ratio_spread=spread,
                      response=dict(zip(palette, gains.mean(axis=0).tolist(), strict=True)))
    return proof, shared


def replay_palette(out, archive, fixed_pairs, targets, palette, curve):
    rendered = {}
    for case, roles in fixed_pairs.items():
        scene, family = case.split("/")
        target = targets[family]
        folder = out / case
        folder.mkdir(parents=True)
        output = rendered[case] = {}
        for role, spec in roles.items():
            key = f"{scene}-{role}"
            stored = archive / case / f"paired-{role}.png"
            if spec["sha256"] != digest(stored):
                raise ValueError(f"Frozen paired render changed: {stored}")
            item = output[role] = dict(source_chroma=palette[key], target_chroma=palette[family],
                                       baseline_offset=spec["offset"], archived_output=spec["sha256"])
            if curve["passed"]:
                factors = curve["response"]
                factor = factors[family] / factors[key]
                offset = np.asarray(spec["offset"]) * factor
                original = target["original"]
                frame = replay(original, target["weight"], offset, "lab_d50")
                item.update(factor=factor, offset=offset.tolist(),
                            sha256=save(folder / f"compensated-{role}.png", finish_spatial(original, frame, STRENGTH)))
    return rendered


def run(base, paired, stress_dir, out, replay_only=False):
    base, out = base.absolute(), out.absolute()
    cli = base / RT_RUNTIME / "rawtherapee-cli.exe"
    if not all(path.as_posix().isascii() for path in (cli, out)):
        raise ValueError("Map the runtime and output directory to ASCII drive paths")
    check(cli, RT_CLI)
    icc = check(cli.parent / "iccprofiles/output/RTv4_sRGB.icc", RT_PROFILE)
    prior = checked_json(stress_dir / "results.json", STRESS_EVIDENCE)
    source = base / "fau-shared-warmth-v1"
    manifest = read_manifest()
    donors = source_frames(source, prior, checked_json(source / "results.json", FAU_EVIDENCE))
    targets = load_targets(base, paired, checked_json(paired / "results.json", PAIRED_EVIDENCE), manifest["families"])
    palette, donor_weights = prepare_palette(stress_dir, donors, targets)
    lion, mit = donors["lion"], targets["mit"]
    target_file = base / mit["specification"]["photos"]["off"]["path"]
    locations = "shared-local-warmth-v2/results.json"
    verified = checked_json(base / locations, manifest["receipts"][locations])["results"]["mit"]["everett_living2"]
    blocks = {
        "lion": (lion["directory"], lion["spec"]["profiles"], lion["areas"], donor_weights["lion"]),
        "mit": (target_file.parent, verified["profiles"], mit["areas"], mit["weight"]),
    }
    out.mkdir()
    calibration, curve = calibrate(cli, icc, out, blocks, palette)
    fixed_pairs = {
        name: {role: {key: item[key] for key in ("offset", "sha256")}
               for role, item in case["paired_control"]["details"].items()}
        for name, case in prior["cases"].items()
    }
    exports = replay_palette(out, stress_dir, fixed_pairs, targets, palette, curve)
    receipt = out / "prediction-receipt.json"
    json_to(receipt, exports)
    ready = len(exports) == 4 and all(len(roles) == 2 and all("sha256" in item for item in roles.values())
                                      for roles in exports.values())
    scores = score_cases(base, out, stress_dir, exports, prior["cases"], targets) if ready and not replay_only else {}
    json_to(out / "results.json", dict(manifest=MANIFEST_SHA, stress=STRESS_EVIDENCE, nominal_lightness=FIXED_L,
                                      strength=STRENGTH, palette=palette, calibration=calibration, curve=curve,
                                      complete_replay=ready, receipt=digest(receipt), cases=scores,
                                      sufficient=bool(scores) and all(result["compensated"]["complete"]
                                                                     and not result["lost_previously_passing_areas"]
                                                                     for result in scores.values())))
    print(f"Calibrated and rendered eight directions: {ready}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "paired", "stress_dir", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--replay-only", action="store_true")
    run(**vars(parser.parse_args()))
