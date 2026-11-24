"""Test a single RawTherapee scope change and replay the measured reference effect."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import re

import numpy as np

from probe_fau_single_reference_warmth import FAU_EVIDENCE, load_targets
from probe_paired_wall_warmth_coordinate import lab, replay
from probe_playraw_flash_room import RT_CLI, read
from probe_playraw_flash_room_workbench import added_endpoints
from probe_shared_authored_tone_admission import support
from probe_single_reference_warmth_plane import read_weight
from probe_target_mirror_chroma import STRENGTH, checked_json, digest, json_to, save
from probe_warmth_lightness_response import AXIS, STRESS_EVIDENCE, score_cases, source_frames
from probe_warmth_surround_context import ARMS
from render_rawtherapee_sky_pair import check, disabled, render
from run_positional_warmth_trial import MANIFEST_SHA, read_manifest
from spatial_render import finish_spatial
from warm_capture_preflight import RT_PROFILE, RT_RUNTIME


SURROUND_EVIDENCE = "0b6f68c908edbee37309f9f6afb6058678db4acb79700b7f17cd27b16c874a3f"
INPUT_PACKET = "b9a4cbe0cdcc73c26bc2886b3dcc3884b4eb1c48b04fdcdba72784294780fa4d"
NORMAL, HIGH = 5, 100


def edit_scope(settings):
    pattern = rb"(?m)^Sensi_0=5(?=\r?$)"
    changed, count = re.subn(pattern, b"Sensi_0=100", settings)
    restored, count_back = re.subn(rb"(?m)^Sensi_0=100(?=\r?$)", b"Sensi_0=5", changed)
    if count != 1 or count_back != 1 or restored != settings:
        raise ValueError("Expected one exact Sensi_0 entry")
    return changed


def read_checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Unexpected frozen output {path}")
    image, _, receipt = read(path, False)
    if receipt["icc_sha256"] != RT_PROFILE:
        raise ValueError(f"Unexpected ICC {path}")
    return image


def gray_control(cli, out, profile, env, gray_dir, files, core, outside, weight):
    before = read_checked(gray_dir / "gray128-off.png", files["off"])
    normal = read_checked(gray_dir / "gray128-on.png", files["on"])
    path = out / "gray-high-on.png"
    render(cli, gray_dir / "gray128-off.png", profile, path, env)
    high, _, receipt = read(path, False)
    if receipt["icc_sha256"] != RT_PROFILE or high.shape != before.shape:
        raise ValueError("Invalid gray control export")
    zone = ~outside
    normalized, strengths, common = {}, {}, weight >= .8
    core_before = before[core]
    for frame in (core_before, normal[core], high[core]):
        common &= support(frame)
    for name, frame in (("normal", normal), ("high", high)):
        difference = frame.astype(np.int16) - before
        direction = np.median(difference[core], axis=0)
        energy = direction @ direction
        if energy <= 0 or not common.any():
            return dict(passed=False, file_sha256=digest(path), reason="gray response not measurable")
        normalized[name] = np.clip(difference[zone] @ direction / energy, 0, 1)
        shift = np.median((lab(frame[core][common]) - lab(core_before[common])) / weight[common, None], axis=0)
        strengths[name] = float(AXIS @ shift[1:])
    p95 = float(np.quantile(np.abs(normalized["normal"] - normalized["high"]), .95))
    coverage = float(common.mean())
    outside_identity = bool(np.array_equal(high[outside], before[outside]))
    endpoints = added_endpoints(before, high)
    return dict(file_sha256=digest(path), common_support=coverage, outside_identity=outside_identity,
                geometry_weight_difference_p95=p95, strength=strengths, added_endpoint_fraction=endpoints,
                passed=coverage >= .8 and outside_identity and p95 <= .10 and endpoints <= .02
                and all(gain > 0 for gain in strengths.values()))


def prepare_reference(cli, out, item, archive, prior, gray):
    out.mkdir()
    profiles = item["profiles"]
    raw_on = check(item["directory"] / "on.pp3", profiles["on"])
    raw_off = check(item["directory"] / "off.pp3", profiles["off"])
    if disabled(raw_on) != raw_off:
        raise ValueError("Scope inputs differ beyond the published enable flag")
    edited_on, edited_off = edit_scope(raw_on), edit_scope(raw_off)
    if disabled(edited_on) != edited_off:
        raise ValueError("New controls differ beyond the enable flag")
    on_profile, off_profile = out / "on.pp3", out / "off.pp3"
    on_profile.write_bytes(edited_on)
    off_profile.write_bytes(edited_off)
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "cache"), OMP_NUM_THREADS="1")
    core, outside = item["areas"]["wall_left"], item["areas"]["outside"]
    weight = item["weight"][core]
    valid = weight >= .8
    scores, cores = {}, {}
    for arm in ARMS:
        records = prior["arms"][arm]["files"]
        path = archive / arm / "input.png"
        original = read_checked(path, records["input"])
        off = read_checked(archive / arm / "off.png", records["off"])
        old = read_checked(archive / arm / "on.png", records["on"])
        new_path = out / f"{arm}-on.png"
        render(cli, path, on_profile, new_path, env)
        new, _, receipt = read(new_path, False)
        if new.shape != off.shape or receipt["icc_sha256"] != RT_PROFILE:
            raise ValueError(f"Unexpected shape or ICC for {arm}")
        verified_off_hash = None
        checks = dict(archived_off_identity=bool(np.array_equal(original, off)),
                      retained_core=bool(np.array_equal(original[core], item["image"][core])),
                      outside_identity=bool(np.array_equal(new[outside], off[outside])))
        if arm == "original":
            off_path = out / "original-off.png"
            render(cli, path, off_profile, off_path, env)
            verified_off_hash = digest(off_path)
            checked_off, _, evidence = read(off_path, False)
            checks["tested_off_identity"] = bool(np.array_equal(checked_off, off) and evidence["icc_sha256"] == RT_PROFILE)
        endpoints = added_endpoints(off, new)
        checks["endpoints"] = endpoints <= .02
        difference = np.abs(new.astype(np.int16) - old)
        cores[arm] = off[core], old[core], new[core]
        scores[arm] = dict(checks=checks, added_endpoint_fraction=endpoints, file_sha256=digest(new_path),
                           own_off_sha256=verified_off_hash, core_mae_from_normal=float(difference[core].mean()),
                           core_valid_by_frame={name: float((support(frame) & (weight >= .8)).mean())
                                                for name, frame in zip(("off", "normal", "high"), cores[arm], strict=True)},
                           changed_pixels_from_normal=int(np.count_nonzero(difference.any(axis=-1))))
        for frame in cores[arm]:
            valid &= support(frame)
    coverage = float(valid.mean())
    passed = coverage >= .8 and gray["passed"] and all(all(record["checks"].values()) for record in scores.values())
    evidence = dict(passed=passed, support=coverage, excluded=1. - coverage, arms=scores,
                    gray_control=gray, profiles=dict(on=digest(on_profile), off=digest(off_profile)))
    if not passed:
        return evidence
    for arm, (off, old, new) in cores.items():
        original = lab(off[valid])
        for state, pixels in (("normal", old), ("high", new)):
            values = np.median((lab(pixels[valid]) - original) / weight[valid, None], axis=0)
            scores[arm][state] = dict(shift=values.tolist(), warmth=float(AXIS @ values[1:]))
    differences = {
        state: scores["near_flat"][state]["warmth"] - scores["far_flat"][state]["warmth"]
        for state in ("normal", "high")
    }
    before, after = abs(differences["normal"]), abs(differences["high"])
    maintained = all(scores[arm]["high"]["warmth"] >= scores[arm]["normal"]["warmth"] - .60
                     for arm in ("far_flat", "near_flat"))
    evidence["neighbor_effect"] = dict(
        differences=differences, reduction=1. - after / before if before else None,
        maintained_absolute_response=maintained,
        supported=(before >= 1. and after <= .25 * before and after < .60 and maintained),
    )
    return evidence


def make_replays(out, archived, source_offsets, targets, fixed_pairs):
    predictions = {}
    for key, sources in fixed_pairs.items():
        scene, family = key.split("/")
        target = targets[family]
        location = out / key
        location.mkdir(parents=True)
        outputs = predictions[key] = {}
        for role, entry in sources.items():
            old = archived / key / f"paired-{role}.png"
            if digest(old) != entry["sha256"]:
                raise ValueError("Frozen graded image changed")
            offset = source_offsets[f"{scene}-{role}"]
            original = target["original"]
            grade = finish_spatial(original, replay(original, target["weight"], offset, "lab_d50"), STRENGTH)
            outputs[role] = dict(fixed_normal_offset=entry["offset"], applied_offset=offset,
                                 reference_state=f"{scene}-{role}", sha256=save(location / f"compensated-{role}.png", grade))
    return predictions


def run(base, paired, stress_dir, surround, out, replay_only=False):
    base, out = base.absolute(), out.absolute()
    cli = base / RT_RUNTIME / "rawtherapee-cli.exe"
    if not all(path.as_posix().isascii() for path in (cli, out)):
        raise ValueError("Map the runtime and output directory to ASCII drive paths")
    check(cli, RT_CLI)
    raw = Path(__file__).with_name("warmth-color-scope-inputs.json").read_bytes().replace(b"\r\n", b"\n")
    if sha256(raw).hexdigest() != INPUT_PACKET:
        raise ValueError("Frozen prediction inputs changed")
    packet = json.loads(raw)
    source_root = base / "fau-shared-warmth-v1"
    source_ledger = checked_json(source_root / "results.json", FAU_EVIDENCE)
    donors = source_frames(source_root, packet["source_gray"], source_ledger)
    targets = load_targets(base, paired, packet["target_gray"], read_manifest()["families"])
    out.mkdir()
    references, grays = {}, {}
    for scene, item in donors.items():
        areas = item["areas"]
        core, outside = areas["wall_left"], areas["outside"]
        gray_dir = stress_dir / scene / "calibration"
        weight = read_weight(gray_dir, item["receipt"], core, outside)
        profile = out / f"gray-{scene}.pp3"
        raw_profile = check(item["directory"] / "on.pp3", item["spec"]["profiles"]["on"])
        profile.write_bytes(edit_scope(raw_profile))
        env = dict(os.environ, RT_SETTINGS=str(out / "gray-settings"), RT_CACHE=str(out / "gray-cache"), OMP_NUM_THREADS="1")
        gray_out = out / f"gray-{scene}"
        gray_out.mkdir()
        grays[scene] = gray_control(cli, gray_out, profile, env, gray_dir, item["receipt"]["samples"]["gray128"]["files"],
                                    core, outside, weight[core])
        for role, picture in item["samples"].items():
            name = f"{scene}-{role}"
            row = packet["reference_context"][name]
            if not row["passed"]:
                raise ValueError(f"Old context comparison is unqualified: {name}")
            source = dict(image=picture, weight=weight, areas=areas, directory=item["directory"],
                          profiles=item["spec"]["profiles"])
            data = prepare_reference(cli, out / f"scope-{name}", source, surround / name, row, grays[scene])
            references[name] = data
            print(name, data.get("neighbor_effect", "failed scope intake"), flush=True)
    offsets = {
        state: {name: [0., *item["arms"]["original"][state]["shift"][1:]]
                for name, item in references.items() if item["passed"]}
        for state in ("normal", "high")
    }
    fixed_pairs = packet["known_offsets"]
    ready = all(len(values) == 4 for values in offsets.values())
    matched = out / "same-support-control"
    predictions = make_replays(out, stress_dir, offsets["high"], targets, fixed_pairs) if ready else {}
    controls = make_replays(matched, stress_dir, offsets["normal"], targets, fixed_pairs) if ready else {}
    receipt = out / "prediction-receipt.json"
    json_to(receipt, dict(high=predictions, normal_same_support=controls))
    scored, matched_scores = {}, {}
    ready = ready and all(len(group) == 4 and all(len(rows) == 2 for rows in group.values())
                          for group in (predictions, controls))
    if ready and not replay_only:
        stress = checked_json(stress_dir / "results.json", STRESS_EVIDENCE)
        scored = score_cases(base, out, stress_dir, predictions, stress["cases"], targets)
        matched_scores = score_cases(base, matched, stress_dir, controls, stress["cases"], targets)
    cups = [references[name] for name in ("bmug-base", "bmug-captured")]
    cup_supported = all(data["neighbor_effect"]["supported"] for data in cups) if all(data["passed"] for data in cups) else None
    json_to(out / "results.json", dict(manifest=MANIFEST_SHA, context=SURROUND_EVIDENCE, stress=STRESS_EVIDENCE,
                                      inputs=INPUT_PACKET, scope_values=[NORMAL, HIGH], strength=STRENGTH,
                                      references=references, receipt=digest(receipt), ready=ready,
                                      cases=scored, normal_same_support=matched_scores,
                                      cup_scope_hypothesis=cup_supported,
                                      complete_replay=bool(scored) and all(row["compensated"]["complete"]
                                                                         and not row["lost_previously_passing_areas"]
                                                                         for row in scored.values())))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "paired", "stress_dir", "surround", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--replay-only", action="store_true")
    run(**vars(parser.parse_args()))
