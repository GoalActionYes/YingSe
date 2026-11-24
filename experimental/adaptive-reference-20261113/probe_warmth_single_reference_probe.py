"""Replay single-reference warmth with one fixed susceptibility measurement."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from probe_fau_single_reference_warmth import FAU_EVIDENCE, LOAD_ORDER, ROLES, SCENES, load_targets
from probe_matched_wall_warmth_workbench import target_areas
from probe_paired_wall_warmth_coordinate import lab, replay
from probe_playraw_flash_room import RT_CLI, read
from probe_playraw_flash_room_workbench import added_endpoints
from probe_shared_authored_tone_admission import support
from probe_single_reference_warmth_plane import estimate, read_weight, region_mask, score_exports
from probe_target_mirror_chroma import STRENGTH, checked_json, digest, json_to, save
from probe_warmth_color_scope import INPUT_PACKET, edit_scope
from probe_warmth_lightness_response import AXIS, STRESS_EVIDENCE
from render_rawtherapee_sky_pair import check, disabled, render
from run_positional_warmth_trial import MANIFEST_SHA, read_manifest
from spatial_render import finish_spatial
from warm_capture_preflight import RT_PROFILE, RT_RUNTIME


ARMS = ("baseline", "uniform", "probe")
PROBE = {
    "labgridALow_0": (1600, 400), "labgridAHigh_0": (1600, 400),
    "labgridBLow_0": (2800, 700), "labgridBHigh_0": (2800, 700),
}


def probe_parameters(raw):
    for name, (old, new) in PROBE.items():
        prefix = name.encode() + b"="
        raw, count = re.subn(rb"(?m)^" + prefix + str(old).encode() + rb"(?=\r?$)", prefix + str(new).encode(), raw)
        if count != 1:
            raise ValueError(f"Expected exactly one frozen probe field: {name}")
    return raw


def setup_reference(base, archive, packet, label, geometry):
    directory = base / "fau" / label
    anonymous, mapping = {}, {}
    for index, role in enumerate(LOAD_ORDER):
        key = f"image-{index}"
        name = ROLES[role]
        light, state = name.split("-")
        photo, _, receipt = read(directory / f"{name}.png", False)
        record = geometry["frames"][light]
        if photo.shape != (260, 452, 3) or receipt != record["files"][state] or receipt["icc_sha256"] != RT_PROFILE:
            raise ValueError(f"Changed reference: {label}/{name}")
        anonymous[key], mapping[role] = photo, key
    photo = anonymous["image-0"]
    areas = target_areas(photo, geometry, False)
    masks = [region_mask(photo.shape, polygon) for polygon in SCENES[label]]
    material = np.logical_or.reduce(masks)
    weight = read_weight(archive / label / "calibration", packet, areas["wall_left"], areas["outside"])
    settings = check(directory / "on.pp3", geometry["profiles"]["on"])
    off = check(directory / "off.pp3", geometry["profiles"]["off"])
    if disabled(settings) != off:
        raise ValueError("Archived reference profiles differ beyond the enable flag")
    simple = probe_parameters(settings)
    return anonymous, mapping, dict(mask=material, areas=areas, weight=weight,
                                    settings=dict(off=disabled(simple), normal=simple, high=edit_scope(simple)))


def read_16bit(path):
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    pixels = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.dtype != np.uint16 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Expected a true three-channel PNG16")
    with Image.open(path) as item:
        if sha256(item.info.get("icc_profile", b"")).hexdigest() != RT_PROFILE:
            raise ValueError("PNG16 color profile changed")
    return pixels[..., ::-1].astype(np.float64) / 257.


def responses(frames, core, valid, weight):
    before = lab(frames["off"][core][valid])
    rows = {}
    for label in ("normal", "high"):
        vector = np.median((lab(frames[label][core][valid]) - before) / weight[valid, None], axis=0)
        rows[label] = dict(shift=vector.tolist(), response=float(AXIS @ vector[1:]))
    normal, high = rows["normal"]["response"], rows["high"]["response"]
    ratio = high / normal if normal > 0 else None
    accepted = min(normal, high) >= .75 and ratio is not None and .75 <= ratio <= 2.50
    return rows, ratio, accepted


def probe_photo(cli, icc, directory, base, geometric, settings):
    directory.mkdir()
    env = dict(os.environ, RT_SETTINGS=str(directory / "settings"), RT_CACHE=str(directory / "cache"),
               OMP_NUM_THREADS="1")
    core, outside = geometric["areas"]["wall_left"], geometric["areas"]["outside"]
    weight = geometric["weight"][core]
    source = directory / "input.png"
    Image.fromarray(base).save(source, icc_profile=icc)
    frames, precise, files, files_16, profile_hashes = {}, {}, {}, {}, {}
    for label, control in settings.items():
        profile = directory / f"{label}.pp3"
        profile.write_bytes(control)
        path = directory / f"{label}.png"
        render(cli, source, profile, path, env)
        frame, _, receipt = read(path, False)
        if receipt["icc_sha256"] != RT_PROFILE or base.shape != frame.shape:
            raise ValueError(f"Unexpected probe export shape or profile: {label}")
        frames[label], files[label], profile_hashes[label] = frame, digest(path), digest(profile)
        path_16 = directory / f"{label}-16.png"
        render(cli, source, profile, path_16, env, depth=16)
        precise[label], files_16[label] = read_16bit(path_16), digest(path_16)
        if precise[label].shape != base.shape:
            raise ValueError("PNG16 probe shape changed")
    checks = dict(off_identity=bool(np.array_equal(frames["off"], base)))
    difference = float(np.abs(precise["off"] - base).max())
    checks["off_16_within_half_code"] = difference <= .50
    valid = weight >= .8
    for frame in (*frames.values(), *precise.values()):
        valid &= support(frame[core])
    coverage = float(valid.mean())
    checks["coverage"] = coverage >= .8
    rows = {}
    for label in ("normal", "high"):
        result = frames[label]
        endpoints = added_endpoints(frames["off"], result)
        outside_identity = bool(np.array_equal(result[outside], base[outside]))
        checks[f"{label}_protected"] = endpoints <= .02 and outside_identity
        rows[label] = dict(added_endpoint_fraction=endpoints, outside_identity=outside_identity)
    receipt = dict(checks=checks, support=coverage, arms=rows, files=files, png16_files=files_16, off_16_max_difference=difference,
                   profile_hashes=profile_hashes, input_sha256=digest(source), gain=None)
    if not all(checks.values()):
        return receipt
    empirical, ratio, accepted = responses(frames, core, valid, weight)
    high_precision, ratio_16, accepted_16 = responses(precise, core, valid, weight)
    for label, row in rows.items():
        row.update(empirical[label])
    receipt["precision_check"] = dict(ratio=ratio_16, arms=high_precision, passes_16=accepted_16,
                                      absolute_ratio_difference=abs(ratio - ratio_16)
                                      if ratio is not None and ratio_16 is not None else None)
    checks["probe_strength_and_gain_range"] = accepted
    checks["precision_keeps_probe_eligible"] = accepted_16
    receipt["apparent_ratio"] = ratio
    if all(checks.values()):
        receipt["gain"] = ratio
    return receipt


def estimate_one(cli, icc, directory, photo, geometry):
    offsets, counts = estimate(photo, geometry["mask"], geometry["weight"])
    precise = offsets.get("plane")
    proof = probe_photo(cli, icc, directory, photo, geometry, geometry["settings"])
    shifts = {}
    if precise is not None:
        shifts["baseline"] = precise["lab_offset"]
        if proof["gain"] is not None:
            shifts["probe"] = (np.asarray(precise["lab_offset"]) * proof["gain"]).tolist()
    return dict(support=counts, measurement=proof, offsets=shifts)


def export_replays(out, estimates, targets, uniform_gain):
    predictions = {}
    for scene, suite in estimates.items():
        for family, target in targets.items():
            key = f"{scene}/{family}"
            out_dir = out / key
            out_dir.mkdir(parents=True)
            exports = predictions[key] = {}
            original = target["original"]
            for item, entry in suite.items():
                offsets = {}
                exports[item] = dict(support=entry["support"], arms=offsets)
                vectors = dict(entry["offsets"])
                if uniform_gain is not None and "baseline" in vectors:
                    vectors["uniform"] = (np.asarray(vectors["baseline"]) * uniform_gain).tolist()
                for arm, vector in vectors.items():
                    moved = replay(original, target["weight"], vector, "lab_d50")
                    frame = finish_spatial(original, moved, STRENGTH)
                    offsets[arm] = dict(lab_offset=vector, sha256=save(out_dir / f"{arm}-{item}.png", frame))
    return predictions


def score_estimates(base, out, predictions, references, targets, previous):
    result = {}
    for key, rows in predictions.items():
        scene, family = key.split("/")
        anonymous, roles, geometry = references[scene]
        target = targets[family]
        args = target["original"], target["weight"], anonymous, geometry["mask"], geometry["weight"]
        saved = args, target["areas"], roles, rows
        score = score_exports(base, out / key, family, target["specification"], saved, arms=ARMS)
        old = previous[key]
        if score["scores"]["baseline"] != old["scores"]["plane"]:
            raise ValueError("Current unadjusted plane does not reproduce the archived scores")
        for role, frame in score["frames"].items():
            if frame["arms"]["baseline"]["sha256"] != old["frames"][role]["arms"]["plane"]["sha256"]:
                raise ValueError("Current unadjusted plane does not reproduce the archived PNG")
        corrected = score["scores"]["probe"]["measured"]
        baseline = score["scores"]["baseline"]["measured"]
        dropped = [name for name, item in baseline.items()
                   if item["passed"] and not corrected[name]["passed"]]
        score["lost_previously_passing_areas"] = dropped
        score["off_controls"] = {}
        score["off_crossed_threshold"] = {}
        for name in baseline:
            keys = ("a_off_matches_original_mae", "b_off_matches_original_mae")
            readable = {arm: [row["measured"][name]["errors"][k] for k in keys] for arm, row in score["scores"].items()}
            score["off_controls"][name] = readable
            crossings = [i for i, (x, y) in enumerate(zip(readable["baseline"], readable["probe"], strict=True)) if x <= 3 < y]
            if crossings:
                score["off_crossed_threshold"][name] = crossings
        result[key] = score
        print(key, {arm: (row["complete"], row["measured"]["wall_left"]["projection"])
                    for arm, row in score["scores"].items()}, flush=True)
    return result


def run(base, paired, archive, out, replay_only=False):
    base, out = base.absolute(), out.absolute()
    cli = base / RT_RUNTIME / "rawtherapee-cli.exe"
    if not all(path.as_posix().isascii() for path in (cli, out)):
        raise ValueError("Map the runtime and output directory to ASCII drive paths")
    check(cli, RT_CLI)
    icc = check(cli.parent / "iccprofiles/output/RTv4_sRGB.icc", RT_PROFILE)
    raw = Path(__file__).with_name("warmth-color-scope-inputs.json").read_bytes().replace(b"\r\n", b"\n")
    if sha256(raw).hexdigest() != INPUT_PACKET:
        raise ValueError("Frozen calibration input packet changed")
    data = json.loads(raw)
    source = base / "fau-shared-warmth-v1"
    ledger = checked_json(source / "results.json", FAU_EVIDENCE)["groups"]
    targets = load_targets(base, paired, data["target_gray"], read_manifest()["families"])
    out.mkdir()
    examples = {}
    refs = {}
    for scene in SCENES:
        (out / f"probe-{scene}").mkdir()
        gray = data["source_gray"]["sources"][scene]["calibration"]
        anonymous, roles, geometry = setup_reference(source, archive, gray, scene, ledger[scene])
        refs[scene] = anonymous, roles, geometry
        examples[scene] = {key: estimate_one(cli, icc, out / f"probe-{scene}" / key, frame, geometry)
                           for key, frame in anonymous.items()}
    gains = [item["measurement"]["gain"] for scene in examples.values() for item in scene.values()]
    uniform_gain = float(np.median(gains)) if len(gains) == 8 and all(gain is not None for gain in gains) else None
    predictions = export_replays(out, examples, targets, uniform_gain)
    receipt = out / "prediction-receipt.json"
    json_to(receipt, predictions)
    ready = len(predictions) == 4 and all(len(rows) == 4 and all(set(row["arms"]) == set(ARMS) for row in rows.values())
                                          for rows in predictions.values())
    scores, strong_lighting = {}, None
    if ready and not replay_only:
        archived = checked_json(archive / "results.json", STRESS_EVIDENCE)
        scores = score_estimates(base, out, predictions, refs, targets, archived["cases"])
        strong_lighting = any(row["strong_mixed_light_pressure"] for row in archived["light_diagnostics"].values())
    all_probe_pass = bool(scores) and all(row["scores"]["probe"]["complete"]
                                         and not row["lost_previously_passing_areas"]
                                         and not row["off_crossed_threshold"] for row in scores.values())
    adds_passing_case = any(row["scores"]["probe"]["complete"] and not row["scores"]["uniform"]["complete"]
                            for row in scores.values())
    json_to(out / "results.json", dict(manifest=MANIFEST_SHA, stress=STRESS_EVIDENCE, calibration_inputs=INPUT_PACKET,
                                      probe=PROBE, strength=STRENGTH, examples=examples, ready=ready,
                                      uniform_gain=uniform_gain, strong_lighting_pressure=strong_lighting,
                                      receipt=digest(receipt), cases=scores,
                                      all_probe_pass=all_probe_pass, adds_passing_case=adds_passing_case,
                                      advance_to_localization=bool(all_probe_pass and adds_passing_case and strong_lighting)))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "paired", "archive", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--replay-only", action="store_true")
    run(**vars(parser.parse_args()))
