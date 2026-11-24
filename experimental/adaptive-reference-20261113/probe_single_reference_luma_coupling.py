"""Test whether local lightness explains false warmth in a single reference."""
from argparse import ArgumentParser
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from probe_fau_single_reference_warmth import FAU_EVIDENCE, SCENES, load_targets
from probe_paired_wall_warmth_coordinate import lab, replay
from probe_playraw_flash_room import read
from probe_shared_authored_tone_admission import support
from probe_single_reference_warmth_plane import PAIRED_EVIDENCE, estimate, fit_ab, prepare, score_exports
from probe_target_mirror_chroma import STRENGTH, checked_json, digest, json_to, save
from probe_warmth_lightness_response import STRESS_EVIDENCE
from probe_warmth_single_reference_probe import INPUT_PACKET, setup_reference
from run_positional_warmth_trial import MANIFEST_SHA, read_manifest
from spatial_render import finish_spatial


ARMS = ("plane", "luma")
ORDER = ("base", "captured", "edited", "captured_edited")
MIN_LIGHTNESS_RMS = 1.0
MIN_TEMPLATE_RMS = .10
PRESSURE_GATE = .50
MATCHED_EVIDENCE = "4ec70fc1ff45d0d6b6dd0d6ea1baaf68a35faa67f5367016271ac41c8085a85d"
REFERENCE_GRAY_FILES = {
    "mit": {"off": "2d730fc759cb9f88574d26025dc5d5234d6a132ce0801d92ad77099fa7775714",
            "on": "0750bb91896da3f862551ba292655bd4ad131cd0ca069f5a607133d702e24f31"},
    "beyond_rgb": {"off": "e17bbb435f32f47492e1beb15d178b3f07a5a6f0cd079ae395824f4714995088",
                   "on": "4c5c255fed231efe9e6f12b3d4217544158baf1d7c27ac29e41da6bc952f4a17"},
}


def brightness_fit(x, y, weight, values):
    """Estimate a*b* along the registered template, allowing local lightness to explain color."""
    plane = np.column_stack((np.ones_like(weight), x, y))
    brightness = values[:, 0]
    trend, _, _, _ = np.linalg.lstsq(plane, brightness, rcond=None)
    light = brightness - plane @ trend
    light_rms = float(np.sqrt(np.mean(light ** 2)))
    details = dict(lightness_residual_rms=light_rms)
    if light_rms < MIN_LIGHTNESS_RMS:
        return {**details, "unavailable": "insufficient independent lightness variation"}
    nuisance = np.column_stack((plane, light / light_rms))
    projection, _, rank, _ = np.linalg.lstsq(nuisance, weight, rcond=None)
    weight_rms = float(np.sqrt(np.mean((weight - nuisance @ projection) ** 2)))
    details.update(template_residual_rms=weight_rms, nuisance_rank=int(rank))
    if rank != 4 or weight_rms < MIN_TEMPLATE_RMS:
        return {**details, "unavailable": "lightness and edit template do not separate"}
    design = np.column_stack((nuisance, weight))
    params, _, _, _ = np.linalg.lstsq(design, values[:, 1:], rcond=None)
    return dict(**details, lab_offset=[0., *params[-1].tolist()],
                ab_per_lightness=(params[-2] / light_rms).tolist())


def samples(mask, weight, *photos):
    valid = mask[::4, ::4].copy()
    for photo in photos:
        valid &= support(photo[::4, ::4])
    row, col = np.nonzero(valid)
    row, col = row * 4, col * 4
    height, width = mask.shape
    return col / width, row / height, weight[row, col], [lab(photo[row, col]) for photo in photos]


def fit_reference(photo, mask, weight):
    fits, counts = estimate(photo, mask, weight)
    if "plane" not in fits:
        return dict(support=counts, offsets={}, luma=dict(unavailable="original sample support"))
    x, y, template, values = samples(mask, weight, photo)
    luma = brightness_fit(x, y, template, values[0])
    offsets = {"plane": fits["plane"]["lab_offset"]}
    if "lab_offset" in luma:
        offsets["luma"] = luma["lab_offset"]
    return dict(support=counts, offsets=offsets, luma=luma)


def physical_pressure(source):
    anonymous, roles, mask, weight = source
    x, y, w, values = samples(mask, weight, *(anonymous[roles[role]] for role in ORDER))
    if not len(w):
        return dict(support=dict(points=0), qualifies=False, original_projection={})
    base, capture, edited, capture_edited = values
    pairs = (capture - base, edited - base, capture_edited - capture)
    names = ("physical", "edit_a", "edit_b")
    checked = [fit_ab(x, y, w, array[:, 1:]) for array in pairs]
    old = {name: result["plane"]["lab_offset"][1:] for name, (result, _) in zip(names, checked, strict=True)
           if "plane" in result}
    projection = projections(old)
    return dict(support=checked[0][1], original_vectors=old, original_projection=projection,
                qualifies=any(value is not None and abs(value) >= PRESSURE_GATE for value in projection.values()))


def signed_projection(vector, axis):
    energy = axis @ axis
    return float(np.asarray(vector) @ axis / energy) if energy else None


def projections(vectors):
    if "physical" not in vectors:
        return {}
    return {name: signed_projection(vectors["physical"], np.asarray(vectors[name]))
            for name in ("edit_a", "edit_b") if name in vectors}


def attribution(source, qualification):
    anonymous, roles, mask, weight = source
    x, y, w, values = samples(mask, weight, *(anonymous[roles[role]] for role in ORDER))
    fitted = [brightness_fit(x, y, w, pixels) for pixels in values]
    if any("lab_offset" not in row for row in fitted):
        return dict(eligible=False, fits=fitted)
    base, capture, edited, capture_edited = [np.array(row["lab_offset"][1:]) for row in fitted]
    axes = {key: np.asarray(qualification["original_vectors"][key]) for key in ("edit_a", "edit_b")}
    edit_delta = {"edit_a": edited - base, "edit_b": capture_edited - capture}
    leakage = {key: signed_projection(capture - base, axis) for key, axis in axes.items()}
    retained = {key: signed_projection(edit_delta[key], axis) for key, axis in axes.items()}
    baseline = qualification["original_projection"]
    supports = all(leakage[key] is not None and abs(leakage[key]) < abs(baseline[key])
                   and retained[key] is not None and .75 <= retained[key] <= 1.25 for key in axes)
    return dict(eligible=True, fits=fitted, lamp_against_original_edit=leakage,
                edit_against_original_edit=retained, supports_separation=supports)


def baseline_match(score, old):
    if score["scores"]["plane"] != old["scores"]["plane"]:
        raise ValueError("Original plane scores changed from archive")
    for role, frame in score["frames"].items():
        if frame["arms"]["plane"]["sha256"] != old["frames"][role]["arms"]["plane"]["sha256"]:
            raise ValueError("Original plane output changed from archive")


def paired_guards(score):
    baseline = score["scores"]["plane"].get("measured")
    candidate = score["scores"]["luma"].get("measured")
    if baseline is None or candidate is None:
        return dict(lost_areas=None, off_crossed=None, guard_unavailable=True)
    off_keys = ("a_off_matches_original_mae", "b_off_matches_original_mae")
    lost = [name for name, row in baseline.items() if row["passed"] and not candidate[name]["passed"]]
    crossings = {name: {key: [value, candidate[name]["errors"][key]] for key, value in row["errors"].items()
                        if value <= 3 < candidate[name]["errors"][key]}
                 for name, row in baseline.items()}
    color = {name: values for name, values in crossings.items() if values}
    off = {name: {key: pair for key, pair in values.items() if key in off_keys}
           for name, values in color.items()}
    bad_projections = [index for index, (a, b) in enumerate(zip(baseline["wall_left"]["projection"],
                                                               candidate["wall_left"]["projection"], strict=True))
                       if .75 <= a <= 1.25 and not .75 <= b <= 1.25]
    bad_endpoints = [role for role, item in score["frames"].items()
                     if item["arms"]["plane"]["added_endpoint_fraction"] <= .02 <
                     item["arms"]["luma"]["added_endpoint_fraction"]]
    old_off, new_off = (max(row["wall_left"]["errors"][key] for key in off_keys) for row in (baseline, candidate))
    return dict(lost_areas=lost, off_crossed={name: values for name, values in off.items() if values},
                all_rgb_crossings=color, lost_wall_projections=bad_projections, lost_endpoint_controls=bad_endpoints,
                worst_core_off=[old_off, new_off], guard_unavailable=False)


def extra_fau_sources(fau, ledger, sources):
    extra = {}
    for name in SCENES:
        archive = ledger[name]["frames"]
        path = fau / "fau" / name
        _, _, mask, weight = sources[name]
        lamps = sorted(archive)
        frames = {}
        for lamp in lamps:
            entry = archive[lamp]
            if not entry["metrics"]["admitted"]:
                raise ValueError(f"Expected four-state grade controls for {name}/{lamp}")
            for state in ("off", "on"):
                photo, _, receipt = read(path / f"{lamp}-{state}.png", False)
                if photo.shape != (260, 452, 3) or receipt != entry["files"][state]:
                    raise ValueError(f"Changed FAU photograph {name}/{lamp}/{state}")
                frames[lamp, state] = photo
        for first, second in combinations(lamps, 2):
            if sum(a != b for a, b in zip(first.split("_"), second.split("_"), strict=True)) != 1:
                continue
            if {first, second} == {"w_b", "r_b"}:
                continue
            photos = (frames[first, "off"], frames[second, "off"], frames[first, "on"], frames[second, "on"])
            labels = {role: f"image-{index}" for index, role in enumerate(ORDER)}
            extra[f"{name}_{first}-{second}"] = dict(zip(labels.values(), photos, strict=True)), labels, mask, weight
    if len(extra) != 10:
        raise ValueError("Unexpected number of additional single-lamp pairs")
    return extra


def load_inputs(base, paired, fau_archive, extended_fau):
    manifest = read_manifest()
    families, hashes = manifest["families"], manifest["receipts"]
    packet = Path(__file__).with_name("warmth-color-scope-inputs.json").read_bytes().replace(b"\r\n", b"\n")
    if sha256(packet).hexdigest() != INPUT_PACKET:
        raise ValueError("Frozen calibration packet changed")
    calibration = json.loads(packet)
    targets = load_targets(base, paired, calibration["target_gray"], families)
    mit_path = "shared-local-warmth-v2/results.json"
    beyond_path = "beyond-rgb-near-white-v2/warm-verified-first/results.json"
    geometries = {
        "mit": checked_json(base / mit_path, hashes[mit_path])["results"]["mit"]["14n_copyroom1"],
        "beyond_rgb": checked_json(base / beyond_path, hashes[beyond_path]),
    }
    sources = {}
    for name, specification in families.items():
        receipt = dict(calibration["target_gray"]["cases"][name]["calibration"])
        receipt["reference"] = {"samples": {"gray128": {"files": REFERENCE_GRAY_FILES[name], "delta": [12., -3., -13.]}}}
        row, _, roles = prepare(base, paired, name, specification, geometries[name], receipt)
        sources[name] = row[2], roles, row[3], row[4]

    fau = base / "fau-shared-warmth-v1"
    ledger = checked_json(fau / "results.json", FAU_EVIDENCE)["groups"]
    source_weights = calibration["source_gray"]["sources"]
    for name in SCENES:
        references, roles, geometry = setup_reference(fau, fau_archive, source_weights[name]["calibration"],
                                                      name, ledger[name])
        sources[name] = references, roles, geometry["mask"], geometry["weight"]

    if extended_fau:
        sources.update(extra_fau_sources(fau, ledger, sources))
    return targets, sources


def run(base, paired, fau_archive, old_matched, out, qualify_only=False, extended_fau=False):
    targets, sources = load_inputs(base, paired, fau_archive, extended_fau)
    pressure = {name: physical_pressure(source) for name, source in sources.items()}
    qualifying = {name for name, row in pressure.items() if row["qualifies"]}
    out.mkdir()
    json_to(out / "qualification.json", dict(manifest=MANIFEST_SHA, threshold=PRESSURE_GATE, pressure=pressure))
    if qualify_only or not qualifying:
        for name, row in pressure.items():
            print(name, row["original_projection"], row["qualifies"], flush=True)
        return
    retained = set(SCENES) | set(targets) | qualifying
    sources = {name: source for name, source in sources.items() if name in retained}

    estimates = {
        name: {key: fit_reference(photo, mask, weight) for key, photo in anonymous.items()}
        for name, (anonymous, _, mask, weight) in sources.items()
    }
    exported = {}
    for name, inputs in estimates.items():
        for family, target in targets.items():
            key = f"{name}/{family}"
            destination = out / key
            destination.mkdir(parents=True)
            exported[key] = {}
            for tag, reference in inputs.items():
                outputs = dict(support=reference["support"], arms={})
                for arm, offset in reference["offsets"].items():
                    frame = replay(target["original"], target["weight"], np.asarray(offset), "lab_d50")
                    result = finish_spatial(target["original"], frame, STRENGTH)
                    outputs["arms"][arm] = dict(lab_offset=offset, sha256=save(destination / f"{arm}-{tag}.png", result))
                exported[key][tag] = outputs
    receipt = out / "prediction-receipt.json"
    json_to(receipt, exported)

    checked_json(paired / "results.json", PAIRED_EVIDENCE)
    old_fau = checked_json(fau_archive / "results.json", STRESS_EVIDENCE)
    baseline = checked_json(old_matched / "results.json", MATCHED_EVIDENCE)
    separation = {name: attribution(sources[name], pressure[name]) for name in pressure if name in qualifying}
    cases = {}
    for key, outputs in exported.items():
        name, target_family = key.split("/")
        anonymous, roles, mask, weight = sources[name]
        target = targets[target_family]
        prepared = target["original"], target["weight"], anonymous, mask, weight
        row = score_exports(base, out / key, target_family, target["specification"],
                            (prepared, target["areas"], roles, outputs), arms=ARMS)
        if name in SCENES:
            baseline_match(row, old_fau["cases"][key])
        elif name == target_family:
            baseline_match(row, baseline["cases"][name])
        cases[key] = {**row, **paired_guards(row)}
        print(key, {arm: row["scores"][arm]["complete"] for arm in ARMS}, flush=True)
    added = [key for key, item in cases.items() if item["scores"]["luma"]["complete"] and not item["scores"]["plane"]["complete"]]
    lost = [key for key, item in cases.items() if item["scores"]["plane"]["complete"] and not item["scores"]["luma"]["complete"]]
    failed_guards = ("guard_unavailable", "lost_areas", "all_rgb_crossings", "lost_wall_projections",
                     "lost_endpoint_controls")
    regressions = any(any(item[field] for field in failed_guards) for item in cases.values())
    strong_gain = [key for key in added if key.split("/")[0] in qualifying]
    mechanism_gain = [key for key in strong_gain
                      if separation[key.split("/")[0]].get("supports_separation")
                      and cases[key]["worst_core_off"][1] < cases[key]["worst_core_off"][0]]
    json_to(out / "results.json", dict(manifest=MANIFEST_SHA, calibration=PAIRED_EVIDENCE, faustress=STRESS_EVIDENCE,
                                      old_matched=MATCHED_EVIDENCE, strength=STRENGTH,
                                      pressure=pressure, estimates=estimates, cases=cases, added=added, lost=lost,
                                      strong_gain=strong_gain, attribution=separation, mechanism_gain=mechanism_gain,
                                      supports_fixed_location_replication=bool(mechanism_gain and not lost and not regressions),
                                      receipt=digest(receipt)))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "paired", "fau_archive", "old_matched", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--extended-fau", action="store_true")
    run(**vars(parser.parse_args()))
