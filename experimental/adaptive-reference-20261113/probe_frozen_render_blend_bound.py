"""Bound per-pixel convex mixing of three frozen two-image renderers."""

import argparse
from pathlib import Path

import numpy as np

from probe_arri_film_b_dose import endpoints, verified
from probe_hdm_hdr_display import digest, json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_target_mirror_chroma import FACD_SOURCE_SHA, ROOT, checked_rgb, save
from probe_facd_two_image_baseline import STYLES, read_checked
from spatial_render import finish_spatial
from verify_development_inputs import manifest


FULL = "frozen-candidate-development-crosscheck-v1"
OFF = "frozen-appearance-crosscheck-off-v1"
FULL_SHA = "6f842f52e434f0a3c71d3ac0eb97415955ac774ed529c36fe7d141d725e2d63a"
OFF_SHA = "ea61ea24be2366c9f8c4026576e36a4e0342802aadb6fe12cca982f45d544d00"
Y = np.array([.299, .587, .114])


def neutral(rgb):
    luma = rgb @ Y
    return (np.ptp(rgb, axis=-1) <= 20) & (luma >= 40) & (luma <= 220)


def inner(rgb):
    return (rgb.min(axis=-1) > 0) & (rgb.max(axis=-1) < 255)


def maximum_dot(response, desired):
    return np.einsum("ahwc,hwc->ahw", response, desired)


def projection(dots, desired, where):
    power = float(np.square(desired[where]).sum())
    if power <= 0:
        raise ValueError("Missing known edit signal")
    value = float(dots.max(axis=0)[where].sum() / power)
    round_slack = float(np.abs(desired[where]).sum() / power)
    return {"endpoint": (dots[:, where].sum(axis=1) / power).tolist(),
            "continuous_max": value, "rgb8_upper": value + round_slack}


def distance_bound(values, wanted, allowance):
    residual = np.maximum(values.min(axis=0) - wanted, wanted - values.max(axis=0))
    return float(np.maximum(0, residual - allowance).mean())


def extremes(out, prefix, off, on, desired, *, luma=False, regions=None):
    on, off = np.asarray(on), np.asarray(off)
    response = on.astype(np.int16) - off
    if luma:
        response = (response @ Y)[..., None]
    dots = maximum_dot(response, desired)
    height, width = desired.shape[:2]
    y, x = np.ogrid[:height, :width]
    choices = dots.argmax(axis=0)
    hashes = {state: save(out / f"{prefix}-{state}.png", value[choices, y, x])
              for state, value in (("off", off), ("on", on))}
    rows = {}
    for name, keep in regions.items():
        result = projection(dots, desired, keep)
        if not luma:
            result["rgb8_error_lower"] = distance_bound(response[:, keep], desired[keep], 1)
        rows[name] = result
    return {"regions": rows, "oracle_endpoint_outputs": hashes}


def background(base, out, full, off):
    manifest_bg = manifest(base, "bg-real-editor-look")
    results = {}
    for name, row in manifest_bg["cases"].items():
        assets = row["artifacts"]
        path = base / "bg-real-editor-look" / name
        target, authored = (checked_rgb(path / key, assets[key]) for key in ("target.png", "expected-on.png"))
        desired = ((authored.astype(np.int16) - target) @ Y)[..., None]
        whole = np.ones(target.shape[:2], dtype=bool)
        by_role = {}
        for role in ("C1", "F1"):
            states = []
            for state in ("off", "on"):
                tag = f"{role}-{state}"
                filename = f"{tag}-13.png"
                baseline = checked_rgb(path / filename, assets[filename])
                states.append([baseline, *[
                    checked_rgb(base / directory / "background" / name / f"{tag}.png",
                                suite["cases"][name]["outputs"][tag]["sha256"])
                    for directory, suite in ((FULL, full), (OFF, off))
                ]])
            by_role[role] = extremes(out, f"bg-{name}-{role}", *states, desired, luma=True, regions={"whole": whole})
        results[name] = {"roles": by_role,
                         "excluded": any(x["regions"]["whole"]["rgb8_upper"] < .75 for x in by_role.values())}
    return results


def film(base, out, full, off):
    prepared = manifest(base, "arri-two-looks-v2")
    crossed = manifest(base, "arri-film-b-crossed-v1")
    results = {}
    for key, row in crossed["cases"].items():
        target, styled = endpoints(base / "arri-two-looks-v2", prepared, row["target"])
        desired = finish_spatial(target, styled, .8).astype(np.int16) - target
        areas = {"whole": np.ones(target.shape[:2], dtype=bool), "neutral": neutral(target) & neutral(styled)}
        states = []
        prefix = f"{row['target']}-from-{row['reference']}"
        for state in ("off", "film_b"):
            name = f"{prefix}-{state}.png"
            baseline = verified(base / "arri-film-b-crossed-v1" / name, row["output_sha256"][state])
            states.append([baseline, *[
                verified(base / directory / "film_b" / name, suite["cases"][key]["outputs"][state]["sha256"])
                for directory, suite in ((FULL, full), (OFF, off))
            ]])
        record = extremes(out, prefix, *states, desired, regions=areas)
        record["excluded"] = any(row["rgb8_upper"] < .75 or row["rgb8_error_lower"] > 3
                                 for row in record["regions"].values())
        results[key] = record
    return results


def facd(base, full, off):
    source = verified_json(ROOT / "docs/research/2026-11-18-facd-border-registration-evidence.json", FACD_SOURCE_SHA)
    baseline = manifest(base, "facd-two-image-baseline-v1")
    aligned = base / "facd-border-registration-v1"
    cases = {}
    for group, case in full["cases"].items():
        target = read_checked(aligned, group, "Origin", source)
        styles = {style: read_checked(aligned, group, style, source) for style in STYLES}
        keep = neutral(target) & inner(target) & np.logical_and.reduce([inner(image) for image in styles.values()])
        if keep.sum() != case["common_neutral_pixels"]:
            raise ValueError("FACD neutral area changed")
        expected = {style: finish_spatial(target, image, .8) for style, image in styles.items()}
        values, intervals = {}, {style: [] for style in styles}
        for name, row in case["outputs"].items():
            key = f"{group}/{name}"
            images = [checked_rgb(base / "facd-two-image-baseline-v1" / key, baseline["artifacts"][key])]
            images += [checked_rgb(base / folder / "facd" / key, suite["cases"][group]["outputs"][name]["sha256"])
                       for folder, suite in ((FULL, full), (OFF, off))]
            style = name.split("-")[1].removesuffix(".png")
            stack = np.array(images, dtype=np.int16)
            intervals[style].append((stack.min(axis=0), stack.max(axis=0)))
            lower = distance_bound(stack[:, keep], expected[style][keep], .5)
            whole = distance_bound(stack, expected[style], .5)
            original_error = float(np.abs(expected[style].astype(np.int16) - target).mean())
            values[name] = {"neutral_error_lower": lower,
                            "neutral_minimum_extra_vs_13": lower - row["neutral_error"]["baseline"],
                            "whole_error_lower": whole, "original_whole_error": original_error,
                            "fidelity_excluded": whole > original_error - 1}
        floor = {}
        for style, ((first_low, first_high), (second_low, second_high)) in intervals.items():
            gap = np.maximum(first_low - second_high, second_low - first_high)
            floor[style] = float(np.maximum(gap - 1, 0).mean())
        swap_limit = baseline["cases"][group]["swap_limit"]
        cases[group] = {"outputs": values, "swap_error_lower": floor, "swap_limit": swap_limit,
                        "excluded": any(row["fidelity_excluded"] for row in values.values())
                        or any(v > swap_limit for v in floor.values())}
    return cases


def run(base, out):
    full = verified_json(base / FULL / "results.json", FULL_SHA)["suites"]
    off = verified_json(base / OFF / "results.json", OFF_SHA)["suites"]
    out.mkdir()
    bg = background(base, out, full["background"], off["background"])
    films = film(base, out, full["film_b"], off["film_b"])
    filters = facd(base, full["facd"], off["facd"])
    decision = {"background_possible": sum(not row["excluded"] for row in bg.values()),
                "film_possible": sum(not row["excluded"] for row in films.values()),
                "facd_possible": sum(not row["excluded"] for row in filters.values())}
    json_to(out / "results.json", {"parents": {"full": FULL_SHA, "off": OFF_SHA},
                                   "arm_order": ["13", "expanded", "expanded_without_appearance"],
                                   "background": bg, "film_b": films, "facd": filters, "decision": decision})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "out"):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
