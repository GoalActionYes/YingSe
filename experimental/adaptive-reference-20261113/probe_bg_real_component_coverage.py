"""Bound edit response obtainable by mixing existing two-image outputs."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_bg_real_crossed_grade_training import (
    BASE, EDITS, digest, grade, headroom, mask_for, pair, read, scores, target_regions,
)
from probe_crossed_edit_pair_supervision import fixed_sets
from probe_dast_exposure_response import save_report, verified_png
from spatial_render import SpatialEngine, finish_spatial

PARTS_SHA = "82dedbf1908f071862068e3c9d745f671822bba461b46d53c01b3a20cd5856d5"
PAIRED_SHA = "186f4fc3813aae32985e2de3ab922091764bf3715d2e27a734df2df7014b5140"
CHECKPOINT = "d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9"
LUMA = np.array([.2126, .7152, .0722])
ROLES = ("C1", "F1")


def ledger(folder, expected):
    path = folder / "results.json"
    if digest(path) != expected:
        raise ValueError(f"Frozen result changed: {path}")
    return json.loads(path.read_bytes())["cases"]


def split(images):
    light = (images @ LUMA)[..., None]
    return light, images - light


def quantization_error(light, chroma):
    lower = light.min(axis=0) + chroma.min(axis=0)
    upper = light.max(axis=0) + chroma.max(axis=0)
    return .5 + np.maximum(0., np.maximum(-lower, upper - 255.))


def interval(base, edited, reference):
    error = (quantization_error(*base) + quantization_error(*edited))
    size = float(np.square(reference).sum())
    light_projection = np.sum((edited[0] - base[0]) * reference, axis=(1, 2, 3)) / size
    chroma_projection = np.sum((edited[1] - base[1]) * reference, axis=(1, 2, 3)) / size
    endpoints = light_projection[:, None] + chroma_projection[None, :]
    slack = float((np.abs(reference) * error).sum() / size)
    lower, upper = float(endpoints.min()), float(endpoints.max())
    bound = [lower - slack, upper + slack]
    return {"light_projection": light_projection.tolist(), "chroma_projection": chroma_projection.tolist(),
            "float_interval": [lower, upper], "round_clip_slack": slack,
            "rgb8_outer_interval": bound, "interval_excludes_target": bound[1] < .75 or bound[0] > 1.25}


def run(output):
    torch.set_num_threads(4)
    _, held, registered = fixed_sets()
    parts_folder = BASE / "bg-real-person-part-descriptor-v1"
    paired_folder = BASE / "crossed-edit-pair-supervision-v1"
    part_cases = ledger(parts_folder, PARTS_SHA)
    paired_cases = ledger(paired_folder, PAIRED_SHA)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("13 checkpoint changed")
    output.mkdir()
    cases = {}
    for item in held.values():
        key = item["target_pair"]
        entry = registered[key]
        source = read(entry["input_file"])
        target = headroom(source)
        refs = pair(BASE / "bg-real-pixel-inputs", item["input"])
        generated = {role: {} for role in ROLES}
        branches = {role: {} for role in ROLES}
        dest = output / key
        dest.mkdir()
        hashes = {}
        for role, style in product(ROLES, EDITS):
            reference = grade(refs[role], EDITS[style])
            result = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
            generated[role][style] = result
            name = f"{role}-{style}.png"
            path = dest / name
            Image.fromarray(result).save(path)
            hashes[name] = digest(path)
            arms = [verified_png(parts_folder / key / arm / name, part_cases[key]["output_sha256"][arm][name])
                    for arm in ("global", "parts")]
            for arm in ("per-image", "paired"):
                filename = f"{arm}-{name}"
                arms.append(verified_png(paired_folder / key / filename, paired_cases[key][arm]["hashes"][filename]))
            branches[role][style] = split(np.array([result, *arms], dtype=float))

        truth = {style: grade(target, params, .8) for style, params in EDITS.items()}
        person = mask_for(entry, key.split("-")[0])
        neutral = target_regions(source, person)["neutral_proxy"] & person
        direct = scores(generated, truth, person, neutral)
        bounds = {}
        for role in ROLES:
            baseline = branches[role]["none"]
            bounds[role] = {style: interval(baseline, branches[role][style], truth[style].astype(float) - truth["none"])
                            for style in EDITS if style != "none"}
        excluded = any(item["interval_excludes_target"] for by_role in bounds.values() for item in by_role.values())
        cases[key] = {"current13": direct, "response_bounds": bounds, "excluded": excluded,
                      "current13_output_sha256": hashes}
    remaining = sum(not row["excluded"] for row in cases.values())
    save_report(output / "results.json", {"checkpoint_sha256": CHECKPOINT, "parts_sha256": PARTS_SHA,
                "paired_sha256": PAIRED_SHA, "arm_order": ["current13", "global", "parts", "per-image", "paired"],
                "cases": cases, "not_excluded": remaining, "addresses_three_of_four": remaining >= 3})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
