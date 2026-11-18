"""Separate two known illuminants using measured skin and frozen RGB controls."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_bg_real_skin_regions import BASE, ROOT, digest
from probe_reference_skin_gray_prior import entries, rgb256
from probe_skin_spectrum_matrix import valid_in_both
from probe_skin_spectrum_white import WORKBOOK, cheek_samples, operators, to_linear, to_srgb


DONORS_SHA = "1b7ebcf36dcb77b077cf5290e8c88a4558068492a8407eaaf74bcbabef933b41"


def color_features(rgb):
    linear = to_linear(rgb)
    valid = np.all(linear >= .002, axis=-1) & np.all(linear <= .98, axis=-1)
    sample = np.log(linear[valid])
    return sample[:, [0, 2]] - sample[:, 1, None], valid


def radius(query, library):
    out = np.empty(len(query))
    for start in range(0, len(query), 128):
        end = min(start + 128, len(query))
        differences = query[start:end, None] - library[None]
        sq = (differences * differences).sum(axis=2)
        out[start:end] = np.sqrt(np.partition(sq, 7, axis=1)[:, 7])
    return out


def classify(rgb, library, thresholds):
    query, valid = color_features(rgb)
    unique, inverse = np.unique(query, axis=0, return_inverse=True)
    radii = np.column_stack([radius(unique, samples)[inverse] for samples in library.values()])
    near = radii.argmin(axis=1)
    small, large = radii.min(axis=1), radii.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.divide(large-small, large, out=np.zeros_like(large), where=large>0)
    qualifies = (margin >= .25) & (small <= np.array(list(thresholds.values()))[near])
    counts = {name: int(np.count_nonzero(qualifies & (near == i))) for i, name in enumerate(library)}
    best = max(counts, key=counts.get)
    return {"masked_pixels": len(rgb), "valid": int(valid.sum()), "counts": counts,
            "label": best if len(query) >= 16 and counts[best] / len(query) >= .8 else None}


def run(output):
    docs = ROOT/"docs/research"
    donor_path = docs/"2026-11-19-skin-spectral-donors-evidence.json"
    if digest(donor_path) != DONORS_SHA:
        raise ValueError("Frozen physical controls changed")
    previous = json.loads(donor_path.read_bytes())
    calib = BASE/"skin-spectra-input"
    white = json.loads((docs/"2026-11-19-skin-spectrum-white-evidence.json").read_bytes())
    for name, sha in white["sources"].items():
        if digest(calib/name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    groups = cheek_samples(calib/WORKBOOK)
    samples = np.concatenate(tuple(groups.values()))
    labels = np.repeat(list(groups), [len(value) for value in groups.values()])
    linear = {key: samples @ model for key, model in operators(calib).items()}
    support = valid_in_both(linear)
    library, thresholds = {}, {}
    for lamp, points in linear.items():
        features, present = color_features(to_srgb(points[support]))
        group = labels[support][present]
        scores = np.empty(len(group))
        for lab in groups:
            selected = group == lab
            scores[selected] = radius(features[selected], features[~selected])
        thresholds[lamp] = float(np.quantile(scores, .95))
        library[lamp] = features
    physics = {}
    for case, row in previous["cases"].items():
        panel = BASE/"skin-spectral-donors"/f"{case}.png"
        if digest(panel) != row["panel_sha256"]:
            raise ValueError("Frozen physical panel changed")
        with Image.open(panel) as picture:
            rgb = np.asarray(picture.convert("RGB"))[35:83]
        cheeks = np.concatenate([rgb[:, x:x+48].reshape(-1, 3) for x in (80, 144)])
        known = case.split("-")[1]
        value = classify(cheeks, library, thresholds)
        physics[case] = {"known": known, **value, "correct": known == value["label"]}
    folder = BASE/"reference-skin-gray-prior"
    mask_index = json.loads((folder/"masks.json").read_bytes())
    ordinary, flips, incomparable = {}, [], []
    for family, by_case in entries().items():
        ordinary[family] = {}
        for case, roles in by_case.items():
            results = {}
            for role, (path, sha) in roles.items():
                name = f"{case}-{role}.npy"
                mask = folder/name
                row = mask_index[name]
                if row["rgb_sha256"] != sha or digest(mask) != row["sha256"]:
                    raise ValueError("Frozen ordinary mask changed")
                results[role] = classify(rgb256(path, sha)[np.load(mask, allow_pickle=False)], library, thresholds)
            ordinary[family][case] = results
            pairs = [("C1", "F1")] if family == "bg" else [("base", "local"), ("base", "whole")]
            for a, b in pairs:
                before, after = results[a]["label"], results[b]["label"]
                if before is None or after is None:
                    incomparable.append([family, case, a, b])
                elif before != after:
                    flips.append([family, case, a, b, before, after])
    empty = sum(value["label"] is not None for case in ordinary["empty"].values() for value in case.values())
    result = {"thresholds": thresholds, "physical": physics, "ordinary": ordinary, "flips": flips,
              "incomparable": incomparable, "empty_triggered": empty,
              "passed": all(v["correct"] for v in physics.values()) and not flips and not empty,
              "source_sha256": digest(Path(__file__)), "masks_sha256": digest(folder/"masks.json"),
              "protocol_sha256": digest(Path(__file__).with_name("spectral-light-choice-protocol.md"))}
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
