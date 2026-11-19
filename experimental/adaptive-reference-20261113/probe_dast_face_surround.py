"""Compare local contrast and face-only warmth on frozen DAST references."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from probe_dast_exposure_response import midtone, save_report, verified_png
from probe_dast_local_edit_cross import INPUT_SHA, STATES, read_ledger

EDITS_SHA = "5ab0a185c1e4ed6010e196597e5e44875ce9974ba5fa8d7ae742854b07a3aa28"


def dilate(core, radius):
    return np.asarray(core.filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0


def warmth(rgb):
    color = np.maximum(rgb.astype(np.float64), 1) / 255
    linear = np.where(color <= .04045, color / 12.92, ((color + .055) / 1.055) ** 2.4)
    return (np.log(linear[..., 0]) - np.log(linear[..., 2])) / np.sqrt(2)


def region_mean(values, region):
    count = int(region.sum())
    return {"count": count, "mean": float(values[region].mean()) if count >= 1024 else None}


def score(rgb, regions):
    valid = midtone(rgb)
    values = warmth(rgb)
    face, ring = (region_mean(values, region & valid) for region in regions)
    difference = None
    if face["mean"] is not None and ring["mean"] is not None:
        difference = face["mean"] - ring["mean"]
    return {"face": face["mean"], "contrast": difference, "ring": ring["mean"],
            "face_pixels": face["count"], "ring_pixels": ring["count"]}


def check_pair(off, on, regions):
    valid = midtone(off) & midtone(on)
    delta = warmth(on) - warmth(off)
    face, ring = (region_mean(delta, region & valid) for region in regions)
    clean = (face["mean"] is not None and ring["mean"] is not None and face["mean"] > 0
             and abs(ring["mean"]) <= .1 * face["mean"])
    return {"face": face, "ring": ring, "clean": clean}


def summarize(rows, key):
    on = [row["on"][key] for row in rows]
    off = [row["off"][key] for row in rows]
    if None in on or None in off:
        return {"margin": None, "off_span": None}
    return {"margin": min(on) - max(off), "off_span": max(off) - min(off)}


def run(source, edits):
    inputs = read_ledger(source, "inputs.json", INPUT_SHA)
    edited = read_ledger(edits, "edits.json", EDITS_SHA)
    groups = []
    for entry in inputs["groups"]:
        group = str(entry["group"])
        mask = verified_png(source / group / "nominal-mask.png", entry["target_mask_sha256"])
        core = Image.fromarray(mask)
        regions = mask > 0, dilate(core, 40) & ~dilate(core, 24)
        rows = []
        for state in STATES:
            off = verified_png(source / group / f"{state}.png", entry["frames"][state])
            on = verified_png(edits / group / f"{state}-on.png", edited["groups"][group][state]["edited_sha256"])
            rows.append({"state": state, "off": score(off, regions), "on": score(on, regions),
                         "edit_delta": check_pair(off, on, regions)})
        face, contrast = (summarize(rows, key) for key in ("face", "contrast"))
        clean = all(row["edit_delta"]["clean"] for row in rows) and contrast["margin"] is not None
        gain = None if contrast["margin"] is None else contrast["margin"] - face["margin"]
        groups.append({"group": entry["group"], "face": face, "contrast": contrast, "clean": clean,
                       "gain": gain, "rows": rows})
    all_rows = [row for group in groups for row in group["rows"]]
    face, contrast = (summarize(all_rows, key) for key in ("face", "contrast"))
    improved = sum(group["clean"] and group["gain"] >= .03 for group in groups)
    return {"inputs_sha256": INPUT_SHA, "edits_sha256": EDITS_SHA, "groups": groups,
            "joint_face": face, "joint_contrast": contrast, "improved_groups": improved,
            "qualified": all(group["clean"] for group in groups) and contrast["margin"] >= .03 and improved >= 4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "edits", "report"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    save_report(args.report, run(args.source, args.edits))
