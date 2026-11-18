"""Measure bright-edge fit with and without independent specular labels."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fetch_mit_specular_samples import ROLES, SCENES
from probe_hdm_hdr_display import LUMA, digest, json_to
from probe_hdm_hdr_equal_luminance import encode, linear_rgb
from probe_reference_highlight_anchor import add_annotation, fit_rays, ray_samples

MANIFEST_SHA = "3246f4fbd7bde030f0f0437878ba6eb96a2b38b98486abe1ea26030fdff5bc87"


def load(source, records):
    photos, metadata = {}, {}
    for role in ROLES:
        row = records[role]
        path = source / row["file"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"Changed input: {path}")
        with Image.open(path) as image:
            expected = "L" if role == "mask" else "RGB"
            if image.mode != expected or image.format != "PNG" or path.read_bytes()[24] != 8:
                raise ValueError(f"Unexpected input encoding: {path}")
            photos[role] = np.asarray(image)
            metadata[role] = {"width_height": list(image.size), "mode": image.mode,
                              "color_metadata": sorted(image.info)}
    if len({tuple(row["width_height"]) for row in metadata.values()}) != 1:
        raise ValueError("Gallery components are not co-sized")
    return photos, metadata


def resize_rgb(linear, width_height):
    channels = [np.asarray(Image.fromarray(values.astype(np.float32)).resize(width_height, Image.Resampling.BOX))
                for values in np.moveaxis(linear, -1, 0)]
    return np.stack(channels, axis=-1).astype(np.float64)


def photo_arms(data, case):
    image = encode(data["original"] / 255.)
    preview, origin, direction, bright, dark = ray_samples(image)
    object_mask = np.asarray(Image.fromarray(data["mask"]).resize(preview.size, Image.Resampling.NEAREST)) > 127
    spec = resize_rgb(data["specular"] * (2. / 255), preview.size)
    spec_y = spec @ LUMA
    input_y = linear_rgb(np.asarray(preview)) @ LUMA
    bright_at, dark_at = tuple(bright.T), tuple(dark.T)
    in_object = object_mask[bright_at] & object_mask[dark_at]
    spec_delta = spec_y[bright_at] - spec_y[dark_at]
    full_delta = input_y[bright_at] - input_y[dark_at]
    checked = in_object & (spec_delta >= .02) & (spec_delta >= .5 * full_delta)
    strong = spec[object_mask & (spec_y >= .02)]
    truth = np.median(strong[:, [0, 2]] / strong.sum(axis=-1, keepdims=True), axis=0) if len(strong) >= 40 else None

    rows, views = {}, []
    for title, selection in (("bright edges", in_object), ("polarization labels", checked)):
        estimate, support = fit_rays(preview.size, origin[selection], direction[selection], bright[selection])
        clouds = bright[selection][support]
        if estimate["usable"] and truth is not None:
            estimate["error_to_measured_rb"] = float(np.linalg.norm(np.subtract(estimate["proposed_rb"], truth)))
        rows[title] = estimate
        views.append(add_annotation(preview, clouds, f"{case} / {title}", estimate))
    auto, assisted = (row.get("error_to_measured_rb") for row in rows.values())
    accepted = assisted is not None and (assisted <= .02 if auto is None else auto - assisted >= .01)
    report = {
        "measured_rb": truth.tolist() if truth is not None else None,
        "strong_specular_pixels": len(strong), "candidate_pairs_in_object": int(in_object.sum()),
        "physically_supported_pairs": int(checked.sum()),
        "physically_supported_fraction": float(checked.sum() / in_object.sum()) if in_object.any() else None,
        "arms": rows, "passed": accepted,
    }
    return report, views


def measure(source, output):
    manifest_path = source / "manifest.json"
    if digest(manifest_path) != MANIFEST_SHA:
        raise ValueError("Gallery manifest differs from frozen inputs")
    manifest = json.loads(manifest_path.read_bytes())
    if tuple(manifest["cases"]) != SCENES:
        raise ValueError("Unexpected preregistered scene set")
    output.mkdir()
    results = {}
    board = Image.new("RGB", (544, 304 * len(SCENES)), "#222730")
    for index, case in enumerate(SCENES):
        data, metadata = load(source, manifest["cases"][case])
        equation_error = data["original"].astype(np.float64) - data["diffuse"] - 2. * data["specular"]
        mask = data["mask"] > 127
        if not mask.any():
            raise ValueError(f"Empty object mask: {case}")
        mae = float(np.abs(equation_error[mask]).mean())
        results[case] = {"source_images": metadata, "relation_mae_255": mae, "mask_pixels": int(mask.sum()),
                         "qualified": mae <= 3, "passed": False}
        if not results[case]["qualified"]:
            continue
        report, views = photo_arms(data, case)
        results[case].update(report)
        for col, tile in enumerate(views):
            board.paste(tile, (272 * col, 304 * index))
    board_file = output / "comparison.png"
    board.save(board_file)
    count = sum(results[case]["passed"] for case in SCENES if case != "paper1")
    json_to(output / "results.json", {"source_sha256": MANIFEST_SHA, "manifest": manifest, "cases": results,
                                   "comparison_sha256": sha256(board_file.read_bytes()).hexdigest(),
                                   "passing_objects": count, "passed": count >= 2})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    measure(args.source, args.output)
