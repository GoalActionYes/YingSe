"""Check whether source-only RGB distance warns of the frozen portrait errors."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_skin_spectral_portraits import ARCHIVE_MD5, FEATHER, PATCH, read_patches
from probe_skin_spectrum_matrix import DIRECTIONS, valid_in_both
from probe_skin_spectrum_rgb_bound import MATRIX_SHA, N_CLOSEST
from probe_skin_spectrum_white import WORKBOOK, cheek_samples, checksum, mix, operators, to_linear, to_srgb


PORTRAIT_SHA = "944c45e7f2101b0a4e154871212a17c6719b607d084d39db1cff2769d8886d9f"


def neighbors(query, train):
    ids = np.empty((len(query), N_CLOSEST), dtype=np.intp)
    radius = np.empty(len(query))
    for start in range(0, len(query), 128):
        stop = min(start + 128, len(query))
        distance = ((query[start:stop, None] - train[None]) ** 2).sum(axis=-1)
        nearest = np.argsort(distance, kind="stable", axis=1)[:, :N_CLOSEST]
        ids[start:stop] = nearest
        radius[start:stop] = np.sqrt(np.take_along_axis(distance, nearest[:, -1:], axis=1)[:, 0])
    return ids, radius


def color_overlay(rgb, mask, valid, tint):
    panel = rgb.copy()
    panel[mask] = np.rint(.28 * panel[mask] + .72 * np.asarray(tint)).astype(np.uint8)
    panel[~valid] = 28
    return panel


def preview(path, photos):
    width = 188
    sheet = Image.new("RGB", (104 + width * len(photos), 524), "#222831")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(("Source", "Error > 1.5", "Low support")):
        draw.text((7, 35 + 165 * i), row, fill="white")
    for i, (name, (image, fail, low, valid)) in enumerate(photos.items()):
        x = 106 + i * width
        draw.text((x, 8), name, fill="white")
        for j, pixels in enumerate((image, color_overlay(image, fail, valid, (236, 54, 104)),
                                    color_overlay(image, low, valid, (248, 177, 65)))):
            sheet.paste(Image.fromarray(pixels).resize((154, 154), Image.Resampling.NEAREST), (x, 29 + 165 * j))
    sheet.save(path)


def segment_counts(error, radius, threshold, outside):
    bad = error > 1.5
    low = radius > threshold
    return {"pixels": len(error), "bad": int(bad.sum()), "flagged": int(low.sum()),
            "flagged_bad": int((low & bad).sum()), "kept_good": int((~low & ~bad).sum()),
            "bad_high_support_outside": int((bad & ~low & outside).sum()),
            "outside": int(outside.sum()), "median_error": float(np.median(error)),
            "median_radius": float(np.median(radius))}


def run(spectra, portraits, coords, previous, portrait_evidence, matrix_evidence, white_evidence, out):
    if checksum(portrait_evidence) != PORTRAIT_SHA or checksum(matrix_evidence) != MATRIX_SHA:
        raise ValueError("Frozen result changed")
    base = json.loads(portrait_evidence.read_bytes())
    calibration = json.loads(matrix_evidence.read_bytes())
    if checksum(white_evidence) != calibration["prior_evidence_sha256"] or checksum(coords) != base["coordinates_sha256"]:
        raise ValueError("Frozen reference inputs changed")
    for name, digest in json.loads(white_evidence.read_bytes())["sources"].items():
        if checksum(spectra / name) != digest:
            raise ValueError(f"ISSA source changed: {name}")
    for name in ARCHIVE_MD5:
        if checksum(portraits / name) != base["archive_sha256"][name]:
            raise ValueError(f"Portrait source changed: {name}")
    transforms = operators(spectra)
    origins = cheek_samples(spectra / WORKBOOK)
    measured = np.concatenate(tuple(origins.values()))
    groups = np.repeat(list(origins), [len(a) for a in origins.values()])
    issa = {key: measured @ model for key, model in transforms.items()}
    training_support = valid_in_both(issa)
    if int(training_support.sum()) != 2095:
        raise ValueError("Unexpected training support")
    group = groups[training_support]
    regions = json.loads(coords.read_bytes())
    cheeks = {name: read_patches(portraits, name, rectangles) for name, rectangles in regions.items()}
    out.mkdir()
    results = {}
    edge = PATCH - FEATHER
    core = np.s_[FEATHER:edge, FEATHER:edge]
    for before, after in DIRECTIONS:
        name = f"{before}-{after}"
        x = to_linear(to_srgb(issa[before][training_support]))
        held_radius = np.empty(len(x))
        for lab in np.unique(group):
            held = group == lab
            held_radius[held] = neighbors(x[held], x[~held])[1]
        threshold = float(np.quantile(held_radius, .95))
        matrix = np.asarray(calibration["controls"][name]["matrix"])
        residuals = issa[after][training_support] - x @ matrix
        records, photos = {}, {}
        for person, patches in cheeks.items():
            parent = previous / f"{person}-{name}.png"
            if checksum(parent) != base["cases"][f"{person}-{name}"]["panel_sha256"]:
                raise ValueError("Parent panel changed")
            with Image.open(parent) as panel:
                parent_output = np.asarray(panel.convert("RGB"))[181:245, 540:796]
            for i, patch in enumerate(patches):
                y = {light: patch[core].reshape(-1, 31) @ model for light, model in transforms.items()}
                valid = valid_in_both(y)
                src = to_srgb(y[before][valid])
                target = mix(src, to_srgb(y[after][valid]))
                query = to_linear(src)
                nearby, radius = neighbors(query, x)
                base_linear = query @ matrix
                correction = residuals[nearby]
                result = mix(src, to_srgb(base_linear + correction.mean(axis=1)))
                historical = parent_output[:, (i + 1) * PATCH:(i + 2) * PATCH][core].reshape(-1, 3)
                np.testing.assert_array_equal(result, historical[valid])
                predicted_by_neighbor = mix(src[:, None, :], to_srgb(base_linear[:, None, :] + correction))
                lo = predicted_by_neighbor.min(axis=1).astype(np.int16) - 1
                hi = predicted_by_neighbor.max(axis=1).astype(np.int16) + 1
                outside = np.any((target < lo) | (target > hi), axis=1)
                error = np.abs(result.astype(np.int16) - target).mean(axis=-1)
                label = f"{person}-{'L' if i == 0 else 'R'}"
                records[label] = segment_counts(error, radius, threshold, outside)
                shape = (PATCH - FEATHER * 2,) * 2
                bad_map, low_map = np.zeros_like(valid), np.zeros_like(valid)
                bad_map[valid], low_map[valid] = error > 1.5, radius > threshold
                photos[label] = (to_srgb(y[before]).reshape(*shape, 3), bad_map.reshape(shape),
                                 low_map.reshape(shape), valid.reshape(shape))
        total = {field: sum(v[field] for v in records.values())
                 for field in ("pixels", "bad", "flagged", "flagged_bad", "kept_good", "bad_high_support_outside", "outside")}
        recall = total["flagged_bad"] / total["bad"] if total["bad"] else None
        good = total["pixels"] - total["bad"]
        retain = total["kept_good"] / good if good else None
        graph = out / f"{name}.png"
        preview(graph, photos)
        results[name] = {"threshold": threshold, "total": total, "error_recall": recall, "good_retention": retain,
                         "passed": recall is not None and retain is not None and min(recall, retain) >= .8,
                         "patches": records, "panel_sha256": checksum(graph)}
    summary = {"results": results, "parent_sha256": PORTRAIT_SHA, "matrix_sha256": MATRIX_SHA,
               "source_sha256": checksum(Path(__file__)),
               "protocol_sha256": checksum(Path(__file__).with_name("skin-spectral-support-protocol.md"))}
    (out / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("spectra", "portraits", "coords", "previous", "portrait_evidence", "matrix_evidence", "white_evidence", "out"):
        cli.add_argument(arg, type=Path)
    run(**vars(cli.parse_args()))
