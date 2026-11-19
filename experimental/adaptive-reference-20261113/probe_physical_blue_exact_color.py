"""Bound exact-RGB-only mappings on the same positions of two blue treatments."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from probe_physical_blue_global_fit import CAPTURE_SHA, DIGITAL_SHA, SCENES, inside, panel, pixels, save, verified


FIT_SHA = "615a518cb0f8ceb95771f7861725daedeae6ec6ede2b56555f33ec6ace92b383"


def flat(rgb):
    windows = np.lib.stride_tricks.sliding_window_view(rgb, (3, 3), axis=(0, 1))
    span = windows.max(axis=(-2, -1)) - windows.min(axis=(-2, -1))
    return np.pad(np.all(span <= 8, axis=2), 1)


def color_keys(rgb):
    r, g, b = np.moveaxis(rgb.astype(np.int32), -1, 0)
    return (r * 65536 + g * 256 + b).ravel()


def domain(white, actual, digital):
    common = np.logical_and.reduce([inside(rgb) & flat(rgb) for rgb in (white, actual, digital)])
    selected = color_keys(white)[common.ravel()]
    keys, groups, counts = np.unique(selected, return_inverse=True, return_counts=True)
    y, x = np.nonzero(common)
    tile = y // 32 * ((white.shape[1] + 31) // 32) + x // 32
    first = np.full(len(keys), np.iinfo(np.int64).max)
    last = np.full(len(keys), -1)
    np.minimum.at(first, groups, tile)
    np.maximum.at(last, groups, tile)
    usable = (counts >= 4) & (first != last)
    main = np.zeros(common.shape, dtype=bool)
    main[common] = usable[groups]
    return main, int(usable.sum())


def minimum_rgb(keys, desired):
    _, groups, counts = np.unique(keys, return_inverse=True, return_counts=True)
    start = np.r_[0, np.cumsum(counts[:-1])]
    lower_median = start + (counts - 1) // 2
    levels = np.empty((len(counts), 3), dtype=np.uint8)
    for channel, values in enumerate(desired.T):
        sorted_indices = np.lexsort((values, groups))
        levels[:, channel] = values[sorted_indices[lower_median]]
    return levels[groups]


def lower_bound(keys, desired):
    best = minimum_rgb(keys, desired)
    differences = np.abs(best.astype(np.int16) - desired)
    return best, {
        "rgb8_mae": float(differences.mean()),
        "pixel_mean_channel_error_p95": float(np.percentile(differences.mean(axis=1), 95)),
        "nonexact_pixels_fraction": float(np.any(differences > 0, axis=1).mean()),
    }


def run(base, out):
    source = base / "chromamill-capture-cross-v1"
    edits_dir = base / "cti-selective-correction-v1"
    capture = json.loads(verified(source / "results.json", CAPTURE_SHA).read_bytes())
    edits = json.loads(verified(edits_dir / "results.json", DIGITAL_SHA).read_bytes())
    old_path = base / "physical-blue-global-fit-v1/results.json"
    old = json.loads(verified(old_path, FIT_SHA).read_bytes())
    out.mkdir()
    results = {}
    for scene, level in product(SCENES, (213, 241)):
        donor = SCENES[1 - SCENES.index(scene)]
        tag = f"{scene}-{level}"
        rows, board = {}, {}
        for state in ("off", "on"):
            white, blue = (pixels(source / f"{scene}-{light}-{level}-{state}.png",
                                  capture["prepared"][scene][f"{light}-{level}"]["hashes"][state])
                           for light in ("white", "blue"))
            digest = edits["digital"][donor][str(level)]["states"][state]["hashes"]["reference"]
            digital = pixels(edits_dir / f"{donor}-{level}-{state}-reference.png", digest)
            main, groups = domain(white, blue, digital)
            if not groups:
                raise ValueError(f"No eligible repeated colors: {tag}/{state}")
            keys = color_keys(white)
            subset = keys[main.ravel()]
            shaded = white.copy()
            shaded[~main] //= 3
            board[f"{state} / measured area"] = shaded
            record = {"support_fraction": float(main.mean()), "support_pixels": int(main.sum()), "color_groups": groups}
            for arm, image in (("physical", blue), ("digital", digital)):
                best, whole = lower_bound(keys, image.reshape(-1, 3))
                _, flat_bound = lower_bound(subset, image[main])
                best = best.reshape(image.shape)
                filename = out / f"{tag}-{state}-{arm}.png"
                record[arm] = {
                    "whole": whole, "shared_flat_domain": flat_bound, "sha256": save(filename, best),
                    "previous_matrix_whole_mae": old["cases"][tag]["states"][state][arm]["whole"]["rgb8_mae"],
                }
                board[f"{state} / {arm}"] = image
                board[f"{state} / {arm} RGB table"] = best
            gate = (record["support_fraction"] >= .10
                    and record["digital"]["shared_flat_domain"]["rgb8_mae"] == 0
                    and record["physical"]["shared_flat_domain"]["rgb8_mae"] >= 1)
            rows[state] = record | {"passes_necessary_condition": gate}
        results[tag] = {"states": rows, "panel_sha256": panel(out / f"{tag}-comparison.png", board)}
    report = {"parents": {"capture": CAPTURE_SHA, "digital": DIGITAL_SHA, "matrix": FIT_SHA}, "cases": results,
              "all_off_pass": all(row["states"]["off"]["passes_necessary_condition"] for row in results.values())}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
