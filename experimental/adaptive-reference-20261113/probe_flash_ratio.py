"""Compare ambient and flash-normalized chromaticity on held-out image tiles."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


SCENES = {
    "book": "bde1d5b2b01b410c59c317777dac7e4180061d0a",
    "conference_room": "5d2dfc274b2f83eb8e2fe1383e3e93f43b72fedd",
}


def chromaticity(rgb):
    return rgb / rgb.sum(axis=1, keepdims=True)


def held_distances(values, tiles):
    distances = np.empty(len(values))
    for tile in np.unique(tiles):
        held = tiles == tile
        training = values[~held]
        center = training.mean(axis=0)
        centered = training - center
        _, axes = np.linalg.eigh(centered.T @ centered)
        distances[held] = np.linalg.norm((values[held] - center) @ axes[:, :2], axis=1)
    return distances


def summarize(ambient, lit, mask):
    if ambient.ndim != 3 or ambient.shape[2] != 3 or lit.shape != ambient.shape:
        raise ValueError("Expected registered RGB arrays")
    if mask.shape != ambient.shape[:2]:
        raise ValueError("Expected a matching author mask")
    ambient = ambient[::4, ::4]
    lit = lit[::4, ::4]
    mask = mask[::4, ::4]
    delta = lit - ambient
    valid = mask & np.all(
        (ambient >= 0.01) & (ambient <= 0.98)
        & (lit >= 0.01) & (lit <= 0.98) & (delta >= 0.02), axis=2,
    )
    height, width = mask.shape
    rows, columns = np.indices(mask.shape)
    tiles = (4 * rows // height * 4 + 4 * columns // width)[valid]
    tile_counts = np.bincount(tiles, minlength=16)
    coverage = float(valid.mean())
    result = {
        "count": int(valid.sum()), "sampled": valid.size, "coverage": coverage,
        "tiles": tile_counts.tolist(),
        "qualified": bool(coverage >= 0.10 and np.count_nonzero(tile_counts >= 100) >= 8),
    }
    if len(tiles) == 0 or len(tiles) - tile_counts.max() < 3:
        return {**result, "passed": False}

    representations = {
        "ambient": chromaticity(ambient[valid]),
        "ratio": chromaticity(ambient[valid] / delta[valid]),
    }
    for key, values in representations.items():
        distances = held_distances(values, tiles)
        result[key] = dict(zip(("median", "p90"), np.quantile(distances, [0.5, 0.9]).tolist()))
    before, after = result["ambient"], result["ratio"]
    reduction = before["median"] - after["median"]
    result["median_reduction"] = reduction
    result["passed"] = (result["qualified"] and reduction >= 0.25 * before["median"]
                        and reduction >= 0.005 and after["p90"] <= before["p90"] + 0.005)
    return result


def synthetic_control():
    rng = np.random.default_rng(701)
    material = rng.uniform(0.25, 0.75, (192, 192, 3))
    mix = rng.uniform(0, 1, (192, 192, 1))
    ambient = material * (mix * [0.42, 0.27, 0.14] + (1 - mix) * [0.13, 0.27, 0.42])
    flash = material * rng.uniform(0.15, 0.30, (192, 192, 1))
    result = summarize(ambient, ambient + flash, np.ones(ambient.shape[:2], dtype=bool))
    if not result["passed"] or result["ratio"]["p90"] >= 1e-10:
        raise RuntimeError("Separable synthetic control failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = {"synthetic": synthetic_control(), "scenes": {}}
    for name, expected_blob in SCENES.items():
        payload = (args.input / f"{name}.mat").read_bytes()
        blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if blob != expected_blob:
            raise ValueError(f"Unexpected author file for {name}")
        source = args.input / f"{name}.mat"
        arrays = loadmat(source, variable_names=["im_nf", "im_f", "mask"])
        ambient = arrays["im_nf"].astype(np.float64)
        lit = arrays["im_f"].astype(np.float64)
        mask = arrays["mask"].squeeze() == 1
        results["scenes"][name] = {
            "git_blob": blob, "sha256": hashlib.sha256(payload).hexdigest(),
            "shape": list(ambient.shape), **summarize(ambient, lit, mask),
        }
    results["passed"] = all(scene["passed"] for scene in results["scenes"].values())
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
