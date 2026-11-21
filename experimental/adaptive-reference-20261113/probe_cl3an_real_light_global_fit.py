"""Measure the capacity of one RGB gain on registered real-light pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_image(path):
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"Expected a source PNG: {path}")
        original_size = image.size
        image = image.convert("RGB")
        native_sample = np.asarray(image)[::5, ::5].copy()
        image.thumbnail((1200, 1200), Image.Resampling.BOX)
        return original_size, np.asarray(image, dtype=np.float32) / 255, native_sample


def linearize(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def encode(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def save_image(image, path):
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(path)


def chromaticity(linear):
    return linear / linear.sum(axis=-1, keepdims=True).clip(min=1e-8)


def source_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pointwise_error_bound(image, ambient):
    at_limit = ((image == 0) | (image == 255)).sum(axis=-1) >= 2
    input_rgb = image[at_limit].astype(np.uint32)
    codes = (input_rgb[:, 0] << 16) | (input_rgb[:, 1] << 8) | input_rgb[:, 2]
    order = np.argsort(codes)
    ordered_targets = ambient[at_limit][order]
    _, counts = np.unique(codes, return_counts=True)
    starts = np.cumsum(counts) - counts
    unavoidable = 0.0
    duplicate_pixels = int(counts[counts > 1].sum())
    for start, count in zip(starts[counts > 1], counts[counts > 1]):
        targets = ordered_targets[start:start + count]
        unavoidable += np.abs(targets - np.median(targets, axis=0)).sum()
    return {
        "sample_pixels": at_limit.size,
        "at_limit_pixels": int(at_limit.sum()),
        "duplicate_pixels": duplicate_pixels,
        "duplicate_rgb8_mae_lower_bound": (
            float(unavoidable / (3 * duplicate_pixels)) if duplicate_pixels else None
        ),
        "whole_image_rgb8_mae_lower_bound": float(unavoidable / (3 * at_limit.size)),
    }


def assess(source, output, scene):
    paths = {
        "ambient": source / f"{scene}_GT.png",
        "color": source / f"{scene}_1_IN.png",
        "white": source / f"{scene}_1_SH_IN.png",
    }
    loaded = {key: read_image(path) for key, path in paths.items()}
    if len({size for size, _, _ in loaded.values()}) != 1:
        raise ValueError(f"Size mismatch: scene {scene}")
    rgb = {key: image for key, (_, image, _) in loaded.items()}
    per_image_valid = {
        key: ((image >= 8 / 255) & (image <= 247 / 255)).all(axis=-1)
        for key, image in rgb.items()
    }
    valid = np.logical_and.reduce(list(per_image_valid.values()))
    for key, image in rgb.items():
        save_image(image, output / f"{scene}-{key}-original.png")
    record = {
        "original_size": loaded["ambient"][0],
        "source_sha256": {key: source_hash(path) for key, path in paths.items()},
        "per_image_valid_fraction": {key: float(mask.mean()) for key, mask in per_image_valid.items()},
        "valid_fraction": float(valid.mean()),
        "pointwise_error_bound": {
            key: pointwise_error_bound(loaded[key][2], loaded["ambient"][2])
            for key in ("color", "white")
        },
        "results": {},
    }
    if valid.mean() < 0.5:
        record["gain_fit_status"] = "insufficient_coverage"
        return record
    record["gain_fit_status"] = "scored"
    linear = {key: linearize(image) for key, image in rgb.items()}
    y, x = np.ogrid[:valid.shape[0], :valid.shape[1]]
    fit = valid & (((y // 32 + x // 32) % 2) == 0)
    score = valid & ~fit
    chroma_change = np.abs(chromaticity(linear["color"]) - chromaticity(linear["ambient"])).mean(axis=-1)
    low, high = np.quantile(chroma_change[score], [0.25, 0.75])
    regions = {
        "whole": score,
        "high_chroma_change": score & (chroma_change >= high),
        "low_chroma_change": score & (chroma_change <= low),
    }
    record["fit_pixels"] = int(fit.sum())
    record["region_pixels"] = {key: int(mask.sum()) for key, mask in regions.items()}
    for key in ("color", "white"):
        sampled = linear[key][fit].astype(np.float64)
        gain = (sampled * linear["ambient"][fit]).sum(axis=0) / (sampled ** 2).sum(axis=0)
        corrected = encode(linear[key] * gain)
        scores = {}
        for region, mask in regions.items():
            scores[region] = {
                "before_rgb8_mae": float(np.abs(rgb[key][mask] - rgb["ambient"][mask]).mean() * 255),
                "after_rgb8_mae": float(np.abs(corrected[mask] - rgb["ambient"][mask]).mean() * 255),
            }
        record["results"][key] = {"gain": gain.tolist(), "regions": scores}

        save_image(corrected, output / f"{scene}-{key}-corrected.png")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    report = {str(scene): assess(args.source, args.output, scene) for scene in (1, 2)}
    (args.output / "measurement.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
