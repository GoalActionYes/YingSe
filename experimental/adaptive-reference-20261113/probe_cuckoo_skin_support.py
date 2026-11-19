"""Replay one real color edit with frozen automatic skin weights instead of hue alone."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from probe_bg_real_legacy_skin import protector
from probe_darktable_cuckoo_mask_coverage import OFF_SHA, REGIONS, chroma_change, tiff
from render_rawtherapee_sky_pair import check


ROOT = Path(__file__).resolve().parents[2]
MEASURE_SHA = "fd0c8242105dada2102d83ea67631876c32f9dea82ce74b88b58b4ba7aeee1ad"
CLOTHING = [587, 414, 615, 431]
PROTECTED = ("light_material", "green_background", "clothing")


def blend(original, full, weight):
    if weight.shape != original.shape[:2] or not np.isfinite(weight).all() or weight.min() < 0 or weight.max() > 1:
        raise ValueError("Invalid skin support")
    pixels = np.rint(original + (full.astype(np.int16) - original) * weight[..., None]).astype(np.uint8)
    np.testing.assert_array_equal(pixels[weight == 0], original[weight == 0])
    return pixels


def run(root, out):
    archive = root / "darktable-cuckoo-legacy-v1"
    current = root / "darktable-cuckoo-mask-coverage-v1"
    evidence = json.loads(check(current / "results.json", MEASURE_SHA))
    source = archive / "off.png"
    check(source, OFF_SHA)
    for name in ("author.tif", "uniform.tif"):
        check(current / name, evidence["hashes"][name])
    with Image.open(source) as image:
        if image.mode != "RGB":
            raise ValueError("Expected RGB8 source")
        original = np.array(image)
    authored, author_mask, _ = tiff(current / "author.tif", True)
    full, _, _ = tiff(current / "uniform.tif", False)
    skin = protector(ROOT / "experimental/skin-protection-20261019/reference_color_app/skin.py")
    weights = {"author": author_mask.astype(np.float32) / 255, "uniform": np.ones(original.shape[:2], np.float32)}
    outputs = {"author": authored, "uniform": full}
    for name, located in (("skin_on_original", original), ("skin_on_uniform", full)):
        small, _, diag = skin._predict(located)
        if not diag["skin_detected"]:
            raise ValueError(f"No eligible skin detection: {name}")
        weight = skin._cv.resize(small, (original.shape[1], original.shape[0]), interpolation=skin._cv.INTER_LINEAR)
        if weight.min() < -1e-6 or weight.max() > 1 + 1e-6:
            raise ValueError("Skin support outside the interpolation tolerance")
        weight = weight.clip(0, 1)
        np.testing.assert_array_equal(blend(original, original, weight), original)
        weights[name] = weight
        outputs[name] = blend(original, full, weight)
    out.mkdir()
    boxes = json.loads(REGIONS.read_bytes()) | {"clothing": CLOTHING}
    results = {}
    for arm, grade in outputs.items():
        path = out / f"{arm}.png"
        mask_path = out / f"{arm}-weights.npy"
        Image.fromarray(grade).save(path)
        weight = weights[arm]
        np.save(mask_path, weight, allow_pickle=False)
        change = chroma_change(grade, original)
        rows = {}
        for name, (x, y, x1, y1) in boxes.items():
            area = np.s_[y:y1, x:x1]
            rows[name] = {"weight_median": float(np.median(weight[area])),
                          "weight_above_half": float((weight[area] > .5).mean()),
                          "chroma_median": float(np.median(change[area]))}
        passes = rows["chest"]["chroma_median"] >= 2 and all(rows[name]["chroma_median"] <= 1.5 for name in PROTECTED)
        results[arm] = {"regions": rows, "passed": passes,
                        "hashes": {p.name: sha256(p.read_bytes()).hexdigest() for p in (path, mask_path)}}
    report = {"parent_sha256": MEASURE_SHA, "regions": boxes, "arms": results,
              "passed": all(results[name]["passed"] for name in ("skin_on_original", "skin_on_uniform"))}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("locator_dependencies", type=Path)
    args = parser.parse_args()
    sys.path.append(str(args.locator_dependencies.resolve(strict=True)))
    run(args.root, args.out)
