"""Compare chroma from native and mirrored target poses under the same reference."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from background_intervention import target_regions
from probe_bg_real_editor_look import BASE, PARENT, PARENT_SHA, STATES, STRENGTH, evaluate, mask_for
from probe_facd_endpoint_support import interior
from probe_facd_two_image_baseline import CHECKPOINT, STYLES, case_metrics, read_checked
from probe_hdm_hdr_display import digest, json_to
from probe_sparks_shared_chroma import LUMA, chroma, render
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial

ROOT = Path(__file__).resolve().parents[2]
FACD_SHA = "f5c752354e97e31abde828b402e56b404a2cdb1ae7da58208fb4de8e4258a7c9"
BG_SHA = "11b72e0865bad7f3d20a4b3b9a8120c5e8df0de8fd2bbc78549625d67fa1d94d"
FACD_SOURCE_SHA = "bdbb3a3f0d331b69bf2334df9472d59c77a5fe6cafa1551ea16df45e6f53e2f2"
LUMA_INTEGER = np.rint(10000 * LUMA).astype(np.int32)


def checked_json(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Changed frozen dataset: {path}")
    return json.loads(path.read_bytes())


def checked_rgb(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Changed frozen input: {path}")
    return decode_image(path)


def mirror_grade(engine, target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), {"identity": True}
    flipped = np.ascontiguousarray(target[:, ::-1])
    opposite = finish_spatial(flipped, engine.prepare(flipped, reference).apply(flipped), STRENGTH)[:, ::-1]
    baseline_y, opposite_y = baseline @ LUMA, opposite @ LUMA
    a, b = chroma(baseline, baseline_y), chroma(opposite, opposite_y)
    output, compressed = render((a + b) / 2, baseline_y)
    error = np.abs((output.astype(np.int32) - baseline) @ LUMA_INTEGER).max() / 10000
    return output, {"native_mirror_chroma_mae": float(np.abs(a - b).mean()),
                    "max_luma_error": float(error), "gamut_fraction": compressed,
                    "output_change_mae": float(np.abs(output.astype(np.int16) - baseline).mean())}


def save(path, pixels):
    Image.fromarray(pixels).save(path)
    return digest(path)


def panel(path, rows):
    width, height = 220, 170
    board = Image.new("RGB", (5 * (width + 10) + 8, len(rows) * (height + 26) + 8), "#222730")
    draw = ImageDraw.Draw(board)
    for row, (label, tiles) in enumerate(rows.items()):
        for col, (name, pixels) in enumerate(tiles.items()):
            left, top = 8 + col * (width + 10), 6 + row * (height + 26)
            draw.text((left, top), f"{label} / {name}", fill="white")
            tile = ImageOps.contain(Image.fromarray(pixels), (width, height))
            board.paste(tile, (left + (width - tile.width) // 2, top + 18))
    board.save(path)
    return digest(path)


def background(engine, output, grade=mirror_grade, label="mirror mean"):
    source = BASE / "bg-real-editor-look"
    known = checked_json(source / "results.json", BG_SHA)
    parent = checked_json(PARENT / "results.json", PARENT_SHA)
    entries = {row["pair"]: row for row in parent["cases"]}
    if known["model_sha256"] != engine.checkpoint_sha or known["strength"] != STRENGTH:
        raise ValueError("Baseline engine changed")
    results = {}
    for name, archived in known["cases"].items():
        root = output / name
        root.mkdir(parents=True)
        files = archived["artifacts"]

        def load(key):
            return checked_rgb(source / name / key, files[key])

        original = load("target.png")
        target_grade = {"off": original, "on": load("expected-on.png")}
        pictures, diagnostics = {}, {}
        rendered, baseline = {}, {}
        for variant in ("C1", "F1"):
            rendered[variant], baseline[variant] = {}, {}
            for state in STATES:
                stem = f"{variant}-{state}"
                guide = load(f"{stem}-reference.png")
                current = load(f"{stem}-13.png")
                result, detail = grade(engine, original, guide, current)
                rendered[variant][state], baseline[variant][state] = result, current
                diagnostics[stem] = {**detail, "sha256": save(root / f"{stem}.png", result)}
                pictures[stem] = {"target": original, "reference": guide, "13": current,
                                  label: result, "known edit": target_grade[state]}
        mask = mask_for(entries[name], name.split("-")[0])
        original_scores = evaluate(original, target_grade, baseline, mask)
        if any(value != archived[key] for key, value in original_scores.items()):
            raise ValueError(f"Background baseline score changed: {name}")
        results[name] = {"baseline": original_scores, "candidate": evaluate(original, target_grade, rendered, mask),
                         "outputs": diagnostics, "comparison_sha256": panel(root / "comparison.png", pictures)}
    count = sum(row["candidate"]["passed"] for row in results.values())
    worst = max(row["candidate"]["max_protected_swap"] for row in results.values())
    return {"cases": results, "passing_cases": count, "worst_protected_swap": worst,
            "passed": count >= 3 and worst <= 3}


def facd(engine, output, grade=mirror_grade, label="mirror mean"):
    aligned = BASE / "facd-border-registration-v1"
    baseline_folder = BASE / "facd-two-image-baseline-v1"
    source = checked_json(ROOT / "docs/research/2026-11-18-facd-border-registration-evidence.json", FACD_SOURCE_SHA)
    baseline = checked_json(baseline_folder / "results.json", FACD_SHA)
    if engine.checkpoint_sha != baseline["checkpoint_sha256"]:
        raise ValueError("FACD baseline engine changed")
    results = {}
    for name, refs in baseline["references"].items():
        root = output / name
        root.mkdir(parents=True)
        original = read_checked(aligned, name, "Origin", source)
        filters = {style: read_checked(aligned, name, style, source) for style in STYLES}
        rgb = (original, *filters.values())
        neutral = target_regions(original, np.zeros(original.shape[:2], dtype=bool))["neutral_proxy"]
        supported = neutral & np.logical_and.reduce([interior(image) for image in rgb])
        if supported.sum() < 256 or supported.sum() < .5 * neutral.sum():
            raise ValueError(f"Lost prevalidated SDR coverage: {name}")
        expected = {style: finish_spatial(original, filtered, STRENGTH).astype(np.int16)
                    for style, filtered in filters.items()}
        pictures, detail = {}, {}
        arms = {arm: {style: {} for style in STYLES} for arm in ("baseline", "candidate")}
        for style in STYLES:
            for ref in refs:
                filename = f"{ref}-{style}.png"
                guide = read_checked(aligned, ref, style, source)
                current = checked_rgb(baseline_folder / name / filename, baseline["artifacts"][f"{name}/{filename}"])
                result, debug = grade(engine, original, guide, current)
                arms["baseline"][style][ref] = current.astype(np.int16)
                arms["candidate"][style][ref] = result.astype(np.int16)
                neutral_errors = {key: float(np.abs(images[style][ref][supported] - expected[style][supported]).mean())
                                  for key, images in arms.items()}
                detail[filename] = {**debug, "neutral_error": neutral_errors, "sha256": save(root / filename, result)}
                pictures[f"{ref} {style}"] = {"target": original, "reference": guide, "13": current,
                                              label: result, "author": expected[style].astype(np.uint8)}
        scores = {arm: case_metrics(original, expected, images, refs) for arm, images in arms.items()}
        if scores["baseline"] != baseline["cases"][name]:
            raise ValueError(f"FACD baseline score changed: {name}")
        results[name] = {"candidate": scores["candidate"], "common_neutral_pixels": int(supported.sum()),
                         "outputs": detail, "comparison_sha256": panel(root / "comparison.png", pictures)}
    count = sum(all(row["candidate"]["gates"].values()) for row in results.values())
    worst = max(err["neutral_error"]["candidate"] - err["neutral_error"]["baseline"]
                for row in results.values() for err in row["outputs"].values())
    return {"cases": results, "passing_cases": count, "max_neutral_degradation": worst,
            "passed": count >= 6 and worst <= 2}


def run(out):
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Unexpected 13 checkpoint")
    toy = np.array([[[0, 10, 100], [128, 129, 240]]], dtype=np.uint8)
    same, _ = mirror_grade(engine, toy, toy.copy(), 255 - toy)
    np.testing.assert_array_equal(same, toy)
    out.mkdir()
    bg, authored = background(engine, out / "bg-real"), facd(engine, out / "facd")
    max_y_error = max(row["max_luma_error"] for block in (bg, authored)
                      for case in block["cases"].values() for row in case["outputs"].values())
    json_to(out / "results.json", {"checkpoint_sha256": CHECKPOINT, "strength": STRENGTH,
                                   "background_baseline_sha256": BG_SHA, "facd_baseline_sha256": FACD_SHA,
                                   "background": bg, "facd": authored, "max_luma_error": max_y_error,
                                   "passed": bg["passed"] and authored["passed"] and max_y_error <= .5})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
