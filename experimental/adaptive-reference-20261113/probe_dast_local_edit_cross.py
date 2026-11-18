"""Cross a fixed local grade with three real portrait exposures."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_dast_exposure_response import digest, midtone, save_report, statistics, verified_png

STATES = ("nominal", "ue1", "oe1")
INPUT_SHA = "7e5137f4e559daf39e4f4f09a7500637beb897713257c4d7101de20a0d9dbf97"
BASE_SHA = "892a26347772b7649adb73593cc1740b5efac9bdafc51aff548aa5b4d32aec0c"
REC709 = np.array([.2126, .7152, .0722])
WARM = np.array([12., -2., -21.])
WARM -= REC709 @ WARM


def read_ledger(root, filename, sha):
    path = root / filename
    if digest(path) != sha:
        raise ValueError(f"Frozen ledger changed: {path}")
    return json.loads(path.read_bytes())


def apply_edit(rgb, feather, strength):
    room = np.where(WARM > 0, 255. - rgb, rgb) / np.abs(WARM)
    scale = np.minimum(room.min(axis=2), 1.) * feather * strength
    return np.round(rgb + scale[..., None] * WARM).astype(np.uint8)


def write_png(path, rgb):
    Image.fromarray(rgb).save(path)
    return digest(path)


def prepare(source, output):
    import cv2
    from segment import Segmenter, SHA256

    groups = read_ledger(source, "inputs.json", INPUT_SHA)["groups"]
    output.mkdir()
    cv2.setNumThreads(1)
    locator = Segmenter()
    report = {}
    try:
        for row in groups:
            group = str(row["group"])
            dest = output / group
            dest.mkdir()
            report[group] = {}
            for state in STATES:
                rgb = verified_png(source / group / f"{state}.png", row["frames"][state])
                probability, _ = locator.predict(rgb)
                core = cv2.erode((probability[..., 3] >= .70).astype(np.uint8),
                                 np.ones((11, 11), np.uint8), borderType=cv2.BORDER_CONSTANT,
                                 borderValue=0) > 0
                if state == "nominal":
                    historical = verified_png(source / group / "nominal-mask.png", row["target_mask_sha256"])
                    np.testing.assert_array_equal(core, historical > 0)
                feather = cv2.GaussianBlur(core.astype(np.float64), (25, 25), 4,
                                           borderType=cv2.BORDER_CONSTANT).clip(0, 1)
                edited = apply_edit(rgb, feather, 1.)
                np.testing.assert_array_equal(rgb[feather == 0], edited[feather == 0])
                entry = {"source_edit": statistics(rgb, edited, core & midtone(rgb) & midtone(edited)),
                         "edited_sha256": write_png(dest / f"{state}-on.png", edited)}
                if state == "nominal":
                    entry["truth_sha256"] = write_png(dest / "truth.png", apply_edit(rgb, feather, .8))
                    entry["outside_sha256"] = write_png(dest / "outside.png", (feather == 0).astype(np.uint8) * 255)
                report[group][state] = entry
    finally:
        locator.close()
    save_report(output / "edits.json", {"inputs_sha256": INPUT_SHA, "locator_sha256": SHA256,
                                        "opencv": cv2.__version__, "groups": report})


def score_edit(original, truth, off, on, core, outside, source_edit):
    region = core & midtone(original) & midtone(truth) & midtone(off) & midtone(on)
    expected = statistics(original, truth, region)
    actual = statistics(off, on, region)
    difference = on.astype(np.float64) - off
    bleed = float(np.abs(difference[outside]).mean())
    entry = {"source": source_edit, "expected": expected, "actual": actual,
             "outside_rgb_change": bleed, "projection": None, "increment_rgb_mae": None, "passes": False}
    if expected["qualified"] and expected["chroma_norm"] >= .03:
        axis = np.array(expected["mean_log_chroma_change"])
        projection = float(axis @ actual["mean_log_chroma_change"] / (axis @ axis))
        error = float(np.abs(difference[region] - (truth[region].astype(np.float64) - original[region])).mean())
        entry.update(projection=projection, increment_rgb_mae=error,
                     passes=source_edit.get("chroma_norm", 0.) >= .03 and .75 <= projection <= 1.25
                     and error <= 3. and bleed <= 1.5)
    return entry


def render(source, edits, baseline, output):
    import torch
    from spatial_render import SpatialEngine, finish_spatial

    torch.set_num_threads(4)
    groups = read_ledger(source, "inputs.json", INPUT_SHA)["groups"]
    prior = read_ledger(baseline, "results.json", BASE_SHA)
    edit_path = edits / "edits.json"
    prepared = json.loads(edit_path.read_bytes())
    if prior["inputs_sha256"] != INPUT_SHA or prepared["inputs_sha256"] != INPUT_SHA:
        raise ValueError("Cohort differs from baseline")
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != prior["weight_sha256"]:
        raise ValueError("Checkpoint differs from baseline")
    output.mkdir()
    results = []
    for row in groups:
        donor = str(row["group"])
        target = str(row["target_group"])
        target_row = groups[int(target) - 1]
        target_edit = prepared["groups"][target]["nominal"]
        original = verified_png(source / target / "nominal.png", target_row["frames"]["nominal"])
        core = verified_png(source / target / "nominal-mask.png", target_row["target_mask_sha256"]) > 0
        truth = verified_png(edits / target / "truth.png", target_edit["truth_sha256"])
        outside = verified_png(edits / target / "outside.png", target_edit["outside_sha256"]) > 0
        dest = output / donor
        dest.mkdir()
        generated, scores, hashes = {}, {}, {}
        for state in STATES:
            source_edit = prepared["groups"][donor][state]
            reference_off = verified_png(source / donor / f"{state}.png", row["frames"][state])
            reference_on = verified_png(edits / donor / f"{state}-on.png", source_edit["edited_sha256"])
            for arm, rgb in (("off", reference_off), ("on", reference_on)):
                result = finish_spatial(original, engine.prepare(original, rgb).apply(original), .8)
                generated[state, arm] = result
                hashes[f"{state}-{arm}"] = write_png(dest / f"{state}-{arm}.png", result)
            if hashes[f"{state}-off"] != prior["groups"][int(donor) - 1]["output_sha256"][state]:
                raise ValueError(f"Off render differs from baseline for {donor}/{state}")
            scores[state] = score_edit(original, truth, generated[state, "off"], generated[state, "on"],
                                       core, outside, source_edit["source_edit"])
        exposure = {}
        for state in STATES[1:]:
            for arm in ("off", "on"):
                a, b = generated["nominal", arm], generated[state, arm]
                exposure[f"{state}-{arm}"] = statistics(a, b, core & midtone(original) & midtone(a) & midtone(b))
        passes = (all(row["passes"] for row in scores.values())
                  and all(row.get("chroma_norm", np.inf) < .03 for row in exposure.values()))
        results.append({"group": int(donor), "edit": scores, "exposure": exposure,
                        "output_sha256": hashes, "passes": passes})
    count = sum(row["passes"] for row in results)
    save_report(output / "results.json", {"inputs_sha256": INPUT_SHA, "baseline_sha256": BASE_SHA,
                "edits_sha256": digest(edit_path), "weight_sha256": engine.checkpoint_sha,
                "groups_passing": count, "supports_joint_preservation": count >= 3, "groups": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage, fields in (("prepare", ("source", "output")),
                          ("render", ("source", "edits", "baseline", "output"))):
        command = commands.add_parser(stage)
        for field in fields:
            command.add_argument(field, type=Path)
    args = parser.parse_args()
    stage = vars(args).pop("stage")
    (prepare if stage == "prepare" else render)(**vars(args))
