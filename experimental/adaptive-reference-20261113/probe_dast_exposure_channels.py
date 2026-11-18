"""Replay reference histogram channels on matched exposure and intentional Film A."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from histogram_replay import conditions
from probe_arri_edge_coherence import CUBE_SHA, render as apply_film
from probe_bg_real_channel_replay import histogram, infer
from probe_dast_exposure_response import SHA256, STATES, digest, midtone, save_report, statistics, verified_png
from spatial_render import SpatialEngine, finish_spatial

INPUTS_SHA = "7e5137f4e559daf39e4f4f09a7500637beb897713257c4d7101de20a0d9dbf97"
BASELINE_SHA = "892a26347772b7649adb73593cc1740b5efac9bdafc51aff548aa5b4d32aec0c"
ARMS = ("y_only", "chroma_only", "full")


def safe_projection(vector, direction):
    power = direction @ direction
    return float(vector @ direction / power) if power > 0 else None


def response(anchor, outputs, area, expected=None, original=None):
    scores = {arm: statistics(anchor, outputs[arm], area) for arm in ARMS}
    if not scores["full"]["qualified"]:
        return {"pixels": scores["full"]["pixels"], "eligible": False}
    changes = {arm: np.asarray(row["mean_log_chroma_change"]) for arm, row in scores.items()}
    interaction = changes["full"] - changes["y_only"] - changes["chroma_only"]
    judge = changes["full"]
    projections = {arm: safe_projection(vector, judge) for arm, vector in changes.items()}
    result = {
        "eligible": True, "pixels": scores["full"]["pixels"], "responses": scores,
        "relative_to_full": projections, "nonadditive_chroma_norm": float(np.linalg.norm(interaction)),
        "nonadditive_projected_share": safe_projection(interaction, judge),
        "chroma_dominant": (projections["chroma_only"] is not None and projections["chroma_only"] >= .75
                            and abs(projections["y_only"]) <= .25),
    }
    if expected is not None:
        known_face = np.asarray(statistics(original, expected, area)["mean_log_chroma_change"])
        known_rgb = expected.astype(np.float64) - original
        judge_rgb = known_rgb.ravel()
        quality = {}
        for arm, img in outputs.items():
            actual = img.astype(np.float64) - anchor
            face = safe_projection(changes[arm], known_face)
            overall = safe_projection(actual.ravel(), judge_rgb)
            error = float(np.abs(actual - known_rgb).mean())
            quality[arm] = {
                "expected_face_projection": face, "expected_rgb_projection": overall, "rgb_response_mae": error,
                "passed": bool(face is not None and .75 <= face <= 1.25 and overall is not None
                               and .75 <= overall <= 1.25 and error <= 3),
            }
        result["intentional_style"] = quality
    return result


def fragments(engine, image):
    source = engine.analysis(image)
    low = engine.api["TF"].resize(source, (64, 64), antialias=True)
    target_histogram = engine.api["batch_histogram_1D"](engine.api["srgb_to_ycbcr"](source))
    return low, target_histogram


@torch.inference_mode()
def run(prepared, baseline, cube_file, output):
    inputs = prepared / "inputs.json"
    saved = baseline / "results.json"
    if (digest(inputs), digest(saved), digest(cube_file)) != (INPUTS_SHA, BASELINE_SHA, CUBE_SHA):
        raise ValueError("Frozen exposure or Film A parents changed")
    plan = json.loads(inputs.read_bytes())
    original_report = json.loads(saved.read_bytes())
    cube = np.loadtxt(cube_file, skiprows=4, dtype=np.float32).reshape(64, 64, 64, 3).transpose(2, 1, 0, 3)
    if plan["archive_sha256"] != SHA256:
        raise ValueError("Unexpected example archive")
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != original_report["weight_sha256"]:
        raise ValueError("Frozen 13 checkpoint changed")
    output.mkdir()
    records = []
    for case, old in zip(plan["groups"], original_report["groups"], strict=True):
        source = case["group"]
        target = plan["groups"][case["target_group"] - 1]
        folder = output / str(source)
        folder.mkdir()
        if source != old["group"]:
            raise ValueError("Group order changed")
        target_path = prepared / str(target["group"])
        rgb = verified_png(target_path / "nominal.png", target["frames"]["nominal"])
        mask = verified_png(target_path / "nominal-mask.png", target["target_mask_sha256"]) > 0
        refs = {state: verified_png(prepared / str(source) / f"{state}.png", case["frames"][state])
                for state in STATES}
        low, target_hist = fragments(engine, rgb)
        anchor_hist = histogram(engine, refs["nominal"])
        anchor = infer(engine, rgb, low, target_hist, anchor_hist)

        def save_frame(name, image):
            path = folder / f"{name}.png"
            Image.fromarray(image).save(path)
            return digest(path)

        hashes = {"nominal": save_frame("nominal", anchor)}
        if hashes["nominal"] != old["output_sha256"]["nominal"]:
            raise ValueError("Nominal playback changed")
        refs["film_a"] = apply_film(refs["nominal"], cube)
        expected = finish_spatial(rgb, apply_film(rgb, cube), .8)
        report = {}
        for state, ref in refs.items():
            if state == "nominal":
                continue
            alt = histogram(engine, ref)
            hybrid = conditions(anchor_hist.cpu().numpy(), alt.cpu().numpy())
            pictures = {arm: infer(engine, rgb, low, target_hist, torch.from_numpy(hybrid[arm]).to(engine.device))
                        for arm in ARMS}
            for arm, photo in pictures.items():
                name = f"{state}-{arm}"
                hashes[name] = save_frame(name, photo)
            if state in old["output_sha256"] and hashes[f"{state}-full"] != old["output_sha256"][state]:
                raise ValueError("Full exposure playback changed")
            valid = mask & midtone(rgb) & midtone(anchor)
            for photo in pictures.values():
                valid &= midtone(photo)
            film_expected = expected if state == "film_a" else None
            if film_expected is not None:
                valid &= midtone(film_expected)
            report[state] = response(anchor, pictures, valid, film_expected, rgb)
        records.append({"group": source, "cases": report, "hashes": hashes})
    exposures = [row["cases"][state] for row in records for state in STATES[2:]]
    dominant = sum(entry.get("chroma_dominant", False) for entry in exposures)
    looks = [row["cases"]["film_a"].get("intentional_style", {}) for row in records]
    summary = {
        "source_sha256": INPUTS_SHA, "baseline_sha256": BASELINE_SHA, "style_cube_sha256": CUBE_SHA,
        "weight_sha256": engine.checkpoint_sha,
        "exposure_count": len(exposures), "exposure_eligible": sum(x["eligible"] for x in exposures),
        "exposure_chroma_dominant": dominant, "chroma_dominant_threshold_pass": dominant >= 16,
        "intentional_film_a_passes": {arm: sum(x.get(arm, {}).get("passed", False) for x in looks) for arm in ARMS},
        "groups": records,
    }
    save_report(output / "results.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("sdr_cube", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.prepared, args.baseline, args.sdr_cube, args.output)
