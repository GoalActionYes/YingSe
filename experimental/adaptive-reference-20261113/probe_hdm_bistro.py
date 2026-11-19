"""Test the fixed Bistro frame and compare a cross-scene reference grade."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from audit_hdm_hdr_still_alignment import gradients, registration, verified_image
from probe_hdm_hdr_display import EXPOSURES, REC2020_TO_REC709, SOURCE_TO_REC709, decode_pq, display
from probe_hdm_hdr_equal_luminance import PREPARED_SHA, compare, digest, json_to, linear_rgb, verified_json, verified_rgb


def raw_image(root, cases, key, role):
    image = verified_image(root, cases[key]["files"][role])
    expected_type = np.float32 if role == "original" else np.uint16
    if image.shape != (1080, 1920, 3) or image.dtype != expected_type:
        raise ValueError(f"Unexpected archive frame format: {key} {role}")
    return image


def compare_frames(root, cases):
    raw = raw_image(root, cases, "bistro", "original")
    grade = raw_image(root, cases, "bistro", "graded")
    expected = gradients(grade)
    candidates = {"0": registration(gradients(raw), expected)}
    for step in (-1, 1):
        neighbor = raw_image(root, cases, f"bistro-{333 + step}", "original")
        candidates[str(step)] = registration(gradients(neighbor), expected)
    center = candidates["0"]
    margin = center["zero"] - max(candidates["-1"]["zero"], candidates["1"]["zero"])
    return raw, grade, {
        "candidates": candidates, "margin": margin,
        "qualified": center["passed"] and margin >= .002,
    }


def save_rgb(output, name, pixels):
    path = output / f"{name}.png"
    Image.fromarray(pixels).save(path)
    return digest(path)


def render(raw, grade, prepared, output):
    import torch
    from spatial_render import SpatialEngine, finish_spatial

    torch.set_num_threads(4)
    model = SpatialEngine("multiref-v1")
    if model.checkpoint_sha != "d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9":
        raise ValueError("Unexpected 13 checkpoint")
    guide_record = verified_json(prepared / "prepared.json", PREPARED_SHA)["cases"]["showgirl_01"]["images"]["graded"]
    guide = verified_rgb(prepared / guide_record["file"], guide_record["sha256"])
    raw_linear = cv2.resize(raw[..., ::-1], (512, 288), interpolation=cv2.INTER_AREA) @ SOURCE_TO_REC709
    grade_linear = cv2.resize(decode_pq(grade[..., ::-1]), (512, 288), interpolation=cv2.INTER_AREA) @ REC2020_TO_REC709
    author, grade_clipping = display(grade_linear)
    scene_images = {key: display(raw_linear * gain) for key, gain in EXPOSURES.items()}
    common = np.logical_and.reduce(
        [(np.ptp(rgb, axis=-1) <= 12) & (rgb.min(-1) >= 32) & (rgb.max(-1) <= 240)
         for rgb, _ in scene_images.values()]
    )
    board = Image.new("RGB", (1088, 522), "#222730")
    draw = ImageDraw.Draw(board)
    rows = {}
    for index, (exposure, (target, input_clipping)) in enumerate(scene_images.items()):
        prediction = finish_spatial(target, model.prepare(target, guide).apply(target), .8)
        shift = compare(linear_rgb(target), linear_rgb(prediction), grade_linear, common)
        absolute = {
            label: np.abs(rgb.astype(np.int16) - author)
            for label, rgb in (("input", target), ("model", prediction))
        }
        rows[exposure] = {
            "input_sha256": save_rgb(output, f"{exposure}-input", target),
            "model_sha256": save_rgb(output, f"{exposure}-13", prediction),
            "input_clipping": input_clipping,
            "whole_mae": {key: float(errors.mean()) for key, errors in absolute.items()},
            "common_neutral_mae": {key: float(errors[common].mean()) for key, errors in absolute.items()}
            if common.sum() >= 256 else None,
            "preclip_neutral_chromaticity": shift,
        }
        for column, (label, pixels) in enumerate(zip(("input", "reference", "13 (80%)", "author SDR"),
                                                     (target, guide, prediction, author), strict=True)):
            left, top = column * 272 + 8, index * 174
            draw.text((left, top + 5), f"{exposure} / {label}", fill="white")
            board.paste(Image.fromarray(pixels).resize((256, 144), Image.Resampling.LANCZOS), (left, top + 24))
    board.save(output / "comparison.png")
    return {
        "weight_sha256": model.checkpoint_sha, "reference_sha256": guide_record["sha256"],
        "author_sha256": save_rgb(output, "author-sdr", author), "author_clipping": grade_clipping,
        "common_neutral_pixels": int(common.sum()), "exposures": rows,
        "chromaticity_worsening_reproduced": all(row["preclip_neutral_chromaticity"]["passed"] for row in rows.values()),
        "comparison_sha256": digest(output / "comparison.png"),
    }


def run(archive, prepared, output):
    cv2.setNumThreads(1)
    manifest = archive / "manifest.json"
    cases = json.loads(manifest.read_bytes())["cases"]
    if set(cases) != {"bistro", "bistro-332", "bistro-334"}:
        raise ValueError("Expected fixed Bistro source and its two neighbors")
    raw, grade, alignment = compare_frames(archive, cases)
    output.mkdir()
    record = {"source_sha256": digest(manifest), "alignment": alignment}
    if alignment["qualified"]:
        record["render"] = render(raw, grade, prepared, output)
    json_to(output / "results.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.archive, args.prepared, args.output)
