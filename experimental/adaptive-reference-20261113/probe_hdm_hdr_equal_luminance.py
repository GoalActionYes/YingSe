"""Compare archived Poker renders after isolating display chromaticity."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_hdm_hdr_display import EXPOSURES, REC2020_TO_REC709, decode_pq, digest, json_to, original_manifest

SCENE = "poker_fullshot"
PREPARED_SHA = "c407b64b4752862231cf56163cf4d56eb94cc904d75515036c7d258c283798fc"
RENDER_SHA = "9d20d4d0aa1841521eef79d69c2e8fc2afff8c54588a3fe90b5542ff0cde9fe5"
SRGB_TO_XYZ = np.array([
    [506752 / 1228815, 87881 / 245763, 12673 / 70218],
    [87098 / 409605, 175762 / 245763, 12673 / 175545],
    [7918 / 409605, 87881 / 737289, 1001167 / 1053270],
])
Y_WEIGHTS = SRGB_TO_XYZ[1]


def verified_json(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Unexpected manifest: {path}")
    return json.loads(path.read_bytes())


def verified_rgb(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Unexpected image: {path}")
    with Image.open(path) as image:
        if (image.mode, image.size) != ("RGB", (512, 288)):
            raise ValueError(f"Unexpected image format: {path}")
        return np.asarray(image)


def linear_rgb(pixels):
    signal = pixels.astype(np.float64) / 255
    return np.where(signal <= .04045, signal / 12.92, ((signal + .055) / 1.055)**2.4)


def encode(linear):
    bounded = np.clip(linear, 0, 1)
    signal = np.where(bounded <= .0031308, bounded * 12.92, 1.055 * bounded**(1 / 2.4) - .055)
    return np.rint(signal * 255).astype(np.uint8)


def uv(linear):
    x, y, z = (linear @ SRGB_TO_XYZ.T).T
    denominator = x + 15 * y + 3 * z
    if (denominator <= 0).any():
        raise ValueError("Chromaticity is undefined at black")
    return np.column_stack((4 * x / denominator, 9 * y / denominator))


def match_y(source, wanted):
    source_y = source @ Y_WEIGHTS
    ratio = np.divide(wanted, source_y, out=np.zeros_like(wanted), where=source_y > 0)
    scaled = np.where(source_y[..., None] > 0, source * ratio[..., None], wanted[..., None])
    chroma = scaled - wanted[..., None]
    excursion = scaled.max(axis=-1) > 1 + 1e-12
    max_chroma = chroma.max(axis=-1)
    factor = np.ones_like(wanted)
    factor[excursion] = (1 - wanted[excursion]) / max_chroma[excursion]
    return wanted[..., None] + chroma * factor[..., None], excursion


def compare(source, model, grade, mask):
    report = {"pixels": int(mask.sum()), "passed": False}
    valid = mask & ((grade @ Y_WEIGHTS) > 0) & ((model @ Y_WEIGHTS) > 0)
    report["undefined_pixels"] = int(mask.sum() - valid.sum())
    if valid.sum() < 256:
        return report
    expected = uv(grade[valid])
    raw, output = (np.linalg.norm(uv(arm[valid]) - expected, axis=1) for arm in (source, model))
    delta = float((output - raw).mean())
    report.update(
        input_mean=float(raw.mean()), model_mean=float(output.mean()), mean_increase=delta,
        worsened_fraction=float((output > raw).mean()),
        passed=report["undefined_pixels"] == 0 and delta >= .001,
    )
    return report


def preclip_grade(archive):
    import cv2
    from audit_hdm_hdr_still_alignment import verified_image

    record = original_manifest(archive)["cases"][SCENE]["files"]["graded"]
    grade = verified_image(archive, record)
    if grade.dtype != np.uint16:
        raise ValueError("Expected a 16-bit PQ archive")
    cv2.setNumThreads(1)
    resized = cv2.resize(decode_pq(grade[..., ::-1]), (512, 288), interpolation=cv2.INTER_AREA)
    return resized @ REC2020_TO_REC709, record["sha256"]


def run(prepared, rendered, output, archive=None):
    images = verified_json(prepared / "prepared.json", PREPARED_SHA)["cases"][SCENE]["images"]
    model_rows = verified_json(rendered / "results.json", RENDER_SHA)["scores"][SCENE]["exposures"]
    original = {key: verified_rgb(prepared / images[key]["file"], images[key]["sha256"]) for key in EXPOSURES}
    models = {key: verified_rgb(rendered / f"{SCENE}-{key}-13.png", model_rows[key]["output_sha256"])
              for key in EXPOSURES}
    author = verified_rgb(prepared / images["graded"]["file"], images["graded"]["sha256"])
    common = np.logical_and.reduce([
        (np.ptp(pixels, axis=-1) <= 12) & (pixels.min(-1) >= 32) & (pixels.max(-1) <= 240)
        for pixels in original.values()
    ])
    if common.sum() != 1397:
        raise ValueError("Unexpected common neutral-proxy coverage")
    interior = common & np.logical_and.reduce([
        (pixels.min(-1) > 0) & (pixels.max(-1) < 255)
        for pixels in (*original.values(), *models.values(), author)
    ])
    endpoint_input = {f"input_{key}": pixels for key, pixels in original.items()}
    endpoint_model = {f"model_{key}": pixels for key, pixels in models.items()}
    endpoints = {}
    for key, pixels in {**endpoint_input, **endpoint_model, "grade": author}.items():
        values = pixels[common]
        endpoints[key] = {"zero": (values == 0).sum(axis=0).tolist(),
                          "full": (values == 255).sum(axis=0).tolist()}
    preclip, grade_sha = preclip_grade(archive) if archive is not None else (None, None)

    output.mkdir()
    board = Image.new("RGB", (1072, 3 * 176), "#222730")
    draw = ImageDraw.Draw(board)
    author_linear = linear_rgb(author)
    rows = {}
    for index, key in enumerate(EXPOSURES):
        raw = linear_rgb(original[key])
        result = linear_rgb(models[key])
        wanted = result @ Y_WEIGHTS
        adjusted, limited = match_y(raw, wanted)
        if not np.allclose(adjusted @ Y_WEIGHTS, wanted, rtol=0, atol=1e-12):
            raise ValueError("Linear luminance matching failed")
        adjusted_png = encode(adjusted)
        residual = np.abs(linear_rgb(adjusted_png) @ Y_WEIGHTS - wanted)
        path = output / f"poker-{key}-input-color-model-y.png"
        Image.fromarray(adjusted_png).save(path)
        rows[key] = {
            "common": compare(raw, result, author_linear, common),
            "interior": compare(raw, result, author_linear, interior),
            "panel": {
                "sha256": digest(path),
                "gamut_limited_fraction": float(limited.mean()),
                "gamut_limited_common_pixels": int((limited & common).sum()),
                "mean_y_residual_after_encoding": float(residual.mean()),
                "max_y_residual_after_encoding": float(residual.max()),
            },
        }
        if preclip is not None:
            rows[key]["preclip_grade_common"] = compare(raw, result, preclip, common)
        arms = zip(("input", "input color / 13 Y", "13 (80%)", "archival grade"),
                   (original[key], adjusted_png, models[key], author), strict=True)
        for col, (label, pixels) in enumerate(arms):
            left, top = col * 268 + 6, index * 176
            draw.text((left, top + 7), f"{key} / {label}", fill="white")
            board.paste(Image.fromarray(pixels).resize((256, 144), Image.Resampling.LANCZOS), (left, top + 24))

    board.save(output / "comparison.png")
    report = {
        "prepared_sha256": PREPARED_SHA, "render_sha256": RENDER_SHA,
        "scene": SCENE, "rows": rows, "rgb_endpoints_in_common_pixels": endpoints,
        "reproduced": all(row[arm]["passed"] for row in rows.values() for arm in ("common", "interior")),
        "comparison_sha256": digest(output / "comparison.png"),
    }
    if preclip is not None:
        report["pq_grade_sha256"] = grade_sha
        report["preclip_grade_reproduced"] = all(row["preclip_grade_common"]["passed"] for row in rows.values())
    json_to(output / "results.json", report)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("prepared", type=Path)
    cli.add_argument("rendered", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--archive", type=Path)
    args = cli.parse_args()
    run(args.prepared, args.rendered, args.output, args.archive)
