"""Measure registered material regions in the two fixed Showgirl 2 frames."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_hdm_hdr_display import REC2020_TO_REC709, SOURCE_TO_REC709, decode_pq, digest, display, json_to
from probe_hdm_hdr_equal_luminance import Y_WEIGHTS, uv

SIZE = (512, 288)
CASES = ("showgirl_02-068", "showgirl_02-272")
REGIONS = {"left_lower_cheek": "#58c9ff", "right_shoulder_white_boa": "#f4c251"}


def material_measurement(linear, box):
    x0, y0, x1, y1 = box
    width, height = SIZE
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height and (x1 - x0) * (y1 - y0) >= 40):
        raise ValueError("Invalid registered material rectangle")
    pixels = linear[y0:y1, x0:x1].reshape(-1, 3)
    chroma = uv(pixels)
    return {
        "pixels": len(chroma),
        "median_uv": np.median(chroma, axis=0).tolist(),
        "median_linear_y": float(np.median(pixels @ Y_WEIGHTS)),
    }


def run(source, regions, output):
    import cv2
    from audit_hdm_hdr_still_alignment import gradients, registration, verified_image

    cv2.setNumThreads(1)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    annotation_bytes = regions.read_bytes().replace(b"\r\n", b"\n")
    annotation = json.loads(annotation_bytes)
    if digest(manifest_path) != annotation["manifest_sha256"]:
        raise ValueError("Material annotations belong to different archive files")

    board = Image.new("RGB", (1048, 650), "#222730")
    draw = ImageDraw.Draw(board)
    report = {"manifest": manifest, "roi_sha256": sha256(annotation_bytes).hexdigest(), "cases": {}}
    for index, case in enumerate(CASES):
        files = manifest["cases"][case]["files"]
        raw, grade = (verified_image(source, files[key]) for key in ("original", "graded"))
        if raw.dtype.kind != "f" or grade.dtype != np.uint16:
            raise ValueError("Unexpected HDR frame encodings")
        match = registration(gradients(raw), gradients(grade))
        row = {"registration": match, "measurements": {}}
        report["cases"][case] = row
        arms = (("original", raw[..., ::-1], SOURCE_TO_REC709),
                ("graded", decode_pq(grade[..., ::-1]), REC2020_TO_REC709))
        for col, (role, frame, matrix) in enumerate(arms):
            small = cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA) @ matrix
            photo, clipping = display(small)
            box_record = annotation["cases"][case]
            row["measurements"][role] = {"display_clipping": clipping}
            left, top = 8 + 520 * col, 29 + 318 * index
            board.paste(Image.fromarray(photo), (left, top))
            draw.text((left, top - 20), f"{case} / {box_record['view']} / {role}", fill="white")
            if match["passed"]:
                for name, color in REGIONS.items():
                    box = box_record[name]
                    row["measurements"][role][name] = material_measurement(small, box)
                    x0, y0, x1, y1 = box
                    draw.rectangle((left + x0, top + y0, left + x1 - 1, top + y1 - 1), outline=color)
    draw.text((8, 639), "Blue: lower cheek     Yellow: white boa", fill="white")

    if all(case["registration"]["passed"] for case in report["cases"].values()):
        changes = {}
        for role in ("original", "graded"):
            changes[role] = {}
            a, b = (report["cases"][case]["measurements"][role] for case in CASES)
            for name in REGIONS:
                shift = np.subtract(b[name]["median_uv"], a[name]["median_uv"])
                changes[role][name] = {"delta_uv": shift.tolist(), "uv_distance": float(np.linalg.norm(shift))}
        report["changes"] = changes
        report["lighting_signal_passed"] = changes["original"]["right_shoulder_white_boa"]["uv_distance"] > .01
    else:
        report["lighting_signal_passed"] = False

    output.mkdir()
    board.save(output / "comparison.png")
    report["comparison_sha256"] = digest(output / "comparison.png")
    json_to(output / "results.json", report)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("regions", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    run(args.source, args.regions, args.output)
