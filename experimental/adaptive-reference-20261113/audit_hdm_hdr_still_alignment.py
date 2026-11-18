"""Check registration of fixed HdM ordinal frame candidates without grading them."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

COEFFICIENTS = np.array([.114, .587, .299], dtype=np.float32)
RADIUS = 3


def verified_image(source, record):
    raw = (source / record["file"]).read_bytes()
    if len(raw) != record["bytes"] or sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Archive frame changed")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("Expected finite three-channel image")
    return image


def gradients(image):
    light = image.astype(np.float32) @ COEFFICIENTS
    maximum = max(image.shape[:2])
    height, width = image.shape[:2]
    geometry = (round(width*512/maximum), round(height*512/maximum))
    # Approximate brightness only for geometry, never a display transform or color score.
    if image.dtype.kind == "f":
        light = np.power(np.maximum(light, 0), .3)
    small = cv2.resize(light, geometry, interpolation=cv2.INTER_AREA)
    return np.stack((np.diff(small, axis=0)[:, :-1], np.diff(small, axis=1)[:-1]), axis=-1)


def registration(first, second):
    if first.shape != second.shape:
        return {"reason": "gradient layouts differ", "passed": False}
    height, width = first.shape[:2]
    anchor = first[RADIUS:-RADIUS, RADIUS:-RADIUS].astype(np.float64).ravel()
    anchor -= anchor.mean()
    reference_scale = np.linalg.norm(anchor)
    values = {}
    for y in range(-RADIUS, RADIUS+1):
        for x in range(-RADIUS, RADIUS+1):
            changed = second[RADIUS+y:height-RADIUS+y, RADIUS+x:width-RADIUS+x].astype(np.float64).ravel()
            changed -= changed.mean()
            denominator = reference_scale*np.linalg.norm(changed)
            values[y, x] = float(np.dot(anchor, changed)/denominator) if denominator else 0.
    peak = max(values, key=values.get)
    zero = values[0, 0]
    return {"zero": zero, "peak": values[peak], "peak_offset_y_x": list(peak),
            "passed": peak == (0, 0) and zero >= .65}


def run(source, dest):
    cv2.setNumThreads(1)
    path = source/"manifest.json"
    index = json.loads(path.read_bytes())
    scores = {}
    for scene, row in index["cases"].items():
        pictures = {role: verified_image(source, value) for role, value in row["files"].items()}
        stats = {role: {"shape": list(image.shape), "dtype": str(image.dtype),
                        "range": [float(image.min()), float(image.max())]} for role, image in pictures.items()}
        matched = registration(gradients(pictures["original"]), gradients(pictures["graded"]))
        scores[scene] = {"source_frames": stats, "registration": matched}
    result = {"manifest_sha256": sha256(path.read_bytes()).hexdigest(), "opencv": cv2.__version__, "cases": scores}
    with dest.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.input, args.output)
