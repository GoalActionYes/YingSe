"""Audit fixed MLS light pairs for measurable local integer translation."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_mls_multiscene_intake import SCENES
from fetch_mls_linear_pilot import LIGHTS
from independent_coco import digest
from probe_fau_two_light_response import encode
from probe_mls_light_response import registration
from probe_mls_linear_pilot import read


RADIUS = 6
KNOWN = (3, -4)


def gradient(frame):
    values = frame.astype(np.float32) / 65535
    gy = (values[2:, 1:-1] - values[:-2, 1:-1]) * .5
    gx = (values[1:-1, 2:] - values[1:-1, :-2]) * .5
    return np.stack((gy, gx), axis=-1)


def measure(first, second, channel):
    height, width = first.shape[:2]
    tiles = []
    for row in range(4):
        for col in range(4):
            y, x = np.mgrid[RADIUS + row * height // 4:(row + 1) * height // 4 - RADIUS - 1:3,
                            RADIUS + col * width // 4:(col + 1) * width // 4 - RADIUS - 1:3]
            vector = first[y, x, channel]
            strength = np.linalg.norm(vector, axis=-1)
            selected = (strength >= np.quantile(strength, .75)) & (strength > 1e-4)
            if selected.sum() < 64:
                continue
            y, x, vector = y[selected], x[selected], vector[selected]
            vector_norm = np.linalg.norm(vector)
            scores = []
            for dy in range(-RADIUS, RADIUS + 1):
                for dx in range(-RADIUS, RADIUS + 1):
                    candidate = second[y + dy, x + dx, channel]
                    norm = vector_norm * np.linalg.norm(candidate)
                    score = float(np.sum(vector * candidate) / norm) if norm else -1.0
                    scores.append((score, dy, dx))
            best, alternate = sorted(scores, reverse=True)[:2]
            margin = best[0] - alternate[0]
            tiles.append({"tile": [row, col], "score": best[0], "margin": margin,
                          "shift": list(best[1:]), "vote": best[0] >= .50 and margin >= .015})
    votes = np.array([tile["shift"] for tile in tiles if tile["vote"]])
    median = np.median(votes, axis=0) if len(votes) else None
    agreement = float((np.abs(votes - median).max(axis=1) <= 1).mean()) if len(votes) else 0.
    return {"tiles": tiles, "votes": len(votes), "agreement": agreement,
            "median": median.tolist() if median is not None else None,
            "reliable": len(votes) >= 6 and agreement >= .75}


def channels(first, second):
    return {key: measure(first, second, i) for i, key in enumerate("RGB")}


def supports(result, expected):
    return all(result[key]["reliable"] and np.max(np.abs(np.array(result[key]["median"]) - expected)) <= 1
               for key in "RG")


def run(original, added, result_path):
    intake = {folder: json.loads((folder / "intake.json").read_bytes())["entries"]
              for folder in (original, added)}
    results = {}
    for scene in SCENES:
        source = original if scene in ("02", "03") else added
        names = [f"{scene}_{light}.png" for light in LIGHTS]
        first, second = [read(source / name, intake[source][name]) for name in names]
        grad_first, grad_second = map(gradient, (first, second))
        real = channels(grad_first, grad_second)
        zero = channels(grad_first, grad_first)
        shifted = channels(grad_first, np.roll(grad_first, KNOWN, axis=(0, 1)))
        rgb8 = [encode((frame.astype(np.uint32) * 4).clip(0, 65535)) for frame in (first, second)]
        results[scene] = {"files": {name: intake[source][name]["sha256"] for name in names},
                          "legacy_correlation": registration(*rgb8), "real": real, "zero_control": zero,
                          "shift_control": shifted, "real_near_zero": supports(real, (0, 0)),
                          "controls_valid": supports(zero, (0, 0)) and supports(shifted, KNOWN)}
    validated = all(row["controls_valid"] for row in results.values()) and all(
        row["real_near_zero"] for row in results.values() if row["legacy_correlation"] >= .5)
    output = {"source_sha256": digest(Path(__file__)), "imposed_shift": KNOWN,
              "control_panel_valid": validated, "scenes": results}
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("original", "added", "result"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    run(args.original, args.added, args.result)
