"""Check whether calibrated near-PS lamps isolate a portrait color change."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REVISION = "30881339ddac975c1faab3df5126b49034e22c13"


def run(folder):
    tree = json.loads((folder / "tree.json").read_bytes())
    if tree["sha"] != REVISION or tree["truncated"]:
        raise ValueError("Expected the fixed, complete published tree")
    files = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
    path = folder / "light.mat"
    data = path.read_bytes()
    entry = files["Datasets/light.mat"]
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if len(data) != entry["size"] or blob != entry["sha"]:
        raise ValueError("Light calibration identity mismatch")
    calibration = loadmat(path)
    locations, power = (calibration[name] for name in ("S", "Phi"))
    if locations.shape != (8, 3) or power.shape != (8, 3) or not np.isfinite(locations).all() or not np.isfinite(power).all():
        raise ValueError("Unexpected calibration dimensions or values")
    if np.any(power <= 0):
        raise ValueError("Invalid positive RGB intensities")
    normalized = power / np.linalg.norm(power, axis=1, keepdims=True)
    pairs = []
    for i, j in itertools.combinations(range(8), 2):
        angle = float(np.degrees(np.arccos(np.clip(normalized[i] @ normalized[j], -1, 1))))
        distance = float(np.linalg.norm(locations[i] - locations[j]))
        pairs.append({"lamps": [i + 1, j + 1], "rgb_angle_degrees": angle, "distance_mm": distance,
                      "eligible": angle >= 3 and distance <= 5})
    sources = {}
    for person in ("Human1", "Human2"):
        sizes = [files[f"Datasets/{person}/photometric_sample_raw_{lamp:04}.png"]["size"] for lamp in range(1, 9)]
        sources[person] = {"individual_lamps": len(sizes), "min_bytes": min(sizes), "max_bytes": max(sizes)}
    return {"revision": REVISION, "calibration_sha256": hashlib.sha256(data).hexdigest(), "git_blob": blob,
            "sources": sources, "pairs": pairs, "eligible_pairs": sum(row["eligible"] for row in pairs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(run(parser.parse_args().folder), indent=2, allow_nan=False))
