"""Verify the graduated filter switch and score regions frozen on the off image."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from audit_rawtherapee_sky_pair import LUMA, read_png
from render_rawtherapee_graduated_sky import PP3_SHA
from render_rawtherapee_sky_pair import check, disabled

BOXES = {
    "sky": (.35, .15, .69, .26),
    "side_sky": (.51, .46, .57, .53),
    "grass": (.24, .84, .42, .94),
    "trees": (.77, .72, .90, .80),
    "boundary": (.67, .49, .82, .61),
}


def audit(base):
    original = check(base / "author.pp3", PP3_SHA)
    frames, profiles, evidence = {}, {}, {}
    for state, prior in (("off", disabled(original, section=b"Gradient")), ("on", original)):
        digests, pixels = [], []
        for directory in (base / state, base / f"{state}-repeat"):
            if (directory / f"{state}.pp3").read_bytes() != prior:
                raise ValueError("Unexpected profile state")
            path = directory / f"{state}.png"
            frame, icc = read_png(path)
            if state in profiles and profiles[state] != icc:
                raise ValueError("Repeat color profile changed")
            profiles[state] = icc
            pixels.append(frame)
            digests.append(sha256(path.read_bytes()).hexdigest())
        frames[state] = pixels[0]
        evidence[state] = {"png_sha256": digests, "exact_repeat": bool(np.array_equal(*pixels)),
                           "profile_sha256": sha256(prior).hexdigest()}
    off, on = frames["off"], frames["on"]
    if off.shape != on.shape or profiles["off"] != profiles["on"]:
        raise ValueError("Dimensions or ICC changed between states")
    h, w = off.shape[:2]
    if (w, h) != (3662, 5488):
        raise ValueError("Frozen study dimensions changed")
    results = {}
    for name, (x0, y0, x1, y1) in BOXES.items():
        bounds = np.s_[round(h*y0):round(h*y1), round(w*x0):round(w*x1)]
        diff = on[bounds].astype(float) - off[bounds]
        change = diff @ LUMA
        results[name] = {"pixels": int(change.size), "median_luma": float(np.median(change)),
                         "median_abs_luma": float(np.median(np.abs(change)))}
    passed = (all(row["exact_repeat"] for row in evidence.values()) and results["sky"]["median_luma"] <= -2
              and all(results[name]["median_abs_luma"] < 1 for name in ("grass", "trees")))
    return {"source": PP3_SHA, "dimensions": [w, h], "states": evidence, "areas": results, "passed": passed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(audit(args.base), stream, indent=2, allow_nan=False)
        stream.write("\n")
