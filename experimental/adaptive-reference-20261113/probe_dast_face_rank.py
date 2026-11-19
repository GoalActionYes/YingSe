"""Screen fixed local order statistics on frozen DAST reference frames."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from probe_dast_exposure_response import save_report, verified_png
from probe_dast_face_surround import EDITS_SHA, dilate, warmth
from probe_dast_local_edit_cross import INPUT_SHA, STATES, read_ledger


def superiority(face, ring):
    face_counts = np.bincount(face, minlength=256).astype(np.int64)
    ring_counts = np.bincount(ring, minlength=256).astype(np.int64)
    wins = face_counts @ (ring_counts.cumsum() - ring_counts)
    ties = face_counts @ ring_counts
    return float((wins + .5 * ties) / (face.size * ring.size))


def score(rgb, face, ring):
    inside, around = rgb[face], rgb[ring]
    red, blue = (superiority(inside[:, c], around[:, c]) for c in (0, 2))
    values = warmth(rgb)
    clipped = (rgb == 0).any(axis=2) | (rgb == 255).any(axis=2)
    return {"red": red, "blue": blue, "rank": red - blue,
            "numeric": float(values[face].mean() - values[ring].mean()),
            "face_clipped": float(clipped[face].mean()), "ring_clipped": float(clipped[ring].mean())}


def extrema(rows, key):
    off = [row["off"][key] for row in rows]
    on = [row["on"][key] for row in rows]
    return {"margin": min(on) - max(off), "off_span": max(off) - min(off)}


def run(source, edits):
    inputs = read_ledger(source, "inputs.json", INPUT_SHA)
    edit_groups = read_ledger(edits, "edits.json", EDITS_SHA)["groups"]
    groups = []
    for entry in inputs["groups"]:
        name = str(entry["group"])
        mask = verified_png(source / name / "nominal-mask.png", entry["target_mask_sha256"])
        core = Image.fromarray(mask)
        face, ring = mask > 0, dilate(core, 40) & ~dilate(core, 24)
        counts = {"face": int(face.sum()), "ring": int(ring.sum())}
        if min(counts.values()) < 1024:
            raise ValueError(f"Insufficient fixed region pixels for group {name}")
        rows = []
        for state in STATES:
            off = verified_png(source / name / f"{state}.png", entry["frames"][state])
            on = verified_png(edits / name / f"{state}-on.png", edit_groups[name][state]["edited_sha256"])
            rows.append({"state": state, "off": score(off, face, ring), "on": score(on, face, ring),
                         "ring_changed": float((off[ring] != on[ring]).any(axis=1).mean())})
        groups.append({"group": entry["group"], "pixels": counts, "rows": rows,
                       **{key: extrema(rows, key) for key in ("rank", "numeric")}})
    rows = [row for group in groups for row in group["rows"]]
    joint = {key: extrema(rows, key) for key in ("rank", "numeric")}
    passed = sum(group["rank"]["margin"] >= .05 and group["rank"]["off_span"] <= .05 for group in groups)
    return {"inputs_sha256": INPUT_SHA, "edits_sha256": EDITS_SHA, "groups": groups, "joint": joint,
            "passing_groups": passed, "qualified": passed >= 4 and joint["rank"]["margin"] >= .05}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "edits", "report"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    save_report(args.report, run(args.source, args.edits))
