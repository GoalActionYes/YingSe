"""Audit native SDR neutral-proxy coverage without rerendering FACD."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from background_intervention import target_regions
from probe_facd_two_image_baseline import STYLES, read_checked


def interior(rgb):
    return (rgb.min(axis=-1) > 0) & (rgb.max(axis=-1) < 255)


def count(mask, neutral):
    eligible = int((mask & neutral).sum())
    return {"pixels": eligible, "fraction": eligible / int(neutral.sum())}


def audit(folder, evidence, output):
    source_bytes = evidence.read_bytes().replace(b"\r\n", b"\n")
    source = json.loads(source_bytes)
    groups = [name for name, row in source["cases"].items()
              if all(filter_row["overlay"]["passed"] for filter_row in row["checks"].values())]
    if len(groups) != 7 or set(source["cases"]) - set(groups) != {"488102"}:
        raise ValueError("Unexpected frozen FACD assignment")

    results = {}
    for group in groups:
        origin = read_checked(folder, group, "Origin", source)
        neutral = target_regions(origin, np.zeros(origin.shape[:2], dtype=bool))["neutral_proxy"]
        common = interior(origin)
        by_style = {}
        for style in STYLES:
            grade = read_checked(folder, group, style, source)
            valid = interior(grade)
            by_style[style] = count(interior(origin) & valid, neutral)
            common &= valid
        support = count(common, neutral)
        results[group] = {
            "neutral_proxy_pixels": int(neutral.sum()), "styles": by_style, "all_styles": support,
            "passed": support["pixels"] >= 256 and support["fraction"] >= .5,
        }
    passed = sum(row["passed"] for row in results.values())
    report = {"source_evidence_sha256_lf": sha256(source_bytes).hexdigest(),
              "groups": results, "passed_groups": passed, "qualified": passed >= 4}
    with output.open("xb") as handle:
        handle.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("folder", "evidence", "output"):
        parser.add_argument(name, type=Path)
    audit(**vars(parser.parse_args()))
