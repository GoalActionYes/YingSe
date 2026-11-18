"""Measure remaining background spill with both known cool and brightness components fixed."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from probe_bg_real_known_cool_axis import (AXIS, BASE, BRIGHTNESS, OTHER, PARENT, PARENT_SHA, ROOT, ROLES, SOURCE_SHA,
                                         axes, compare, cool, digest, finish_spatial, gates, mask_for, read,
                                         target_regions, write)


PRIOR_SHA = "f3db9d6abb9d5f6c3a29953af6d611cbb0e315c5f4401015b0f38b1805638f73"


def fixed_plane(base, expected):
    desired = (base.astype(float)-expected) @ OTHER
    origin = expected.astype(float)
    endpoints = np.stack((-origin/OTHER, (255-origin)/OTHER))
    lo, hi = endpoints.min(axis=0).max(axis=-1), endpoints.max(axis=0).min(axis=-1)
    allowed = np.clip(desired, lo, hi)
    rgb = origin + allowed[..., None]*OTHER
    np.testing.assert_allclose((rgb-origin) @ np.stack([BRIGHTNESS, AXIS]).T, 0, atol=1e-10)
    return np.rint(rgb.clip(0,255)).astype(np.uint8), np.abs(desired-allowed)>1e-9


def run(output):
    prior_file = ROOT/"docs/research/2026-11-19-bg-real-known-cool-axis-evidence.json"
    manifest = PARENT/"results.json"
    if digest(prior_file) != PRIOR_SHA or digest(manifest) != PARENT_SHA:
        raise ValueError("Frozen paired controls changed")
    previous = json.loads(prior_file.read_bytes())["cases"]
    records = {row["pair"]: row for row in json.loads(manifest.read_bytes())["cases"]}
    original = ROOT/"docs/research/2026-11-18-bg-real-style-cross-evidence.json"
    if digest(original) != SOURCE_SHA:
        raise ValueError("Frozen 13 outputs changed")
    hashes = json.loads(original.read_bytes())["cases"]
    output.mkdir()
    result = {}
    for pair, prior in previous.items():
        row = records[pair]
        target = read(row["input_file"])
        edited = finish_spatial(target, cool(target), .8)
        human = mask_for(row, pair.split("-")[0])
        masks = target_regions(target, human)
        masks["neutral_person"] = masks["neutral_proxy"] & human
        outputs, limited, stats, comparison = {}, {}, {}, {}
        for role in ROLES:
            source = BASE/"bg-real-style-cross"/f"{pair}-{role}.png"
            earlier = BASE/"bg-real-known-cool-axis"/f"{pair}-{role}.png"
            if digest(source) != hashes[pair]["output_sha256"][role] or digest(earlier) != prior["adjustments"][role]["output_sha256"]:
                raise ValueError("Frozen image changed")
            with Image.open(source) as frame, Image.open(earlier) as old:
                rgb = np.asarray(frame.convert("RGB"))
                comparison[role] = np.asarray(old.convert("RGB"))
            wanted = edited if role.endswith("-cool") else target
            candidate, limited[role] = fixed_plane(rgb, wanted)
            error = (candidate.astype(float)-wanted) @ np.stack([BRIGHTNESS, AXIS]).T
            stats[role] = {"limited": float(limited[role].mean()), "rounding_max": np.abs(error).max(axis=(0,1)).tolist(),
                           "sha256": write(output/f"{pair}-{role}.png", candidate)}
            outputs[role] = candidate
        expected = edited.astype(np.int16)-target
        scores, remainder = {}, {}
        for region in ("person", "neutral_person", "whole"):
            mask = masks[region]
            scores[region] = compare(outputs, expected, mask)
            remainder[region] = {}
            for x in ("", "-cool"):
                unrestricted = mask & ~limited[f"C1{x}"] & ~limited[f"F1{x}"]
                error = outputs[f"F1{x}"].astype(np.int16)-outputs[f"C1{x}"]
                remainder[region]["cool" if x else "base"] = {"pixels": int(unrestricted.sum()),
                    "rgb_mae": float(np.abs(error[unrestricted]).mean()) if unrestricted.any() else None}
            if mask.any():
                scores[region]["axes"] = axes(outputs, mask)
        checks = gates(scores)
        sheet = Image.new("RGB", (1064,566), "#20242a")
        draw = ImageDraw.Draw(sheet)
        for y, (caption, images) in enumerate((("Known cool", comparison), ("Known cool + luma", outputs))):
            for x, role in enumerate(ROLES):
                draw.text((8+x*264, 5+y*278), f"{caption} / {role}", fill="white")
                sheet.paste(Image.fromarray(images[role]), (8+x*264, 24+y*278))
        panel = output/f"{pair}-comparison.png"
        sheet.save(panel)
        result[pair] = {"scores": scores, "unlimited": remainder, "checks": checks, "adjustments": stats,
                        "passed": all(v for v in checks.values() if v is not None), "panel_sha256": digest(panel)}
    report = {"cases": result, "passed": all(r["passed"] for r in result.values()), "prior_sha256": PRIOR_SHA,
              "source_sha256": digest(Path(__file__)),
              "protocol_sha256": digest(Path(__file__).with_name("bg-real-known-cool-luma-protocol.md"))}
    (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
