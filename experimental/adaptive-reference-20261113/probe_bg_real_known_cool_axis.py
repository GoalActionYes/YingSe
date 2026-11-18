"""Check the residual background spill after replaying a known cooling operation."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from background_intervention import LUMA, target_regions
from probe_bg_real_style_cross import compare, cool, gates, write
from probe_gray_edge_correction import PARENT_SHA
from probe_local_edit_toggle import COOL, mask_for
from probe_reference_chroma_spread import BASE, PARENT, ROOT, digest, read
from spatial_render import finish_spatial


SOURCE_SHA = "b824303d1a13792fb896db79ef1465a7da12be7c1834d701840272e2dbf7bb36"
ROLES = ("C1", "F1", "C1-cool", "F1-cool")
AXIS = COOL / np.linalg.norm(COOL)
BRIGHTNESS = LUMA / np.linalg.norm(LUMA)
OTHER = np.cross(BRIGHTNESS, AXIS)


def replay(base, desired):
    base = base.astype(float)
    delta = (desired-base) @ AXIS
    lows = -base / AXIS
    highs = (255-base) / AXIS
    allowed = np.clip(delta, np.minimum(lows, highs).max(axis=-1), np.maximum(lows, highs).min(axis=-1))
    composed = base + allowed[..., None]*AXIS
    np.testing.assert_allclose(composed @ LUMA, base @ LUMA, atol=1e-10, rtol=0)
    np.testing.assert_allclose(composed @ OTHER, base @ OTHER, atol=1e-10, rtol=0)
    finished = np.rint(composed.clip(0,255)).astype(np.uint8)
    return finished, {"limited_ratio": float((np.abs(delta-allowed)>1e-9).mean()),
                      "max_rounded_y_delta": float(np.abs((finished-base)@LUMA).max())}


def axes(outputs, mask):
    values = {}
    for suffix in ("", "-cool"):
        delta = outputs[f"F1{suffix}"].astype(float) - outputs[f"C1{suffix}"]
        values["cool" if suffix else "base"] = dict(zip(("brightness", "cool", "other_chroma"),
            np.abs(delta[mask] @ np.stack([BRIGHTNESS, AXIS, OTHER]).T).mean(axis=0).tolist()))
    return values


def panel(path, baseline, candidate):
    out = Image.new("RGB", (1064, 570), "#1c2128")
    draw = ImageDraw.Draw(out)
    for j, (group, images) in enumerate((("YingSe 13", baseline), ("Known cool axis", candidate))):
        for i, role in enumerate(ROLES):
            x, y = 8+i*264, 6+j*280
            draw.text((x, y), f"{group} / {role}", fill="white")
            out.paste(Image.fromarray(images[role]), (x, y+20))
    out.save(path)


def run(folder):
    manifest = ROOT/"docs/research/2026-11-18-bg-real-style-cross-evidence.json"
    raw_index = PARENT/"results.json"
    if digest(manifest) != SOURCE_SHA or digest(raw_index) != PARENT_SHA:
        raise ValueError("Frozen controls changed")
    original = json.loads(manifest.read_bytes())["cases"]
    sources = {n["pair"]: n for n in json.loads(raw_index.read_bytes())["cases"]}
    np.testing.assert_allclose(LUMA@AXIS, 0, atol=1e-12, rtol=0)
    folder.mkdir()
    report = {}
    for pair, metadata in original.items():
        row = sources[pair]
        src = read(row["input_file"])
        known = finish_spatial(src, cool(src), .8)
        expected = known.astype(np.int16) - src
        human = mask_for(row, pair.split("-")[0])
        regions = target_regions(src, human)
        regions["neutral_person"] = regions["neutral_proxy"] & human
        baseline, candidate, adjustments = {}, {}, {}
        for role in ROLES:
            path = BASE/"bg-real-style-cross"/f"{pair}-{role}.png"
            if digest(path) != metadata["output_sha256"][role]:
                raise ValueError("Frozen 13 output changed")
            with Image.open(path) as decoded:
                baseline[role] = np.asarray(decoded.convert("RGB"))
            candidate[role], moves = replay(baseline[role], known if role.endswith("-cool") else src)
            moves["output_sha256"] = write(folder/f"{pair}-{role}.png", candidate[role])
            adjustments[role] = moves
        compare_regions = ("person", "neutral_person", "whole")
        scores = {name: {region: compare(data, expected, regions[region]) for region in compare_regions}
                  for name, data in (("baseline", baseline), ("candidate", candidate))}
        for region in compare_regions:
            for name, data in (("baseline", baseline), ("candidate", candidate)):
                if regions[region].any():
                    scores[name][region]["axis_means"] = axes(data, regions[region])
        conditions = {name: gates(values) for name, values in scores.items()}
        path = folder/f"{pair}-comparison.png"
        panel(path, baseline, candidate)
        report[pair] = {"scores": scores, "conditions": conditions, "adjustments": adjustments,
                        "passed": all(x for x in conditions["candidate"].values() if x is not None),
                        "panel_sha256": digest(path)}
    artifact = {"cases": report, "passed": all(row["passed"] for row in report.values()), "parent_sha256": SOURCE_SHA,
                "source_sha256": digest(Path(__file__)),
                "protocol_sha256": digest(Path(__file__).with_name("bg-real-known-cool-axis-protocol.md"))}
    (folder/"results.json").write_text(json.dumps(artifact, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
