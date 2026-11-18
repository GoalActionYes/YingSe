"""Test an edge-based split-tone cue on frozen real-image controls."""

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image, ImageDraw
import torch

from background_intervention import LUMA, target_regions
from probe_bg_real_style_cross import ROLES, write
from probe_gray_edge_correction import PARENT_SHA
from probe_local_edit_toggle import COOL, mask_for
from probe_reference_chroma_spread import BASE, PARENT, ROOT, digest, read
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


AXIS = COOL / np.linalg.norm(COOL)
ORTHOGONAL = np.cross(LUMA, AXIS)
ORTHOGONAL /= np.linalg.norm(ORTHOGONAL)
STYLES = ("base", "uniform", "light-cool", "dark-cool")
SLOPES = {"base": 0, "uniform": 0, "light-cool": .18, "dark-cool": -.18}


def quantize(rgb):
    return np.rint(rgb.clip(0, 255)).astype(np.uint8)


def versions(rgb):
    base = rgb.astype(np.float64) * (207 / 255) + 24
    pivoted_y = base @ LUMA - 128
    results = {}
    for style in STYLES:
        shift = 6 if style == "uniform" else SLOPES[style] * pivoted_y[..., None]
        grade = base + shift * AXIS
        if grade.min() < 0 or grade.max() > 255:
            raise ValueError("Authored grade exceeded headroom")
        results[style] = quantize(grade)
    return results


def slope(x, y):
    centered = x - x.mean()
    energy = np.sum(centered ** 2)
    return float(np.sum(centered * y) / energy) if energy else 0.0


def estimate(photo):
    image = np.asarray(Image.fromarray(photo).resize((256, 256), Image.Resampling.BILINEAR), dtype=float)
    y, cold, other = image @ LUMA, image @ AXIS, image @ ORTHOGONAL
    ratios = []
    for first, second in ((np.s_[:, :-8], np.s_[:, 8:]), (np.s_[:-8], np.s_[8:])):
        dy = y[second] - y[first]
        use = ((y[first] >= 40) & (y[first] <= 215) & (y[second] >= 40) & (y[second] <= 215)
               & (np.abs(dy) >= 18) & (np.abs(dy) <= 80)
               & (np.abs(other[second] - other[first]) <= 4))
        ratios.append((cold[second] - cold[first])[use] / dy[use])
    ratios = np.concatenate(ratios)
    return dict(pairs=len(ratios), local=float(np.median(ratios)) if len(ratios) >= 256 else None,
                global_slope=slope(y, cold))


def render(baseline, target, local):
    if local is None or abs(local) < .08:
        return baseline
    y = target @ LUMA
    added = (baseline.astype(float) - target) @ AXIS
    change = (.8 * local - slope(y, added)) * (y - 128)
    return quantize(baseline + change[..., None] * AXIS)


def cue_pass(value, style):
    if value is None:
        return False
    if SLOPES[style]:
        return abs(value - SLOPES[style]) <= .06
    return abs(value) < .08


def scores(reference_outputs, truth, neutral):
    def difference(images):
        return (images["light-cool"].astype(float) - images["dark-cool"]) @ AXIS

    expected = difference(truth)
    result = {}
    for name, variants in reference_outputs.items():
        error = np.abs(difference(variants) - expected)
        result[name] = {"whole": float(error.mean()),
                        "neutral": float(error[neutral].mean()) if int(neutral.sum()) >= 256 else None}
    baseline, edge = result["13"], result["edge"]
    n0, n1 = baseline["neutral"], edge["neutral"]
    result["passed"] = (n0 is not None and n1 is not None and n0 - n1 >= max(.5, .2 * n0)
                        and edge["whole"] <= baseline["whole"] + .5)
    return result


def panel(path, out):
    rows = ["Reference", "Target edit", "YingSe 13", "Edge cue"]
    sheet = Image.new("RGB", (888, 1000), "#191d23")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        for c, style in enumerate(STYLES):
            x, y = 8 + 220 * c, 8 + 246 * r
            draw.text((x, y), f"{row} / {style}", fill="white")
            image = Image.fromarray(out[row][style])
            image.thumbnail((212, 212))
            sheet.paste(image, (x, y + 23))
    sheet.save(path, quality=92, subsampling=0)


@torch.inference_mode()
def run(output):
    start = monotonic()
    torch.set_num_threads(4)
    manifest = ROOT / "docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json"
    cross = ROOT / "docs/research/2026-11-18-bg-real-style-cross-evidence.json"
    frozen = json.loads(manifest.read_bytes())
    if (digest(manifest) != json.loads(cross.read_bytes())["source_evidence_sha256"]
            or digest(PARENT / "results.json") != PARENT_SHA or len(frozen["cases"]) != 4):
        raise ValueError("Frozen inputs changed")
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    output.mkdir()
    cohort = {row["pair"]: row for row in json.loads((PARENT / "results.json").read_bytes())["cases"]}
    results = {}
    for source, item in frozen["cases"].items():
        pair = item["target_pair"]
        row = cohort[pair]
        target_styles = versions(read(row["input_file"]))
        target = target_styles["base"]
        truth = {style: finish_spatial(target, image, .8) for style, image in target_styles.items()}
        human = mask_for(row, pair.split("-")[0])
        neutral = target_regions(target, human)["neutral_proxy"] & human
        report = {"source": source, "neutral_pixels": int(neutral.sum()), "roles": {}}
        neutral_controls = {}
        for role in ROLES:
            entry = item["input"][role]
            path = BASE / "bg-real-pixel-inputs" / entry["name"]
            if digest(path) != entry["sha256"]:
                raise ValueError(f"Unexpected reference {pair}/{role}")
            refs = versions(decode_image(path))
            edited = {"13": {}, "edge": {}}
            estimates, hashes = {}, {}
            for style, ref in refs.items():
                cue = estimate(ref)
                cue["local_pass"] = cue_pass(cue["local"], style)
                cue["global_pass"] = cue_pass(cue["global_slope"], style)
                estimates[style] = cue
                baseline = finish_spatial(target, engine.prepare(target, ref).apply(target), .8)
                edited["13"][style] = baseline
                edited["edge"][style] = render(baseline, target, cue["local"])
                hashes[style] = {key: write(output / f"{pair}-{role}-{style}-{key}.png", arm[style])
                                 for key, arm in edited.items()}
            record = scores(edited, truth, neutral)
            neutral_controls[role] = {key: {style: (arm[style] @ AXIS)[neutral]
                                            for style in ("base", "uniform")} for key, arm in edited.items()}
            record.update(estimates=estimates, outputs=hashes)
            if pair == "872-1268":
                comparison = output / f"white-jersey-{role}.jpg"
                panel(comparison, {"Reference": refs, "Target edit": truth,
                                   "YingSe 13": edited["13"], "Edge cue": edited["edge"]})
                record["panel_sha256"] = digest(comparison)
            report["roles"][role] = record
        if int(neutral.sum()) >= 256:
            expected = ((truth["uniform"].astype(float) - truth["base"]) @ AXIS)[neutral]
            report["controls"] = {
                key: {
                    "base_swap": float(np.abs(neutral_controls["C1"][key]["base"]
                                              - neutral_controls["F1"][key]["base"]).mean()),
                    "uniform_error": {
                        role: float(np.abs(values[key]["uniform"] - values[key]["base"] - expected).mean())
                        for role, values in neutral_controls.items()
                    },
                } for key in ("13", "edge")
            }
        else:
            report["controls"] = None
        results[pair] = report
    evidence = {"checkpoint_sha256": engine.checkpoint_sha, "input_evidence_sha256": digest(manifest),
                "source_sha256": digest(Path(__file__)),
                "protocol_sha256": digest(Path(__file__).with_name("two-tone-edge-protocol.md")),
                "cases": results}
    (output / "results.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if monotonic() - start > 120:
        raise RuntimeError("Run exceeded two-minute budget")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
