"""Compare the existing 13 renderer and sky-conditioned histograms on true edit switches."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from audit_rawtherapee_graduated_sky import BOXES as GRAD_BOXES
from audit_rawtherapee_sky_pair import AREAS, BOXES, LUMA, read_png
from check_scene_regions import ROOT, digest, SegformerImageProcessor, SegformerForSemanticSegmentation
from probe_tst_neutral_direction_fit import json_to, load, save
from region_histogram import predict
from spatial_render import SpatialEngine, finish_spatial

NAMES = ("R6", "graduated")
STRENGTH = .8


def sources(base):
    r6 = base / "rawtherapee-sky-pair-v1/single-thread"
    grad = base / "rawtherapee-graduated-sky-v1"
    a = load(r6 / "results.json", "3c8256292d0c92f11df08cc81c6bd865866f541188ee3de6339fc4b202045bf5")
    b = load(grad / "results.json", "aaafb4d68acac62b53dc779177861e1099e2e506f6847712a603803e2b9a1337")
    a = a["cases"]["R6_06283"]
    if not (a["passed"] and b["passed"]):
        raise ValueError("Unqualified reference controls")
    return {
        "R6": [(r6 / s / f"R6_06283-{s}.png", a["fingerprints"][s]["png"]) for s in ("off", "on")],
        "graduated": [(grad / s / f"{s}.png", b["states"][s]["png_sha256"][0]) for s in ("off", "on")],
    }


def image(item):
    path, sha = item
    if digest(path) != sha:
        raise ValueError("Pinned image identity changed")
    pixels, _ = read_png(path)
    working = Image.fromarray(pixels)
    working.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    return np.array(working)


def sky_model(device):
    assets = ROOT / ".tools/scene-segformer"
    import json
    manifest = json.loads((assets / "manifest.json").read_bytes())
    for file in manifest["files"]:
        if digest(assets / file["path"]) != file["sha256"]:
            raise ValueError("SegFormer file changed")
    processor = SegformerImageProcessor.from_pretrained(assets, local_files_only=True)
    model, info = SegformerForSemanticSegmentation.from_pretrained(
        assets, local_files_only=True, use_safetensors=True, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Incomplete SegFormer restoration")
    model.eval().requires_grad_(False).to(device)
    category, = [i for i, label in model.config.id2label.items() if label.strip().lower() == "sky"]

    def segment(rgb):
        result = processor(images=rgb, return_tensors="pt").to(device)
        logits = F.interpolate(model(**result).logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        return logits.softmax(1)[0, category].cpu().numpy()

    return segment


def score(frames, truth, boxes, control):
    h, w = truth[0].shape[:2]
    result = {}
    for key, (x0, y0, x1, y1) in boxes.items():
        area = np.s_[round(h*y0):round(h*y1), round(w*x0):round(w*x1)]
        a, b = (x[area].astype(float) for x in frames)
        off, on = (x[area].astype(float) for x in truth)
        first, second = (x[area].astype(float) for x in control)
        actual, expected = (b-a) @ LUMA, (on-off) @ LUMA
        norm = np.square(expected).sum()
        result[key] = {
            "projection": float((actual*expected).sum()/norm) if norm else None,
            "switch_mae": float(np.abs(actual-expected).mean()),
            "switch_median": float(np.median(actual)),
            "abs_switch_median": float(np.median(np.abs(actual))),
            "state_rgb_mae": [float(np.abs(x-y).mean()) for x,y in ((a,off),(b,on))],
            "worst_extra_mae": float(max(np.abs(a-off).mean()-np.abs(first-off).mean(),
                                         np.abs(b-on).mean()-np.abs(second-on).mean())),
        }
    sky = result["sky"]
    fg = ("foreground_1", "foreground_2") if "foreground_1" in boxes else ("grass", "trees")
    passed = (sky["projection"] is not None and .75 <= sky["projection"] <= 1.25 and sky["switch_mae"] <= 3
              and all(result[k]["abs_switch_median"] < 1 and result[k]["worst_extra_mae"] <= 2 for k in fg))
    return {"regions": result, "passed": passed}


def grade_pair(engine, target, guide, target_mask, segment):
    if np.array_equal(target, guide):
        return target.copy(), target.copy(), {"applied": False, "identity": True}
    baseline = engine.prepare(target, guide).apply(target)
    candidate, info = predict(engine, target, guide, target_mask, segment(guide), baseline)
    return baseline, candidate, info


@torch.inference_mode()
def run(base, out):
    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    segment = sky_model(engine.device)
    inputs = sources(base)
    targets = {n: image(items[0]) for n, items in inputs.items()}
    out.mkdir()
    cases = {}
    for target_name, donor_name in (NAMES, NAMES[::-1]):
        target = targets[target_name]
        target_mask = segment(target)
        images = {"13": [], "regional": []}
        hashes, support = {}, []
        for state, item in zip(("off", "on"), inputs[donor_name], strict=True):
            guide = image(item)
            baseline, candidate, info = grade_pair(engine, target, guide, target_mask, segment)
            support.append(info)
            for name, grade in (("13", baseline), ("regional", candidate)):
                frame = finish_spatial(target, grade, STRENGTH)
                images[name].append(frame)
                hashes[f"{name}-{state}"] = save(out / f"{target_name}-{name}-{state}.png", frame)
        first, second, identity = grade_pair(engine, target, target.copy(), target_mask, segment)
        if not identity["identity"] or not (np.array_equal(first, target) and np.array_equal(second, target)):
            raise ValueError("Same-image shortcut changed")
        blank = np.zeros_like(target_mask)
        result, fallback = predict(engine, target, guide, blank, segment(guide), baseline)
        if fallback["applied"] or not np.array_equal(result, baseline):
            raise ValueError("No-sky fallback changed")
        cases[target_name] = {"hashes": hashes, "support": support, "images": images}
    for name, row in cases.items():
        target = targets[name]
        truth = [target, finish_spatial(target, image(inputs[name][1]), STRENGTH)]
        boxes = dict(zip(AREAS, BOXES["R6_06283"], strict=True)) if name == "R6" else GRAD_BOXES
        images = row.pop("images")
        row["scores"] = {arm: score(frames, truth, boxes, images["13"]) for arm, frames in images.items()}
    json_to(out / "results.json", {"cases": cases, "passed": all(r["scores"]["regional"]["passed"] for r in cases.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
