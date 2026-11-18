"""Check frozen screen vetoes on newly registered TV and monitor faces."""

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from probe_skin_screen_boundary import convex_regions, measure
from probe_skin_screen_context import BASE, MODEL_SHA, ROOT, SCREEN, digest


INTAKE_SHA = "13cd9d6b043666959340e4f479b6297dc0ddcd91da0b7b31ae92b1d78cee1a01"
REVIEW_SHA = "4ee7cbb47df7497a6ff80a2f7f7cd5914ce59cfcf88f508fa068fc14b0ffc570"
DIRECT = {"463618": [389, 105, 414, 121], "345252": [409, 123, 425, 143]}


def inputs():
    source = ROOT / "docs/research/2026-11-19-screen-occlusion-intake-evidence.json"
    annotations = Path(__file__).with_name("screen-occlusion-review.json")
    if (digest(source), digest(annotations)) != (INTAKE_SHA, REVIEW_SHA):
        raise ValueError("Frozen intake or annotations changed")
    registered = {key: value["screen_skin_xyxy"] for key, value in json.loads(annotations.read_bytes()).items()
                  if "screen_skin_xyxy" in value}
    rows = {str(x["id"]): x for x in json.loads(source.read_bytes())["photos"]}
    for key, (l, t, r, b) in dict(registered, **DIRECT).items():
        path = BASE / "screen-occlusion-intake" / f"{int(key):012d}.jpg"
        if digest(path) != rows[key]["sha256"]:
            raise ValueError(f"Image changed: {key}")
        with Image.open(path) as frame:
            width, height = frame.size
            rgb = np.asarray(frame.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS))
        yield key, rgb, (math.floor(l*256/width), math.floor(t*256/height),
                         math.ceil(r*256/width), math.ceil(b*256/height)), key in registered


def skin_masks(output):
    from probe_bg_real_legacy_skin import protector

    skin = protector(ROOT / "experimental/skin-protection-20261019/reference_color_app/skin.py")
    for key, rgb, _, _ in inputs():
        np.save(output / f"{key}-weight.npy", skin._predict(rgb)[0], allow_pickle=False)


def run(runtime, output):
    import torch
    from torch.nn import functional as F
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    assets = ROOT / ".tools/scene-segformer"
    manifest = json.loads((assets / "manifest.json").read_bytes())
    if digest(assets / "model.safetensors") != MODEL_SHA or any(
            digest(assets / v["path"]) != v["sha256"] for v in manifest["files"]):
        raise ValueError("Frozen scene segmentation assets changed")
    torch.set_num_threads(4)
    model = SegformerForSemanticSegmentation.from_pretrained(assets, local_files_only=True, use_safetensors=True).eval()
    processor = SegformerImageProcessor.from_pretrained(assets, local_files_only=True)
    ids = [k for k, v in model.config.id2label.items() if v in SCREEN]
    output.mkdir()
    launcher = ("import sys,runpy,pathlib; sys.argv=sys.argv[1:]; "
                "sys.path.insert(0,str(pathlib.Path(sys.argv[0]).resolve().parent)); "
                "runpy.run_path(sys.argv[0],run_name='__main__')")
    subprocess.run([str(runtime.resolve()), "-c", launcher, str(Path(__file__).resolve()), "--worker", str(output)], check=True)
    results = {}
    for key, rgb, (l, t, r, b), screen in inputs():
        with torch.inference_mode():
            logits = model(**processor(images=rgb, return_tensors="pt")).logits
            prob, labels = F.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False).softmax(1)[0].max(0)
        labels = labels.numpy()
        raw = np.isin(labels, ids) & (prob.numpy() >= .70)
        hull, _, _ = convex_regions(raw)
        weight = np.load(output / f"{key}-weight.npy", allow_pickle=False)
        masks = {"skin": np.zeros_like(raw), "pixel": raw, "boundary": hull}
        crop = np.s_[t:b, l:r]
        category, totals = np.unique(labels[crop], return_counts=True)
        panel = Image.new("RGB", (1064, 291), "#20252c")
        draw = ImageDraw.Draw(panel)
        panel.paste(Image.fromarray(rgb), (8, 26))
        draw.rectangle((8+l, 26+t, 7+r, 25+b), outline="#ffff60")
        scores = {}
        for col, (arm, veto) in enumerate(masks.items(), start=1):
            surviving = weight * ~veto
            image = np.rint(np.clip(rgb + surviving[..., None] * [-8, 0, 12], 0, 255)).astype(np.uint8)
            panel.paste(Image.fromarray(image), (8+264*col, 26))
            draw.text((8+264*col, 6), arm, fill="white")
            scores[arm] = {**measure(surviving, veto, crop, screen),
                           "rgb_mae": float(np.abs(image[crop].astype(float)-rgb[crop]).mean())}
        path = output / f"{key}.png"
        panel.save(path)
        results[key] = {"screen_face": screen, "region": [l, t, r, b], "scores": scores,
                        "raw_area": int(raw.sum()), "hull_area": int(hull.sum()),
                        "labels": {model.config.id2label[int(k)]: int(v) for k, v in zip(category, totals, strict=True)},
                        "panel_sha256": digest(path)}
    report = {"cases": results, "intake_sha256": INTAKE_SHA, "review_sha256": REVIEW_SHA, "model_sha256": MODEL_SHA,
              "source_sha256": digest(Path(__file__)),
              "protocol_sha256": digest(Path(__file__).with_name("screen-face-transfer-protocol.md"))}
    (output / "results.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--worker", type=Path)
    cli.add_argument("--portable", type=Path)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    if args.worker:
        skin_masks(args.worker)
    elif args.portable and args.output:
        run(args.portable, args.output)
    else:
        cli.error("--portable and --output are required")
