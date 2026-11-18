"""Test fixed screen semantics as a veto on an existing skin locator."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from audit_bg_real_skin_regions import BASE, PROTOCOL, ROOT, digest, read


MODEL_SHA = "6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9"
SOURCE_SHA = "83be16a13691b50cd3449375df656b0cedabdbf3b31e63639b43d6e42f3c68d9"
REGION_SHA = "2e0fe1e6b0b47e3579263bbd7cee626713955fee6c3cde71ef8323784de702b1"
SCREEN = {"screen", "television receiver", "crt screen", "monitor"}
NEGATIVE = {"雪地", "浅墙", "白球服", "投影幕中的脸"}
BODY = {"手背", "前景人物手臂"}


def sources():
    manifest = ROOT / "docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json"
    if digest(manifest) != SOURCE_SHA or digest(PROTOCOL) != REGION_SHA:
        raise ValueError("Frozen inputs changed")
    folder = BASE / "background-intervention-20261116-v2"
    parent = folder / "results.json"
    metadata = json.loads(manifest.read_bytes())
    if digest(parent) != metadata["parent_sha256"]:
        raise ValueError("Frozen source index changed")
    index = {r["pair"]: r["input_file"] for r in json.loads(parent.read_bytes())["cases"]}
    for item in metadata["cases"].values():
        pair = item["target_pair"]
        entry = index[pair]
        yield pair, read(folder / entry["path"], entry["sha256"]).astype(np.uint8)


def export_masks(folder):
    from probe_bg_real_legacy_skin import protector

    model = protector(ROOT / "experimental/skin-protection-20261019/reference_color_app/skin.py")
    for pair, rgb in sources():
        np.save(folder / f"{pair}.npy", model._predict(rgb)[0], allow_pickle=False)


def evaluate(portable, output):
    import torch
    from torch.nn import functional as F
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    assets = ROOT / ".tools/scene-segformer"
    manifest = json.loads((assets / "manifest.json").read_bytes())
    if digest(assets / "model.safetensors") != MODEL_SHA or any(
        digest(assets / entry["path"]) != entry["sha256"] for entry in manifest["files"]
    ):
        raise ValueError("Frozen segmentation assets changed")
    torch.set_num_threads(4)
    processor = SegformerImageProcessor.from_pretrained(assets, local_files_only=True)
    model, info = SegformerForSemanticSegmentation.from_pretrained(
        assets, local_files_only=True, use_safetensors=True, output_loading_info=True,
    )
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Incomplete model restoration")
    model.eval()
    keys = [i for i, name in model.config.id2label.items() if name in SCREEN]
    if len(keys) != len(SCREEN):
        raise ValueError("Unexpected screen taxonomy")
    output.mkdir()
    masks = output / "local-masks"
    masks.mkdir()
    loader = ("import sys,runpy,pathlib; sys.argv=sys.argv[1:]; "
              "sys.path.insert(0,str(pathlib.Path(sys.argv[0]).resolve().parent)); "
              "runpy.run_path(sys.argv[0],run_name='__main__')")
    subprocess.run([str(portable.resolve()), "-c", loader, str(Path(__file__).resolve()), "--worker", str(masks)], check=True)
    boxes = re.findall(r'^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$',
                       PROTOCOL.read_text(encoding="utf-8"), re.MULTILINE)
    assert len(boxes) == 9
    table = {}
    for pair, rgb in sources():
        with torch.inference_mode():
            logits = model(**processor(images=rgb, return_tensors="pt")).logits
            posterior = F.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False).softmax(1)[0]
            confidence, label = posterior.max(0)
            categories = label.numpy()
            screen_prob = posterior[keys].sum(0).numpy()
        veto = np.isin(categories, keys) & (confidence.numpy() >= .70)
        semantic = np.load(masks / f"{pair}.npy", allow_pickle=False)
        values = {"semantic": semantic, "screen_veto": semantic * ~veto}
        edits = {arm: np.rint(np.clip(rgb + weight[..., None] * [-8, 0, 12], 0, 255)).astype(np.uint8)
                 for arm, weight in values.items()}
        regions = {}
        for a, b, label_name, l, t, r, y in boxes:
            if f"{a}-{b}" != pair:
                continue
            crop = np.s_[int(t):int(y), int(l):int(r)]
            ids, counts = np.unique(categories[crop], return_counts=True)
            row = {"screen_veto_pixels": int(veto[crop].sum()), "screen_posterior_mean": float(screen_prob[crop].mean()),
                   "labels": {model.config.id2label[int(i)]: int(n) for i, n in zip(ids, counts, strict=True)}}
            for arm, weight in values.items():
                active, touched = (float((weight[crop] > x).mean()) for x in (.5, .05))
                passed = (None if label_name in BODY else touched <= .01 if label_name in NEGATIVE else active >= .8)
                row[arm] = {"active": active, "touched": touched, "passes": passed,
                            "rgb_mae": float(np.abs(edits[arm][crop].astype(float)-rgb[crop]).mean())}
            regions[label_name] = row
        panel = Image.new("RGB", (800, 291), "#20252c")
        draw = ImageDraw.Draw(panel)
        for col, (arm, frame) in enumerate({"source": rgb, **edits}.items()):
            x = 8 + col * 264
            draw.text((x, 6), arm, fill="white")
            panel.paste(Image.fromarray(frame), (x, 26))
        path = output / f"{pair}.png"
        panel.save(path)
        table[pair] = {"regions": regions, "veto_pixels": int(veto.sum()), "panel_sha256": digest(path)}
    report = {"cases": table, "model_sha256": MODEL_SHA, "source_sha256": digest(Path(__file__)),
              "protocol_sha256": digest(Path(__file__).with_name("skin-screen-context-protocol.md"))}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--portable", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        export_masks(args.worker)
    elif args.portable and args.output:
        evaluate(args.portable, args.output)
    else:
        parser.error("--portable and --output are required")
