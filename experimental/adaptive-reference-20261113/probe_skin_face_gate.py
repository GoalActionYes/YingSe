"""Compare a fixed face constraint against the existing skin locator."""

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from audit_bg_real_skin_regions import BASE, PROTOCOL, ROOT, digest, read
from probe_bg_real_legacy_skin import protector


INPUT_SHA = "83be16a13691b50cd3449375df656b0cedabdbf3b31e63639b43d6e42f3c68d9"
REGION_SHA = "2e0fe1e6b0b47e3579263bbd7cee626713955fee6c3cde71ef8323784de702b1"
CASCADE_SHA = "0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0"
NEGATIVES = {"雪地", "浅墙", "白球服", "投影幕中的脸"}
BODY = {"手背", "前景人物手臂"}


def run(cascade, output):
    manifest = ROOT / "docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json"
    if (digest(manifest), digest(PROTOCOL), digest(cascade)) != (INPUT_SHA, REGION_SHA, CASCADE_SHA):
        raise ValueError("Unexpected model or evaluation assets")
    source = json.loads(manifest.read_bytes())
    folder = BASE / "background-intervention-20261116-v2"
    parent = folder / "results.json"
    if digest(parent) != source["parent_sha256"]:
        raise ValueError("Source images changed")
    index = {row["pair"]: row["input_file"] for row in json.loads(parent.read_bytes())["cases"]}
    boxes = re.findall(r'^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$',
                       PROTOCOL.read_text(encoding="utf-8"), re.MULTILINE)
    assert len(boxes) == 9
    skin = protector(ROOT / "experimental/skin-protection-20261019/reference_color_app/skin.py")
    # FileStorage opens the frozen model through a Unicode-safe in-memory buffer.
    storage = cv2.FileStorage(cascade.read_text(encoding="utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    detector = cv2.CascadeClassifier()
    if not detector.read(storage.getFirstTopLevelNode()) or detector.empty():
        raise ValueError("Face classifier did not load")
    results = {}
    output.mkdir()
    for case in source["cases"].values():
        pair = case["target_pair"]
        image = read(folder / index[pair]["path"], index[pair]["sha256"]).astype(np.uint8)
        semantic, gray, _ = skin._predict(image)
        face = np.zeros_like(semantic, dtype=bool)
        rectangles = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(16, 16))
        for x, y, w, h in rectangles:
            face[y:y+h, x:x+w] = True
        weights = {"semantic": semantic, "face_gated": semantic * face}
        images = {arm: np.rint(np.clip(image + mask[..., None] * [-8, 0, 12], 0, 255)).astype(np.uint8)
                  for arm, mask in weights.items()}
        values = {}
        sheet = Image.new("RGB", (800, 291), "#20252c")
        draw = ImageDraw.Draw(sheet)
        for col, (arm, frame) in enumerate({"source": image, **images}.items()):
            x = 8 + col * 264
            draw.text((x, 6), arm, fill="white")
            sheet.paste(Image.fromarray(frame), (x, 26))
        for left_id, right_id, label, left, top, right, bottom in boxes:
            if f"{left_id}-{right_id}" != pair:
                continue
            x, y, r, b = map(int, (left, top, right, bottom))
            patch = np.s_[y:b, x:r]
            category = "negative" if label in NEGATIVES else "body" if label in BODY else "face"
            values[label] = {"category": category}
            for arm, mask in weights.items():
                active = float((mask[patch] > .5).mean())
                touched = float((mask[patch] > .05).mean())
                passed = active >= .8 if category == "face" else touched <= .01 if category == "negative" else None
                values[label][arm] = {"active": active, "touched": touched, "passes": passed,
                                      "rgb_mae": float(np.abs(images[arm][patch].astype(float)-image[patch]).mean())}
            draw.rectangle((8+x, 26+y, 7+r, 25+b), outline="#ffff60")
        path = output / f"{pair}.png"
        sheet.save(path)
        results[pair] = {"rectangles": np.asarray(rectangles).tolist(), "regions": values, "panel_sha256": digest(path)}
    summary = {"cases": results, "cascade_sha256": CASCADE_SHA, "source_sha256": digest(Path(__file__)),
               "protocol_sha256": digest(Path(__file__).with_name("skin-face-gate-protocol.md"))}
    (output / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cascade", type=Path)
    cli.add_argument("output", type=Path)
    run(**vars(cli.parse_args()))
