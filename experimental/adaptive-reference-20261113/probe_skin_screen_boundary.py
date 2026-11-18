"""Compare pixel screen labels with a fixed connected-component boundary rule."""

import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from probe_skin_screen_context import BASE, MODEL_SHA, NEGATIVE, PROTOCOL, ROOT, SCREEN, digest, sources


PRIOR_SHA = "2e21b44ce816e04e11402d9b902727efd4ca41f37c3b5d40e65c0bd252cf73a4"


def composite(images):
    index_path = BASE / "background-intervention-20261116-v2/results.json"
    index = {v["pair"]: v for v in json.loads(index_path.read_bytes())["cases"]}
    name = "9483-human-mask.png"
    atom = BASE / "independent-coco" / name
    if digest(atom) != index["9483-19432"]["parents"][name]:
        raise ValueError("Foreground mask changed")
    with Image.open(atom) as tm:
        alpha = np.asarray(tm.convert("L").resize((256, 256), Image.Resampling.NEAREST)) > 0
    if not np.all(alpha[39:45, 42:46]):
        raise ValueError("Foreground forehead not fully included")
    subject = np.s_[24:105, 25:73]
    image = images["8021-8690"].copy()
    place = image[60:141, 108:156]
    place[alpha[subject]] = images["9483-19432"][subject][alpha[subject]]
    return image


def convex_regions(screen):
    from scipy.ndimage import label, find_objects
    from scipy.spatial import ConvexHull, QhullError

    zones, num = label(screen, structure=np.ones((3, 3), dtype=int))
    canvas = Image.fromarray(screen)
    pen = ImageDraw.Draw(canvas)
    expanded = 0
    for component, bounds in enumerate(find_objects(zones), start=1):
        if bounds is None:
            continue
        y, x = np.nonzero(zones[bounds] == component)
        if len(x) < 64:
            continue
        xy = np.column_stack((x + bounds[1].start, y + bounds[0].start))
        try:
            outer = ConvexHull(xy)
        except QhullError:
            continue
        pen.polygon([tuple(p) for p in xy[outer.vertices].tolist()], fill=1)
        expanded += 1
    return np.asarray(canvas, dtype=bool), int(num), expanded


def measure(weight, veto, crop, negative):
    cut = weight[crop]
    active, touch = (float((cut > limit).mean()) for limit in (.5, .05))
    return {"active": active, "touch": touch, "passes": touch <= .01 if negative else active >= .8,
            "veto_pixels": int(veto[crop].sum())}


def evaluate(portable, output):
    import torch
    from torch.nn import functional as F
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    parent = ROOT / "docs/research/2026-11-19-skin-screen-context-evidence.json"
    if digest(parent) != PRIOR_SHA:
        raise ValueError("Previous screen evidence changed")
    old = json.loads(parent.read_bytes())["cases"]
    assets = ROOT / ".tools/scene-segformer"
    manifest = json.loads((assets / "manifest.json").read_bytes())
    if digest(assets / "model.safetensors") != MODEL_SHA or any(
            digest(assets / v["path"]) != v["sha256"] for v in manifest["files"]):
        raise ValueError("Screen model assets changed")
    torch.set_num_threads(4)
    model = SegformerForSemanticSegmentation.from_pretrained(assets, local_files_only=True, use_safetensors=True).eval()
    processor = SegformerImageProcessor.from_pretrained(assets, local_files_only=True)
    screen_ids = [k for k, v in model.config.id2label.items() if v in SCREEN]
    images = dict(sources())
    images["foreground"] = composite(images)
    output.mkdir()
    path = output / "composite-input.png"
    Image.fromarray(images["foreground"]).save(path)
    loader = ("import sys,runpy,pathlib; sys.argv=sys.argv[1:]; "
              "sys.path.insert(0,str(pathlib.Path(sys.argv[0]).resolve().parent)); "
              "runpy.run_path(sys.argv[0],run_name='__main__')")
    subprocess.run([str(portable.resolve()), "-c", loader, str(Path(__file__).resolve()), "--worker", str(output)], check=True)
    boxes = re.findall(r'^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$',
                       PROTOCOL.read_text(encoding="utf-8"), re.MULTILINE)
    results = {}
    for name, rgb in images.items():
        with torch.inference_mode():
            logits = model(**processor(images=rgb, return_tensors="pt")).logits
            label_conf, categories = F.interpolate(logits, size=rgb.shape[:2], mode="bilinear",
                                                    align_corners=False).softmax(1)[0].max(0)
        veto = np.isin(categories.numpy(), screen_ids) & (label_conf.numpy() >= .70)
        if name in old and int(veto.sum()) != old[name]["veto_pixels"]:
            raise ValueError("The original display segmentation did not reproduce")
        hull, components, used = convex_regions(veto)
        skin = np.load(output / f"{name}-skin.npy", allow_pickle=False)
        zero = np.zeros_like(veto)
        variants = {"semantic": (skin, zero), "pixel": (skin * ~veto, veto), "boundary": (skin * ~hull, hull)}
        areas = [(label, np.s_[int(t):int(b), int(l):int(r)]) for a, z, label, l, t, r, b in boxes
                 if f"{a}-{z}" == name and label not in {"手背", "前景人物手臂"}]
        if name == "foreground":
            areas = [("foreground forehead", np.s_[75:81, 125:129])]
        records = {area: {arm: measure(weight, masked, crop, area in NEGATIVE)
                          for arm, (weight, masked) in variants.items()} for area, crop in areas}
        frames = {"source": rgb, **{arm: np.rint(np.clip(rgb + w[..., None] * [-8, 0, 12], 0, 255)).astype(np.uint8)
                                   for arm, (w, _) in variants.items()}}
        panel = Image.new("RGB", (1064, 291), "#20252c")
        draw = ImageDraw.Draw(panel)
        for col, (arm, frame) in enumerate(frames.items()):
            x = 8 + col * 264
            draw.text((x, 6), arm, fill="white")
            panel.paste(Image.fromarray(frame), (x, 26))
        file = output / f"{name}.png"
        panel.save(file)
        np.savez_compressed(output / f"{name}-display.npz", pixel=veto, boundary=hull)
        results[name] = {"regions": records, "components": components, "hulls": used, "pixel_area": int(veto.sum()),
                         "boundary_area": int(hull.sum()), "panel_sha256": digest(file)}
    qualification = results["foreground"]["regions"]["foreground forehead"]["semantic"]["passes"]
    passed = qualification and all(v["boundary"]["passes"] for i in results.values() for v in i["regions"].values())
    report = {"cases": results, "foreground_qualified": qualification, "passed": passed, "prior_sha256": PRIOR_SHA,
              "source_sha256": digest(Path(__file__)),
              "protocol_sha256": digest(Path(__file__).with_name("skin-screen-boundary-protocol.md"))}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def worker(output):
    from probe_bg_real_legacy_skin import protector

    skin = protector(ROOT / "experimental/skin-protection-20261019/reference_color_app/skin.py")
    images = dict(sources())
    with Image.open(output / "composite-input.png") as image:
        images["foreground"] = np.asarray(image.convert("RGB"))
    for key, frame in images.items():
        np.save(output / f"{key}-skin.npy", skin._predict(frame)[0], allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--portable", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.portable and args.output:
        evaluate(args.portable, args.output)
    else:
        parser.error("--portable and --output are required")
