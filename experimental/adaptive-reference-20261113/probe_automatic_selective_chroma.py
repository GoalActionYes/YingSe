"""Try a two-image, automatically gated spatial desaturation on frozen controls."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from person_segmenter import CHECKPOINT_SHA, ROOT, PersonSegmenter
from probe_arri_film_b_dose import verified
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_mls_film_light_cross import SOURCE_SHA
from probe_psbench_background_desaturation import CHECKPOINT, REC709, TASKS, luminance_chroma, load, panel, regions, score, sha
from spatial_render import SpatialEngine, finish_spatial


PARENT_SHA = "6dbe4870e44c81bd7614a779febb7c1806a51662d5e5d25c675e91280b4c540b"
BG_SHA = "11b72e0865bad7f3d20a4b3b9a8120c5e8df0de8fd2bbc78549625d67fa1d94d"
STRENGTH = .8


class Objects:
    def __init__(self):
        from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

        weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        cache = ROOT / ".tools/torch-cache/segmentation" / weights.url.rsplit("/", 1)[-1]
        if not cache.is_file() or sha(cache) != CHECKPOINT_SHA:
            raise ValueError("Expected the existing local segmentation checkpoint")
        self.detector = PersonSegmenter()
        self.background_label = weights.meta["categories"].index("__background__")
        self.results = {}

    @torch.inference_mode()
    def __call__(self, rgb):
        key = (rgb.shape, hashlib.sha256(rgb.tobytes()).digest())
        if key not in self.results:
            engine = self.detector
            tensor = engine.transform(Image.fromarray(rgb))[None].to(engine.device)
            logits = engine.model(tensor)["out"]
            background = torch.softmax(logits, 1)[:, self.background_label:self.background_label + 1]
            probability = 1 - F.interpolate(background, size=rgb.shape[:2], mode="bilinear",
                                             align_corners=False)[0, 0].cpu().numpy()
            if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
                raise ValueError("Invalid object probability")
            self.results[key] = probability
        return self.results[key]


def coverage(probability):
    foreground, background = probability >= .70, probability <= .30
    minimum = max(512, .02 * probability.size)
    return foreground, background, min(foreground.sum(), background.sum()) >= minimum


def render(target, baseline, probability):
    _, color = luminance_chroma(target)
    light = baseline @ REC709
    residual = color * (1 - STRENGTH * (1 - probability[..., None]))
    pole = light[..., None]
    upper = np.divide(255 - pole, residual, out=np.ones_like(residual), where=residual > 0)
    lower = np.divide(-pole, residual, out=np.ones_like(residual), where=residual < 0)
    scale = np.minimum(1, np.minimum(upper, lower).min(axis=2))
    pixels = np.rint(np.clip(pole + residual * scale[..., None], 0, 255)).astype(np.uint8)
    error = float(np.abs((pixels.astype(float) - baseline) @ REC709).max())
    if error > .5 + 1e-9:
        raise ValueError("Luminance preservation failed")
    return pixels, {"peak_luma_error": error, "limited_fraction": float((scale < 1 - 1e-9).mean())}


def candidate(target, reference, baseline, detect):
    if np.array_equal(target, reference):
        return target.copy(), {"applied": False, "identity": True}
    fg, bg, supported = coverage(detect(reference))
    _, ref_color = luminance_chroma(reference)
    amount = np.linalg.norm(ref_color, axis=2)
    neutral = float((amount[bg] <= 3).mean()) if bg.any() else None
    saturated = float((amount[fg] >= 12).mean()) if fg.any() else None
    report = {"reference_object_fraction": float(fg.mean()), "reference_background_fraction": float(bg.mean()),
              "neutral_background_fraction": neutral, "colorful_object_fraction": saturated, "applied": False}
    if supported and neutral >= .80 and saturated >= .20:
        probability = detect(target)
        if coverage(probability)[2]:
            output, metadata = render(target, baseline, probability)
            return output, report | metadata | {"applied": True}
    return baseline.copy(), report


def save(path, image):
    Image.fromarray(image).save(path)
    return sha(path)


def controls(base, pairs):
    bg_base = base / "bg-real-editor-look"
    bg = verified_json(bg_base / "results.json", BG_SHA)
    for name, case in bg["cases"].items():
        file = "C1-off-reference.png"
        yield f"bg-{name}", verified(bg_base / name / file, case["artifacts"][file])
    mls_base = base / "mls-linear-pilot"
    mls = verified_json(mls_base / "results.json", SOURCE_SHA)
    for name in ("02", "03"):
        file = f"{name}_2HAL_DESK.png"
        image = Image.fromarray(verified(mls_base / file, mls["scenes"][name]["inputs_sha256"][file]))
        image.thumbnail((512, 512), Image.Resampling.BILINEAR)
        yield f"room-{name}", np.array(image)
    for name, (original, _) in pairs.items():
        gray = np.rint(original @ REC709).astype(np.uint8)
        yield f"gray-{name}", np.repeat(gray[..., None], 3, axis=2)


def run(base, out):
    torch.set_num_threads(4)
    history = base / "psbench-background-desaturation-v1"
    frozen = verified_json(history / "results.json", PARENT_SHA)
    media = base / "psbench-background-desaturation-assets-v1"
    pairs = {task: [load(media, name, frozen["input_files"][name])[0]
                    for name in (f"{task}_start.jpg", f"{task}_end.jpg")] for task in TASKS}
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed model")
    detect = Objects()
    patch = np.empty((16, 16, 3), np.uint8)
    patch[:] = [250, 35, 20]
    low_light = np.full_like(patch, 2)
    boundary, limit = render(patch, low_light, np.ones(patch.shape[:2]))
    assert limit["limited_fraction"] == 1 and boundary.min() == 0
    out.mkdir()
    result = {}
    for task, donor in (TASKS, TASKS[::-1]):
        target = pairs[task][0]
        previous, outputs, audit = {}, {}, {}
        for state, guide in zip(("off", "on"), pairs[donor], strict=True):
            previous[state] = verified(history / f"{task}-{state}.png", frozen["results"][task]["hashes"][state])
            outputs[state], detail = candidate(target, guide, previous[state], detect)
            audit[state] = detail | {"sha256": save(out / f"{task}-{state}.png", outputs[state])}
        truth = pairs[task][1]
        areas = regions(target, truth)
        tested = score(target, truth, outputs["off"], outputs["on"], areas)
        old = frozen["results"][task]["regions"]
        error = {name: row["absolute_chroma_errors"]["on"] for name, row in tested["regions"].items()}
        improved = (error["desaturated"] < old["desaturated"]["absolute_chroma_errors"]["on"]
                    and error["retained"] <= old["retained"]["absolute_chroma_errors"]["on"] + 1.5)
        result[task] = tested | {"audit": audit, "absolute_gate": improved,
                                 "panel_sha256": panel(out / f"{task}-comparison.png",
                                                       {"Target": target, "Edited reference": pairs[donor][1],
                                                        "Author output": truth, "13 (edited reference)": previous["on"],
                                                        "Candidate (edited reference)": outputs["on"]}, areas)}

    negative = {}
    for index, (name, guide) in enumerate(controls(base, pairs)):
        target = pairs[TASKS[index % 2]][0]
        baseline = finish_spatial(target, engine.prepare(target, guide).apply(target), STRENGTH)
        output, detail = candidate(target, guide, baseline, detect)
        negative[name] = detail | {"pass": not detail["applied"] and np.array_equal(output, baseline),
                                   "baseline_sha256": save(out / f"{name}-13.png", baseline),
                                   "sha256": save(out / f"{name}.png", output)}
    for task, (target, _) in pairs.items():
        output, detail = candidate(target, target.copy(), np.zeros_like(target), detect)
        negative[f"identity-{task}"] = detail | {
            "pass": np.array_equal(target, output), "sha256": save(out / f"identity-{task}.png", output),
        }
    overlap = {}
    for task, (original, authored) in pairs.items():
        # The edited photo already served as the ordinary reference in the opposite direction.
        # Only this retrospective measurement reads the author-derived score regions.
        fg, bg, _ = coverage(detect(authored))
        overlap[task] = {
            role: {"confident_object": float(fg[area].mean()), "confident_background": float(bg[area].mean())}
            for role, area in regions(original, authored).items()
        }
    if len(detect.results) > 12 or len(negative) != 10:
        raise ValueError("Unexpected analysis scope")
    passed = (all(row["passed"] and row["absolute_gate"] for row in result.values())
              and all(row["pass"] for row in negative.values()))
    json_to(out / "results.json", {"source_sha256": PARENT_SHA, "weights": CHECKPOINT_SHA, "baseline": CHECKPOINT,
                                   "segmented_images": len(detect.results), "synthetic": limit,
                                   "cases": result, "negative": negative, "reference_score_region_overlap": overlap,
                                   "passed": passed})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
