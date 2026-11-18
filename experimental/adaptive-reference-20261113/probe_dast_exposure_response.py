"""Test portrait exposure color and transfer while grouping by capture metadata."""
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from audit_dast_example import SHA256, capture_key

STATES = ("nominal", "repeat", "ue1", "ue2", "oe1", "oe2")
LUMA = np.array([.299, .587, .114])
SIZE = (320, 416)


def digest(path):
    from hashlib import sha256
    return sha256(path.read_bytes()).hexdigest()


def save_report(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_inputs(archive):
    if digest(archive) != SHA256:
        raise ValueError("DAST public archive changed")
    groups = defaultdict(lambda: defaultdict(list))
    with ZipFile(archive) as source:
        paths = {capture_key(name): name for name in source.namelist()
                 if name.startswith("example-data/") and name.lower().endswith(".jpg")}
        ledger = source.read("example-data/example_overview.csv").decode("utf-8-sig")
        for row in csv.reader(io.StringIO(ledger), delimiter=";"):
            if row[8] and row[9]:
                key = capture_key(row[3])
                groups[int(row[0]), row[8]][row[9]].append(key)
        selected = []
        for (person, _), exposures in sorted(groups.items()):
            if not {"optimal", *STATES[2:]} <= exposures.keys():
                continue
            nominal, repeat = sorted(exposures["optimal"])
            keys = (nominal, repeat, *(sorted(exposures[name])[0] for name in STATES[2:]))
            pictures = {}
            for name, key in zip(STATES, keys, strict=True):
                with Image.open(io.BytesIO(source.read(paths[key]))) as image:
                    pictures[name] = np.array(image.convert("RGB").resize(SIZE, Image.Resampling.BILINEAR))
            selected.append((person, pictures))
    if len(selected) != 5 or len({person for person, _ in selected}) != 3:
        raise ValueError("Expected five complete groups")
    return selected


def midtone(rgb):
    light = rgb @ LUMA
    return (rgb.min(axis=2) > 5) & (rgb.max(axis=2) < 250) & (light >= 40) & (light <= 215)


def statistics(before, after, region):
    count = int(region.sum())
    if count < 1024:
        return {"pixels": count, "qualified": False}
    a, b = before[region].astype(np.float64), after[region].astype(np.float64)
    def log_chroma(pixels):
        x = pixels / 255.
        linear = np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
        return np.log(linear[:, [0, 2]] / linear[:, 1:2])
    diff = log_chroma(b) - log_chroma(a)
    vector = diff.mean(axis=0)
    return {"pixels": count, "qualified": True, "luma_delta": float(((b - a) @ LUMA).mean()),
            "mean_log_chroma_change": vector.tolist(), "chroma_norm": float(np.linalg.norm(vector)),
            "mean_absolute_log_chroma_change": float(np.abs(diff).mean())}


def registered(before, after, first_mask, second_mask, cv):
    matrix = np.eye(2, 3, dtype=np.float32)
    gray = lambda image: cv.equalizeHist(cv.cvtColor(image, cv.COLOR_RGB2GRAY))
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 120, 1e-6)
    try:
        corr, matrix = cv.findTransformECC(gray(before), gray(after), matrix, cv.MOTION_AFFINE, criteria, None, 5)
    except cv.error:
        return {"alignment": "ecc_failed", "qualified": False}
    scale = np.linalg.svd(matrix[:, :2], compute_uv=False)
    admissible = (corr >= .75 and scale.min() >= .88 and scale.max() <= 1.12
                  and np.linalg.det(matrix[:, :2]) > 0 and np.all(np.abs(matrix[:, 2]) <= .1 * np.array(SIZE)))
    detail = {"alignment": "qualified" if admissible else "outside_alignment_limits",
              "correlation": float(corr), "qualified": False}
    if admissible:
        flags = cv.WARP_INVERSE_MAP
        moved = cv.warpAffine(after, matrix, SIZE, flags=cv.INTER_LINEAR | flags)
        mask = cv.warpAffine(second_mask.astype(np.uint8), matrix, SIZE, flags=cv.INTER_NEAREST | flags) > 0
        detail.update(statistics(before, moved, first_mask & mask & midtone(before) & midtone(moved)))
    return detail


def prepare(archive, out):
    import cv2
    from segment import Segmenter
    cv2.setNumThreads(1)
    cases = load_inputs(archive)
    out.mkdir()
    segmenter = Segmenter()
    data = []
    try:
        for index, (person, inputs) in enumerate(cases):
            dest = out / str(index + 1)
            dest.mkdir()
            target = next((index + n) % len(cases) + 1 for n in range(1, len(cases))
                          if cases[(index + n) % len(cases)][0] != person)
            masks, hashes = {}, {}
            for name, image in inputs.items():
                probability, provenance = segmenter.predict(image)
                binary = (probability[..., 3] >= .70).astype(np.uint8)
                masks[name] = cv2.erode(binary, np.ones((11, 11), np.uint8),
                                         borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
                path = dest / f"{name}.png"
                Image.fromarray(image).save(path)
                hashes[name] = digest(path)
            mask_path = dest / "nominal-mask.png"
            Image.fromarray(masks["nominal"].astype(np.uint8) * 255).save(mask_path)
            comparisons = {name: registered(inputs["nominal"], inputs[name], masks["nominal"], masks[name], cv2)
                           for name in STATES[1:]}
            data.append({"group": index + 1, "target_group": target, "source": comparisons,
                         "frames": hashes, "target_mask_sha256": digest(mask_path)})
    finally:
        segmenter.close()
    save_report(out / "inputs.json", {"archive_sha256": SHA256, "opencv": cv2.__version__,
                                     "locator_sha256": provenance["model_sha256"], "groups": data})


def verified_png(path, expected):
    if digest(path) != expected:
        raise ValueError("Prepared input changed")
    with Image.open(path) as image:
        return np.array(image)


def render(prepared, out):
    import torch
    from spatial_render import SpatialEngine, finish_spatial
    torch.set_num_threads(4)
    parent = prepared / "inputs.json"
    plan = json.loads(parent.read_bytes())
    if plan["archive_sha256"] != SHA256 or len(plan["groups"]) != 5:
        raise ValueError("Unexpected prepared cohort")
    out.mkdir()
    engine = SpatialEngine("multiref-v1")
    results = []
    for row in plan["groups"]:
        donor = row["group"]
        target = plan["groups"][row["target_group"] - 1]
        folder = prepared / str(target["group"])
        original = verified_png(folder / "nominal.png", target["frames"]["nominal"])
        mask = verified_png(folder / "nominal-mask.png", target["target_mask_sha256"]) > 0
        generated, hashes = {}, {}
        dest = out / str(donor)
        dest.mkdir()
        for state in STATES:
            ref = verified_png(prepared / str(donor) / f"{state}.png", row["frames"][state])
            generated[state] = finish_spatial(original, engine.prepare(original, ref).apply(original), .8)
            png = dest / f"{state}.png"
            Image.fromarray(generated[state]).save(png)
            hashes[state] = digest(png)
        base = generated["nominal"]
        comparisons = {}
        for state in STATES[1:]:
            region = mask & midtone(original) & midtone(base) & midtone(generated[state])
            score = statistics(base, generated[state], region)
            source = row["source"][state]
            score["direction_projection"] = None
            if source["qualified"] and score["qualified"] and source["chroma_norm"] > 0:
                axis = np.asarray(source["mean_log_chroma_change"])
                score["direction_projection"] = float(axis @ score["mean_log_chroma_change"] / (axis @ axis))
            comparisons[state] = score
        repeat = comparisons["repeat"]
        qualifying = [
            state for state in STATES[2:] if repeat["qualified"] and row["source"]["repeat"]["qualified"]
            and row["source"][state]["qualified"] and row["source"][state]["chroma_norm"] >= .03
            and comparisons[state]["qualified"] and comparisons[state]["chroma_norm"] >= .03
            and comparisons[state]["direction_projection"] >= .25
            and comparisons[state]["chroma_norm"] > repeat["chroma_norm"]
        ]
        results.append({"group": donor, "source": row["source"], "target": comparisons,
                        "qualifying_exposures": qualifying, "output_sha256": hashes})
    count = sum(len(row["qualifying_exposures"]) >= 2 for row in results)
    save_report(out / "results.json", {"source_sha256": SHA256, "inputs_sha256": digest(parent),
                "weight_sha256": engine.checkpoint_sha, "groups_passing": count,
                "supports_exposure_chroma_propagation": count >= 3, "groups": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "render"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else render)(args.source, args.output)
