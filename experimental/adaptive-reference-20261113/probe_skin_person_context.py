"""Test whole-image person support and surrounding-body context as skin gates."""
import argparse
from pathlib import Path
import re

import numpy as np
from scipy import ndimage

from person_segmenter import CHECKPOINT_SHA, ROOT, PersonSegmenter
from probe_dast_local_edit_cross import INPUT_SHA
from probe_screen_face_transfer import inputs as screen_inputs
from probe_skin_spectral_crop_gate import INPUTS, PARENT
from probe_skin_spectral_preview_gate import verified, write_json
from probe_skin_spectral_scene_gate import REGIONS, STATES, image, ledger, save

BASELINE = "21806a6f58f0ab59a7f1f0f472e9d6c719b516426f88515274578501ce585486"
CROP_MASKS = "f0baff9fcc99928a33a24e5b62594751fb1612422208ea505d7a5ecb509f57af"
EXTRA = {
    "340894": "7be9d74f0bf7805f01d177a0b07dde66f766e3b7589554773ae8211cb7f29676",
    "342367": "26281eb7e3e393c9bf65d3ed403736ed8b76daabe1f689809626602a25db0cf0",
    "345252": "b1a5c60ff9cc358cf00376dfe8f07c9f0b8a1a8518728275abedbd181856d8a3",
    "463618": "77e388607bd3999b548317a22d7ec0e929edc0876fcbe1f2322c87f2c94c28fe",
    "479248": "be4d455776863f6ea2744536286b70ee4fbb0cfba5f55fdb13ae685b2679b63d",
}
NEIGHBORS = np.ones((3, 3), dtype=bool)


def supports(face, person):
    body_ids, count = ndimage.label(person, structure=NEIGHBORS)
    face_ids, _ = ndimage.label(face, structure=NEIGHBORS)
    context = np.zeros_like(person)
    for idx, (ys, xs) in enumerate(ndimage.find_objects(face_ids), start=1):
        fragment = face_ids[ys, xs] == idx
        labels = body_ids[ys, xs][fragment]
        sizes = np.bincount(labels, minlength=count + 1)
        sizes[0] = 0
        selected = sizes.argmax()
        if sizes[selected] == 0:
            continue
        h, w = ys.stop - ys.start, xs.stop - xs.start
        below = body_ids[ys.stop:ys.stop + 2*h, max(0, xs.start-w):xs.stop+w]
        if np.count_nonzero(below == selected) >= fragment.sum():
            context[ys, xs] |= fragment & (body_ids[ys, xs] == selected)
    return {"person": face & person, "body_context": context}


def score(out, stem, name, original, adjusted, gate, areas):
    final = np.where(gate[..., None], adjusted, original)
    delta = np.abs(final.astype(np.int16) - original)
    return {"sha256": save(out, f"{stem}-{name}", final), "selected": int(gate.sum()),
            "areas": {key: {"fraction": float(gate[mask].mean()), "rgb_mae": float(delta[mask].mean()),
                            "changed": int(np.any(delta[mask] != 0, axis=-1).sum())}
                      for key, mask in areas.items()}}


def run(base, out):
    import torch
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

    torch.set_num_threads(4)
    weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    verified(ROOT / ".tools/torch-cache/segmentation" / weights.url.rsplit("/", 1)[1], CHECKPOINT_SHA)
    model = PersonSegmenter("cpu")
    frozen = base / "skin-spectral-crop-gate-v1"
    frozen_mask = base / "skin-spectral-crop-masks-v1"
    prior = base / "skin-spectral-scene-gate-v1"
    source = base / "skin-spectral-scene-inputs-v1"
    previous = ledger(frozen / "results.json", BASELINE)["cases"]
    full = ledger(prior / "results.json", PARENT)["cases"]
    masks = ledger(frozen_mask / "masks.json", CROP_MASKS)["images"]
    dast = base / "dast-exposure-prepared-v1"
    cores = {f"dast-{row['group']}": image(dast / str(row["group"]) / "nominal-mask.png",
                                           row["target_mask_sha256"], "L") > 0
             for row in ledger(dast / "inputs.json", INPUT_SHA)["groups"]}
    path = verified(Path(__file__).with_name("bg-real-skin-regions-protocol.md"), REGIONS)
    areas_by_key = {}
    for a, b, name, x, y, xx, yy in re.findall(r"^\| (\d+)–(\d+) \| ([^|]+?) \| (\d+),(\d+),(\d+),(\d+) \|$",
                                               path.read_text(encoding="utf-8"), re.M):
        area = np.zeros((256, 256), dtype=bool)
        area[int(y):int(yy), int(x):int(xx)] = True
        areas_by_key.setdefault(f"{a}-{b}", {})[name] = area
    out.mkdir()
    result = {}
    for key, hashes in ledger(source / "inputs.json", INPUTS).items():
        areas = {"archived_core": cores[key]} if key in cores else areas_by_key[key]
        for state in STATES:
            stem = f"{key}-{state}"
            rgb = image(source / f"{stem}.png", hashes[state])
            initial = image(frozen_mask / f"{stem}.png", masks[key][state], "L") == 255
            ordinary = image(prior / f"{stem}-matrix.png", full[key]["states"][state]["baseline_sha256"])
            adjusted = image(frozen / f"{stem}-auto.png", previous[key]["states"][state]["sha256"])
            person = model.predict(rgb)[0] >= .50
            arms = {"crop": initial, **supports(initial, person)}
            result[stem] = {name: score(out, stem, name, ordinary, adjusted, gate, areas)
                            for name, gate in arms.items()}

    external = {}
    for key, rgb, (left, top, right, bottom), screen in screen_inputs():
        weight = np.load(verified(base / "screen-face-transfer" / f"{key}-weight.npy", EXTRA[key]), allow_pickle=False)
        initial = weight > .05
        gate = supports(initial, model.predict(rgb)[0] >= .50)
        area = np.zeros(rgb.shape[:2], dtype=bool)
        area[top:bottom, left:right] = True
        rows = {}
        scoring_support = initial if screen else weight > .50
        for name, mask in {"legacy": np.ones_like(initial), **gate}.items():
            candidate = weight * mask
            output = np.rint(np.clip(rgb + candidate[..., None] * [-8, 0, 12], 0, 255)).astype(np.uint8)
            rows[name] = score(out, key, name, rgb, output, mask & scoring_support, {"face": area})
        external[key] = {"screen": screen, "arms": rows}
    write_json(out / "results.json", {"model_sha256": CHECKPOINT_SHA, "input_sha256": INPUTS,
                                     "parent_sha256": BASELINE, "scene": result, "external": external})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
