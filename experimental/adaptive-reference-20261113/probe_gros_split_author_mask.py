"""Isolate the effect of SPLiT's published face masks on lighting decomposition."""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch

from probe_gros_film_b_cross import GROUPS, LIGHTS
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_gros_split_face_albedo import DEV, mask_and_feather
from probe_gros_split_light_readout import COMPONENTS, STATES, assess, crop_face, decompose, load_models
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel

PRIOR = "d6cbed78adb806dde06d284daee2488fc9b372fcd634bcaeff2e6da29395d443"
SOURCE_SHA = "d6b6fd8fce7babee37d8dbf1c9f7aeec4e395f54030ff09a7b1b6e4e69f20da3"
WEIGHT_SHA = "ab06dedaea0d5429ebc539cb899688b627a4e3c040eca9dbd8b733843766852b"
FACE_CLASSES = (1, 2, 3, 6, 7, 11, 12)
UPPER_ERROR_LIMIT = 5


def load_parser(root):
    if digest(root / "models/deeplab.py") != SOURCE_SHA:
        raise ValueError("Changed face parser source")
    from models.deeplab import resnet101

    model = resnet101(pretrained=False, num_classes=19, num_groups=32, weight_std=True, beta=False)
    file = root / "deeplab_model.pth"
    if digest(file) != WEIGHT_SHA:
        raise ValueError("Changed face parser weight")
    checkpoint = torch.load(file, weights_only=True, map_location="cpu")
    weights = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()
               if "tracked" not in key}
    model.load_state_dict(weights, strict=True)
    return model.to(DEV).eval(), len(weights)


@torch.inference_mode()
def author_mask(parser, crop):
    image = torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].to(DEV, torch.float32) / 255
    mean = image.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = image.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    classes = parser((image - mean) / std).argmax(dim=1)[0].cpu().numpy()
    raw = (np.isin(classes, FACE_CLASSES) * 255).astype(np.uint8)
    seed = raw.copy()
    cv2.floodFill(seed, None, (0, 0), 255)
    holes = 255 - seed
    filled = raw + holes
    kernel_face = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    kernel_eye = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    eroded = cv2.erode(filled, kernel_face)
    dilated = cv2.dilate(holes, kernel_eye)
    return (eroded - dilated) == 255


def mean_error(value, mask):
    return float(value[mask].mean()) if np.any(mask) else None


def run(out):
    parent = BASE / "gros-film-b-cross-v2"
    manifest = checked_json(parent / "results.json", BASELINE_SHA)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    frozen = checked_json(BASE / "gros-split-light-readout-v1" / "results.json", PRIOR)
    root = BASE / "split-intake-v1"
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    first, second, tensors = load_models(root)
    parser, tensors["parser"] = load_parser(root)
    old_mask, _ = mask_and_feather()
    out.mkdir()
    cases = {}
    for person in GROUPS:
        frames = {}
        comparison = {}
        for light in LIGHTS:
            frames[light], comparison[light] = {}, {}
            key = f"{person}-{light}"
            for state in STATES:
                path = parent / f"{key}-{state}.png"
                portrait = checked_rgb(path, manifest["inputs"][key]["hashes"][state])
                crop = crop_face(portrait, labels[person][light]["eyes"])
                valid = author_mask(parser, crop)
                common = old_mask & valid
                if not common.any():
                    raise ValueError(f"No common face region {key}-{state}")
                old, old_image = decompose(first, second, crop, old_mask)
                previous_case = frozen["cases"][person]["frames"][light][state]
                if old != {k: v for k, v in previous_case.items() if k != "panel_sha"}:
                    raise ValueError("Original mask did not reproduce prior decomposition")
                current, current_image = decompose(first, second, crop, valid)
                frames[light][state] = current
                err_old = np.abs(crop.astype(np.int16) - old_image["reconstruction"])
                err_current = np.abs(crop.astype(np.int16) - current_image["reconstruction"])
                display_mask = lambda m: np.repeat((m * 255).astype(np.uint8)[..., None], 3, axis=2)
                board = panel(out / f"{key}-{state}-comparison.png", {"portrait": {
                    "input": crop, "old mask": display_mask(old_mask), "author mask": display_mask(valid),
                    "old reconstruction": old_image["reconstruction"],
                    "author reconstruction": current_image["reconstruction"],
                }})
                comparison[light][state] = dict(
                    previous_full=old["rgb8_recon_mae"], old_common=mean_error(err_old, common),
                    new_common=mean_error(err_current, common), old_removed=mean_error(err_old, old_mask & ~valid),
                    common_over_old=float(common.sum() / old_mask.sum()), mask_pixels=int(valid.sum()), board=board,
                )
        cases[person] = dict(score=assess(frames), frames=frames, comparison=comparison)
    shared_support = all(row["common_over_old"] >= .5 for case in cases.values()
                         for group in case["comparison"].values() for row in group.values())
    first_face_reconstructed = all(row["new_common"] <= UPPER_ERROR_LIMIT
                                   for group in cases["01"]["comparison"].values() for row in group.values())
    admitted = {key: shared_support and first_face_reconstructed
                and all(case["score"][key]["qualified"] for case in cases.values()) for key in COMPONENTS}
    json_to(out / "results.json", dict(previous=PRIOR, source=SOURCE_SHA, weights=WEIGHT_SHA, loaded=tensors,
                                       shared_support=shared_support, cases=cases, admitted=admitted,
                                       peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
