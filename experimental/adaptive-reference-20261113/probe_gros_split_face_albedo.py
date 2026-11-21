"""Test portrait-specific diffuse color before applying any product changes."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from probe_gros_dinolight_reference import mae
from probe_gros_film_b_cross import GROUPS, LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_physical_blue_global_fit import display, linear
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)

WEIGHT = "caaa771310be6964d7cf01ca41d30cfefafad69c14719e320d51bbf0055d486a"
CODE = {
    "blurpool.py": "2dbb95d83b7a661e9fb78a9f63220a1180d4ac7571b02c3614446a66aeaed4d1",
    "intrinsic_networks.py": "7ef182f61ef751cc6c80a6622bc2c688d0e23bdaf44c0b863de38c75b29184a3",
}
WINDOW = 512
LUMA = np.array([.2126, .7152, .0722])
DEV = torch.device("cuda")


def get_model(root):
    source = root / "intrinsic_decom_f_AN.pth"
    if digest(source) != WEIGHT or any(digest(root / "models" / name) != sha for name, sha in CODE.items()):
        raise ValueError("Changed SPLiT weights or source")
    sys.path.insert(0, str(root.resolve()))
    try:
        from models.intrinsic_networks import I2ANNet
    finally:
        sys.path.pop(0)
    model = I2ANNet(base_filters=48)
    state = torch.load(source, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state, strict=True)
    return model.to(DEV).eval(), len(state)


def mask_and_feather():
    pixel = (np.arange(WINDOW) + .5) * (2 / WINDOW) - 1
    x, y = pixel[None, :], pixel[:, None]
    radius = np.hypot(x / .80, (y - .17) / .78)
    eye_left = np.hypot((x + .4) / .19, (y + .19) / .11)
    eye_right = np.hypot((x - .4) / .19, (y + .19) / .11)
    mouth = np.hypot(x / .28, (y - .45) / .12)
    valid = (radius < 1) & (eye_left > 1) & (eye_right > 1) & (mouth > 1)
    feather = .5 + .5 * np.cos(np.pi * np.clip((radius - .70) / .30, 0, 1))
    return valid, feather * valid


@torch.inference_mode()
def normalize_face(model, image, eyes):
    distance = np.linalg.norm(np.subtract(*eyes))
    cx, cy = np.mean(eyes, axis=0)
    length = int(round(2.5 * distance))
    x0, y0 = round(cx - length / 2), round(cy + .24 * distance - length / 2)
    crop = image[y0:y0+length, x0:x0+length]
    if crop.shape != (length, length, 3):
        raise ValueError("Face crop exceeds image")
    small = np.asarray(Image.fromarray(crop).resize((WINDOW, WINDOW), Image.Resampling.BILINEAR))
    valid, feather = mask_and_feather()
    face = torch.from_numpy(linear(small)).permute(2, 0, 1)[None].to(DEV, torch.float32)
    mask = torch.from_numpy(valid)[None, None].to(DEV, torch.float32)
    albedo = model({"face": face, "mask": mask})["albedo"]
    if albedo.shape != face.shape or not torch.isfinite(albedo).all():
        raise ValueError("Invalid predicted diffuse color")
    prior = F.interpolate(albedo, size=(length, length), mode="bilinear", align_corners=False)
    prior = prior[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    mix = F.interpolate(torch.from_numpy(feather)[None, None], size=(length, length),
                        mode="bilinear", align_corners=False)[0, 0].numpy()
    source = linear(crop)
    estimate = prior * ((source @ LUMA) / np.maximum(prior @ LUMA, 1e-8))[..., None]
    merged = source * (1 - mix[..., None]) + estimate * mix[..., None]
    changed = image.copy()
    changed[y0:y0+length, x0:x0+length] = display(merged)
    return changed, dict(crop=[x0, y0, length], clip_fraction=float((merged > 1).mean()),
                         changed_pixels=int((changed != image).any(axis=-1).sum()))


def diagnostic(frames, zones):
    cheeks = {}
    for name in ("cheek_left", "cheek_right"):
        samples = {light: {state: picture[zones[light][name]] for state, picture in states.items()}
                   for light, states in frames.items()}
        edit = {light: mae(states["off"], states["on"]) for light, states in samples.items()}
        # Pose varies slightly across captures. Compare per-region RGB medians.
        lamp = {state: float(np.abs(np.median(samples["cold"][state], axis=0)
                                  - np.median(samples["warm"][state], axis=0)).mean())
                for state in ("off", "on")}
        cheeks[name] = dict(edit=edit, lamp=lamp)
    return cheeks


def run(out):
    parent = BASE / "gros-film-b-cross-v2"
    manifest = checked_json(parent / "results.json", BASELINE_SHA)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    torch.set_num_threads(4)
    model, tensors = get_model(BASE / "split-intake-v1")
    out.mkdir()
    refs, statistics, files = {}, {}, {}
    for person in GROUPS:
        original, normalized, zones = {}, {}, {}
        files[person] = {}
        for light in LIGHTS:
            original[light], normalized[light], files[person][light] = {}, {}, {}
            key = f"{person}-{light}"
            for state in ("off", "on"):
                image = checked_rgb(parent / f"{key}-{state}.png", manifest["inputs"][key]["hashes"][state])
                corrected, details = normalize_face(model, image, labels[person][light]["eyes"])
                original[light][state] = image
                normalized[light][state] = corrected
                files[person][light][state] = dict(details, sha=save(out / f"{key}-{state}-split.png", corrected))
            zones[light] = masks(labels[person][light], labels["screen"], image.shape[:2])
        refs[person] = normalized
        statistics[person] = dict(original=diagnostic(original, zones), normalized=diagnostic(normalized, zones))

    del model
    torch.cuda.empty_cache()
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or manifest["strength"] != STRENGTH:
        raise ValueError("Changed 13 model or strength")
    case = manifest["cases"]["01-from-02"]
    subject = checked_rgb(parent / "01-cold-off.png", manifest["inputs"]["01-cold"]["hashes"]["off"])
    expected = checked_rgb(parent / "01-from-02-expected.png", case["expected_sha"])
    zones = masks(labels["01"]["cold"], labels["screen"], subject.shape[:2])
    frames, boards, grades = {}, {}, {}
    for light, states in refs["02"].items():
        frames[light] = {}
        baseline = {}
        for state, image in states.items():
            result = finish_spatial(subject, engine.prepare(subject, image).apply(subject), STRENGTH)
            frames[light][state] = result
            grades[f"{light}-{state}"] = save(out / f"target-{light}-{state}.png", result)
            baseline[state] = checked_rgb(parent / f"01-from-02-13-{light}-{state}.png",
                                          case["hashes"][f"13-{light}-{state}"])
        boards[light] = panel(out / f"{light}-comparison.png", {
            "target": {"original": subject, "Film B": expected},
            "reference": states, "13": baseline, "SPLiT": frames[light],
        })
    json_to(out / "results.json", dict(
        parent=BASELINE_SHA, checkpoint=CHECKPOINT, source=CODE, weight=WEIGHT, loaded=tensors,
        files=files, stats=statistics, grades=grades, boards=boards,
        scores=score(subject, expected, frames, zones),
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
