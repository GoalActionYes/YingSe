"""Measure whether portrait illumination decomposition separates capture from edit."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from probe_gros_film_b_cross import GROUPS, LIGHTS
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_gros_split_face_albedo import CODE, DEV, WEIGHT, WINDOW, get_model, mask_and_feather
from probe_physical_blue_global_fit import display, linear
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel

DS_SHA = "30a867e93900fbd0d77b176695eb5a9dc412486948567b44b0f12c2939048e35"
STATES = ("off", "on")
COMPONENTS = ("shading", "specular")


def crop_face(image, eyes):
    distance = np.linalg.norm(np.subtract(*eyes))
    cx, cy = np.mean(eyes, axis=0)
    size = int(round(2.5 * distance))
    x0 = round(cx - size / 2)
    y0 = round(cy + .24 * distance - size / 2)
    cropped = Image.fromarray(image[y0:y0+size, x0:x0+size])
    if cropped.size != (size, size):
        raise ValueError("Face crop exceeds image")
    return np.asarray(cropped.resize((WINDOW, WINDOW), Image.Resampling.BILINEAR))


def load_models(root):
    first, count = get_model(root)
    from models.intrinsic_networks import IAN2DSNet

    path = root / "intrinsic_decom_f_DS.pth"
    if digest(path) != DS_SHA:
        raise ValueError("Changed SPLiT shading and specular weights")
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    model = IAN2DSNet(base_filters=48)
    model.load_state_dict(state, strict=True)
    return first, model.to(DEV).eval(), dict(first=count, second=len(state))


def rgb8_difference(a, b):
    return float(np.abs(a.astype(np.int16) - b).mean())


@torch.inference_mode()
def decompose(first, second, crop, valid):
    original = linear(crop)
    face = torch.from_numpy(original).permute(2, 0, 1)[None].to(DEV, torch.float32)
    mask = torch.from_numpy(valid)[None, None].to(DEV, torch.float32)
    inputs = dict(face=face, mask=mask)
    parts = first(inputs)
    parts.update(second(inputs | parts))
    predicted = {name: value[0].permute(1, 2, 0).cpu().numpy()
                 for name, value in parts.items() if name != "normal"}
    if any(img.shape != crop.shape or not np.isfinite(img).all() for img in predicted.values()):
        raise ValueError("Invalid prediction")
    median = {name: np.median(img[valid], axis=0).astype(np.float64).tolist()
              for name, img in ({"original": original} | predicted).items()}
    reconstruction = display(predicted["face"].astype(np.float64))
    error = rgb8_difference(reconstruction[valid], crop[valid])
    preview = {"original": crop, "reconstruction": reconstruction,
               **{name: display(predicted[name].astype(np.float64)) for name in ("albedo", *COMPONENTS)}}
    return dict(median=median, rgb8_recon_mae=error), preview


def chroma(median):
    log_rgb = np.log(np.maximum(median, 1e-6))
    return log_rgb[[0, 2]] - log_rgb[1]


def assess(frames):
    signals = {}
    for key in ("original", *COMPONENTS):
        values = {light: {state: chroma(frames[light][state]["median"][key]) for state in STATES}
                  for light in LIGHTS}
        lamp = {state: values["warm"][state] - values["cold"][state] for state in STATES}
        edit = {light: values[light]["on"] - values[light]["off"] for light in LIGHTS}
        signals[key] = dict(lamp=lamp, edit=edit)
    scored = {}
    recon_ok = all(info["rgb8_recon_mae"] <= 5 for light in frames.values() for info in light.values())
    for key, signals_by_kind in signals.items():
        lamp = signals_by_kind["lamp"]
        edit = signals_by_kind["edit"]
        norms = {state: float(np.linalg.norm(value)) for state, value in lamp.items()}
        style = {light: float(np.linalg.norm(value)) for light, value in edit.items()}
        orientation = {}
        for state, value in lamp.items():
            original = signals["original"]["lamp"][state]
            den = float(np.linalg.norm(original)) * norms[state]
            orientation[state] = None if den == 0 else float(np.dot(original, value) / den)
        ratio = max(style.values()) / min(norms.values()) if min(norms.values()) > 0 else None
        qualified = (recon_ok and min(norms.values()) >= .10 and ratio is not None and ratio <= .5
                     and all(item is not None and item >= .75 for item in orientation.values()))
        scored[key] = dict(lamp_norm=norms, edit_norm=style, lamp_cosine=orientation,
                           max_edit_over_min_lamp=ratio, qualified=qualified,
                           vectors={kind: {k: v.tolist() for k, v in rows.items()}
                                    for kind, rows in signals_by_kind.items()})
    return scored


def run(out):
    parent = BASE / "gros-film-b-cross-v2"
    manifest = checked_json(parent / "results.json", BASELINE_SHA)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    torch.set_num_threads(4)
    first, second, tensors = load_models(BASE / "split-intake-v1")
    valid, _ = mask_and_feather()
    out.mkdir()
    cases = {}
    for person in GROUPS:
        frames = {}
        for light in LIGHTS:
            frames[light] = {}
            key = f"{person}-{light}"
            for state in STATES:
                portrait = checked_rgb(parent / f"{key}-{state}.png", manifest["inputs"][key]["hashes"][state])
                crop = crop_face(portrait, labels[person][light]["eyes"])
                info, previews = decompose(first, second, crop, valid)
                info["panel_sha"] = panel(out / f"{key}-{state}-decomposition.png", {"components": previews})
                frames[light][state] = info
        cases[person] = dict(frames=frames, score=assess(frames))
    admitted = {key: all(case["score"][key]["qualified"] for case in cases.values()) for key in COMPONENTS}
    json_to(out / "results.json", dict(parent=BASELINE_SHA, source=CODE, weights=dict(first=WEIGHT, second=DS_SHA),
                                       loaded=tensors, cases=cases, admitted=admitted,
                                       peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
