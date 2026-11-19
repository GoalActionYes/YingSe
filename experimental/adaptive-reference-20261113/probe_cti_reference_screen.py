"""Screen independently trained illuminant estimates and corrections for two-image transfer."""

import argparse
import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from background_intervention import target_regions
from probe_arri_film_b_dose import verified
from probe_arri_shared_look import digest, quantize
from probe_arri_two_looks import CHECKPOINT
from probe_capture_invariant_look import natural_scenes
from probe_hdm_hdr_display import json_to
from probe_hdm_hdr_equal_luminance import verified_json
from probe_pair_supervised_reference_correction import CAPTURE_SHA, LEVELS, LIGHTS, SCENES, STATES, save, score_target
from probe_psbench_background_desaturation import REC709
from spatial_render import SpatialEngine, finish_spatial


ASSET_SHA = {
    "archs.py": "c008f1a2fb969accde84e7d43c53b609429689c51fac186cde77719f6bcb13aa",
    "cti-phone.pth": "8f3ade3c7c39c09b16de398b419a17d5f957ceef7167edd134eeddc90b2fa8cf",
}
ARMS = ("author", "chroma")
LIGHT_XY = {"white": np.array([.3127, .3290]), "blue": np.array([.1968, .1623])}


def model(assets, device):
    for name, identity in ASSET_SHA.items():
        if digest(assets / name) != identity:
            raise ValueError(f"Changed author asset: {name}")
    spec = importlib.util.spec_from_file_location("cti_archs_local", assets / "archs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.CTIRetinexformer()
    container = torch.load(assets / "cti-phone.pth", map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): tensor for key, tensor in container.get("params", container).items()}
    if not all(isinstance(tensor, torch.Tensor) and tensor.isfinite().all() for tensor in state.values()):
        raise ValueError("Invalid author model state")
    net.load_state_dict(state, strict=True)
    return net.to(device).eval().requires_grad_(False)


def estimate(net, rgb, device):
    x = rgb.astype(np.float32) / 255
    lin = np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
    height, width = rgb.shape[:2]
    raw = torch.from_numpy(lin).permute(2, 0, 1)[None].to(device)
    pred, intensity, xy, _ = net(F.pad(raw, (0, -width % 4, 0, -height % 4), mode="reflect"))
    if not all(tensor.isfinite().all() for tensor in (pred, intensity, xy)):
        raise ValueError("Nonfinite author inference")
    out = pred[0, :, :height, :width].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    output = quantize(np.where(out <= .0031308, 12.92 * out, 1.055 * out ** (1 / 2.4) - .055))
    xy = xy.mean(dim=(0, 2, 3)).cpu().numpy()
    return output, {"xy": xy.tolist(), "intensity": float(intensity.mean()),
                    "nearest": min(LIGHT_XY, key=lambda label: float(np.linalg.norm(xy - LIGHT_XY[label])))}


def preserve_luma(reference, corrected):
    original_luma = (reference @ REC709)[..., None]
    residual = corrected - (corrected @ REC709)[..., None]
    high = np.divide(255 - original_luma, residual, out=np.ones_like(residual), where=residual > 0)
    low = np.divide(-original_luma, residual, out=np.ones_like(residual), where=residual < 0)
    scale = np.minimum(1, np.minimum(high, low).min(axis=2))
    result = np.rint(np.clip(original_luma + residual * scale[..., None], 0, 255)).astype(np.uint8)
    peak = float(np.abs((result.astype(float) - reference) @ REC709).max())
    if peak > .5 + 1e-9:
        raise ValueError("Luminance preservation failed")
    return result, peak


@torch.inference_mode()
def run(base, assets, out):
    torch.set_num_threads(4)
    parent = base / "chromamill-capture-cross-v1"
    prior = verified_json(parent / "results.json", CAPTURE_SHA)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or prior["model_sha256"] != CHECKPOINT:
        raise ValueError("Changed 13 model")
    net = model(assets, engine.device)
    out.mkdir()
    folders = {arm: out / arm for arm in ARMS}
    for folder in folders.values():
        folder.mkdir()
    correction, source_images, reference_evidence = {}, {}, {}
    negatives = {arm: {} for arm in ARMS}

    def alter(name, rgb):
        author, illumination = estimate(net, rgb, engine.device)
        chroma, peak = preserve_luma(rgb, author)
        images = {"author": author, "chroma": chroma}
        effects = {arm: {"sha256": save(folders[arm], f"{name}-reference", image),
                         "change_rgb_mae": float(np.abs(image.astype(float) - rgb).mean())}
                   for arm, image in images.items()}
        effects["chroma"]["peak_luma_error"] = peak
        return images, illumination | {"effects": effects}

    for scene, light, level, state in product(SCENES, LIGHTS, LEVELS, STATES):
        name = f"{scene}-{light}-{level}-{state}"
        identity = prior["prepared"][scene][f"{light}-{level}"]["hashes"][state]
        rgb = verified(parent / f"{name}.png", identity)
        source_images[scene, light, level, state] = rgb
        corrected, record = alter(name, rgb)
        correction[scene, light, level, state] = corrected
        reference_evidence[name] = {"label": light, **record}
    for scene, images in natural_scenes(base).items():
        for state in ("off", "film_b"):
            tag = f"{scene}-{state}"
            _, record = alter(tag, images[state])
            reference_evidence[tag] = record
            for arm in ARMS:
                edit = record["effects"][arm]
                negatives[arm][tag] = edit | {"passed": edit["change_rgb_mae"] <= 1.5}

    report = {arm: {} for arm in ARMS}
    for scene, donor in (SCENES, SCENES[::-1]):
        old = prior["cases"][scene]
        target = source_images[scene, "white", 241, "off"]
        for arm in ARMS:
            pictures, files = {}, {}
            for light, level in product(LIGHTS, LEVELS):
                pictures[light, level] = {}
                for state in STATES:
                    reference = correction[donor, light, level, state][arm]
                    grade = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
                    name = f"{light}-{level}-{state}"
                    files[name] = save(folders[arm], f"{scene}-{name}-13", grade)
                    pictures[light, level][state] = grade
            styled = source_images[scene, "white", 241, "on"]
            expected = finish_spatial(target, styled, .8)
            whole = np.ones(target.shape[:2], dtype=bool)
            neutral = np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"] for rgb in (target, styled)))
            if neutral.sum() != prior["checks"][scene]["neutral_pixels"]:
                raise ValueError("Changed neutral scoring support")
            masks = {"whole": whole, "neutral": neutral}
            rows = score_target(target, expected, pictures, old, masks)
            for label, mask in masks.items():
                previous = {}
                for level, state in product(LEVELS, STATES):
                    filename = f"white-{level}-{state}"
                    frame = verified(parent / f"{scene}-{filename}-13.png", old["hashes"][filename])
                    expected_state = expected if state == "on" else target
                    error = float(np.abs(frame[mask].astype(float) - expected_state[mask]).mean())
                    previous[f"{level}-{state}"] = rows[label]["white_error"][f"{level}-{state}"] - error
                rows[label]["white_error_increase"] = previous
            passes = all(row["passed"] for col in rows["neutral"]["effects"].values() for row in col.values())
            passes &= all(row["passed"] for field in rows.values() for row in field["styles"].values())
            passes &= all(value <= 2 for field in rows.values() for value in field["white_error_increase"].values())
            report[arm][scene] = {"hashes": files, "rows": rows, "passed": passes}

    sources = [row for row in reference_evidence.values() if "label" in row]
    transitions = [reference_evidence[f"{scene}-{light}-{level}-off"]["nearest"] ==
                   reference_evidence[f"{scene}-{light}-{level}-on"]["nearest"]
                   for scene, light, level in product(SCENES, LIGHTS, LEVELS)]
    gates = {arm: all(row["passed"] for row in (*report[arm].values(), *negatives[arm].values())) for arm in ARMS}
    json_to(out / "results.json", {"assets": ASSET_SHA, "capture_sha256": CAPTURE_SHA, "model_sha256": CHECKPOINT,
                                   "light_correct": sum(row["nearest"] == row["label"] for row in sources),
                                   "light_total": len(sources), "classification_stable": sum(transitions),
                                   "classification_pairs": len(transitions), "references": reference_evidence,
                                   "arms": report, "negatives": negatives, "gates": gates, "passed": any(gates.values())})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "assets", "out"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
