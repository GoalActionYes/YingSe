"""Check whether author-trained DINOLight helps reference-based portrait grading."""

from argparse import ArgumentParser
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, STRENGTH, SpatialEngine, checked_json, checked_rgb, digest,
    finish_spatial, json_to, panel, save,
)

DINO_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHTS = {
    "dinolight.pth": "da21365039dfa8e924ee544d217095ed6469aeffa1593403093c857d1b3a32ca",
    "dinov2.pth": "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71",
}
DINO_ZIP = "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48"
SIZE = (672, 448)
DEV = torch.device("cuda")


def resized(image):
    return np.asarray(Image.fromarray(image).resize(SIZE, Image.Resampling.LANCZOS)).copy()


def load(intake):
    if any(digest(intake / name) != sha for name, sha in WEIGHTS.items()):
        raise ValueError("Changed model weights")
    if digest(intake / "dinov2.zip") != DINO_ZIP:
        raise ValueError("Changed DINO source bundle")
    if digest(intake / "model_DINOLight_amp.py") != "c247017a6133c611532b90295dfed5de2c71a36467d66670ec8679370c2e73a8":
        raise ValueError("Changed DINOLight source")
    spec = importlib.util.spec_from_file_location("author_dinolight", intake / "model_DINOLight_amp.py")
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    backbone = torch.hub.load(str((intake / f"dinov2-{DINO_COMMIT}").resolve()),
                              "dinov2_vitb14_reg", source="local", pretrained=False)
    dino = torch.load(intake / "dinov2.pth", weights_only=True, map_location="cpu")
    backbone.load_state_dict(dino, strict=True)
    light = author.DINOLight_amp(channels=32)
    ckpt = torch.load(intake / "dinolight.pth", weights_only=True, map_location="cpu")
    state = {key.removeprefix("module."): tensor for key, tensor in ckpt["network_state_dict"].items()}
    light.load_state_dict(state, strict=True)
    return backbone.eval().to(DEV), light.eval().to(DEV), {"dinov2": len(dino), "dinolight": len(state)}


def features(model, image):
    mean = image.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = image.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    tokens = model.prepare_tokens_with_masks((image - mean) / std, None)
    early = []
    for index, block in enumerate(model.blocks):
        tokens = block(tokens)
        if index in (1, 6):
            early.append(tokens)
    start = 1 + model.num_register_tokens
    height, width = image.shape[-2:]
    final = [model.norm(tokens), *early]
    return torch.cat([F.normalize(table[:, start:].transpose(1, 2).reshape(1, -1, height // 14, width // 14),
                                  dim=1) for table in final], dim=1)


@torch.inference_mode()
def restore(backbone, model, frame):
    image = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(DEV, torch.float32) / 255
    patch = SIZE[1]
    overlap = 2 * patch - SIZE[0]

    def restore_patch(tile):
        return model(tile, features(backbone, tile))

    left = restore_patch(image[..., :patch])
    right = restore_patch(image[..., -patch:])
    ramp = torch.linspace(0, 1, overlap, device=image.device)
    joint = left[..., -overlap:] * (1 - ramp) + right[..., :overlap] * ramp
    output = torch.cat([left[..., :-overlap], joint, right[..., overlap:]], dim=-1)
    if output.shape != image.shape or not torch.isfinite(output).all():
        raise ValueError("Invalid restoration result")
    clip = float(((output < 0) | (output > 1)).float().mean())
    output = output.clamp(0, 1).mul(255).to(torch.uint8)
    return output[0].permute(1, 2, 0).cpu().numpy(), clip


def mae(a, b):
    return float(np.abs(a.astype(np.int16) - b).mean())


def run(out):
    intake = BASE / "dinolight-intake-v1"
    images = BASE / "gros-film-b-cross-v2"
    manifest = checked_json(images / "results.json", BASELINE_SHA)
    annotations = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    torch.set_num_threads(4)
    backbone, model, tensors = load(intake)
    out.mkdir()
    physical = {}
    cl3an = BASE / "cl3an-real-light-input"
    cl3an_id = BASE.parents[1] / "docs/research/2026-11-21-cl3an-pointwise-bound-evidence.json"
    expected_hashes = checked_json(cl3an_id, "2ac743dd76c29035a18ceaf37d7a8c7eecd3e511ea987d146517428f6dd95d30")
    for scene, record in expected_hashes.items():
        source = resized(checked_rgb(cl3an / f"{scene}_1_IN.png", record["source_sha256"]["color"]))
        ambient = resized(checked_rgb(cl3an / f"{scene}_GT.png", record["source_sha256"]["ambient"]))
        result, clip = restore(backbone, model, source)
        before, after = mae(source, ambient), mae(result, ambient)
        physical[scene] = dict(before=before, after=after, passed=after <= .8 * before, clip=clip,
                               sha=save(out / f"cl3an-{scene}-restored.png", result))

    refs, info = {}, {}
    for light in LIGHTS:
        key = f"02-{light}"
        refs[light], info[light] = {}, {}
        for state in ("off", "on"):
            image = resized(checked_rgb(images / f"{key}-{state}.png", manifest["inputs"][key]["hashes"][state]))
            result, clip = restore(backbone, model, image)
            refs[light][state] = {"resize": image, "DINOLight": result}
            info[light][state] = dict(clip=clip, shift=mae(result, image),
                                      sha=save(out / f"ref-{light}-{state}.png", result))
        off, on = refs[light]["off"], refs[light]["on"]
        info[light]["reference_edit_mae"] = {key: mae(off[key], on[key]) for key in off}
    del model, backbone
    torch.cuda.empty_cache()

    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT or manifest["strength"] != STRENGTH:
        raise ValueError("Changed grade engine or strength")
    original = checked_rgb(images / "01-cold-off.png", manifest["inputs"]["01-cold"]["hashes"]["off"])
    expected = checked_rgb(images / "01-from-02-expected.png", manifest["cases"]["01-from-02"]["expected_sha"])
    zones = masks(annotations["01"]["cold"], annotations["screen"], original.shape[:2])
    predictions, outputs = {}, {}
    for arm in ("resize", "DINOLight"):
        predictions[arm], outputs[arm] = {}, {}
        for light in LIGHTS:
            predictions[arm][light] = {}
            for state in ("off", "on"):
                ref = refs[light][state][arm]
                rgb = finish_spatial(original, engine.prepare(original, ref).apply(original), STRENGTH)
                predictions[arm][light][state] = rgb
                outputs[arm][f"{light}-{state}"] = save(out / f"{arm}-{light}-{state}.png", rgb)
    scores = {arm: score(original, expected, renders, zones) for arm, renders in predictions.items()}
    panels = {light: panel(out / f"{light}-comparison.png", {
        "target": {"original": original, "Film B": expected},
        "input ref": {state: refs[light][state]["resize"] for state in ("off", "on")},
        "restored ref": {state: refs[light][state]["DINOLight"] for state in ("off", "on")},
        **{arm: group[light] for arm, group in predictions.items()},
    }) for light in LIGHTS}
    json_to(out / "results.json", dict(
        parent=BASELINE_SHA, checkpoint=CHECKPOINT, assets=WEIGHTS, dino_source=DINO_ZIP,
        light_source=digest(intake / "model_DINOLight_amp.py"), loaded=tensors,
        physical=physical, references=info, scores=scores, outputs=outputs, panels=panels,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
