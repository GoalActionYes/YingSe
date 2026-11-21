"""Estimate an unfiltered reference with CIFR and replay the inferred editing."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision.models.vgg import cfgs, make_layers

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_paired_cubic_cross_content import apply, paired_fit
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save

ASSETS = {
    "cifr.pth": "7445bef23789f3eb8cff4d18b5d5872c24ac312376601e350e78e5cce94ec491",
    "vgg-head.bin": "5f5f67ed9fcae04f75dfa0cbc8caa5e83312b3e4e00f26344933edfd2475f41a",
    "vgg-features.bin": "d649de3c8126095cbd4ddf98051ee4c6eb54127548e4623473d5a202c46418fe",
    "modeling/arch.py": "9f2a5ef3848f1b58b132fed15d915b7671ce281458c37b793759a68078f571e8",
    "modeling/base.py": "738bf2c317747a96c0e148352eafc2099e1e33214bdadc6f2c112a6eecb9eba6",
    "modules/blocks.py": "d48c1a37b8f1f2c436f0fc0439d792476083c4863244e6d2ef78d476b8f12b33",
    "modules/normalization.py": "cedca4270cd0d535e0a0597275a03cefb7b098515cd2cc7f845298c95622c3e2",
}
DEV = torch.device("cuda")
STATES = ("off", "on")


def vgg_weights(root):
    first = (root / "vgg-head.bin").read_bytes()
    length = int.from_bytes(first[:8], "little")
    if 8 + length != 2810:
        raise ValueError("Changed VGG offsets")
    index = json.loads(first[8:8+length])
    raw = (root / "vgg-features.bin").read_bytes()
    weights = {}
    for name, table in index.items():
        if not name.startswith("features."):
            continue
        start, stop = table["data_offsets"]
        if table["dtype"] != "F32" or not 0 <= start < stop <= len(raw):
            raise ValueError(f"Invalid metadata: {name}")
        array = np.frombuffer(raw[start:stop], dtype="<f4").reshape(table["shape"]).copy()
        weights[name.removeprefix("features.")] = torch.from_numpy(array)
    return weights


def load(root):
    if any(digest(root / path) != sha for path, sha in ASSETS.items()):
        raise ValueError("Changed CIFR or VGG assets")
    sys.path.insert(0, str(root.resolve()))
    try:
        from modeling.arch import CIFR_Encoder
    finally:
        sys.path.pop(0)
    model = CIFR_Encoder(base_n_channels=32, destyler_n_channels=32)
    state = torch.load(root / "cifr.pth", weights_only=True, map_location="cpu")["ifr"]
    model.load_state_dict(state, strict=True)
    vgg = make_layers(cfgs["D"])
    layers = vgg_weights(root)
    vgg.load_state_dict(layers, strict=True)
    return model.to(DEV).eval(), vgg.to(DEV).eval(), dict(cifr=len(state), vgg=len(layers))


@torch.inference_mode()
def restore(model, vgg, picture):
    tensor = torch.from_numpy(picture.copy()).to(DEV, torch.float32).permute(2, 0, 1)[None] / 255
    tensor = tensor.mul(255).div(127.5).sub(1)
    output, _ = model(tensor, vgg(tensor))
    if output.shape != tensor.shape or not torch.isfinite(output).all():
        raise ValueError("CIFR did not return a valid RGB image")
    clip = float(((output < 0) | (output > 1)).float().mean())
    frame = output[0].clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return frame, clip


def small(picture):
    return np.asarray(Image.fromarray(picture).resize((256, 256), Image.Resampling.BILINEAR))


def mae(a, b):
    return float(np.abs(a.astype(np.int16) - b).mean())


def grade(target, inferred, input_ref):
    if np.array_equal(inferred, input_ref):
        return target.copy(), dict(identity=True)
    fitted, stats = paired_fit(inferred, input_ref)
    return apply(target, fitted), stats


def run(out):
    repo = BASE / "gros-film-b-cross-v2"
    ledger = checked_json(repo / "results.json", BASELINE_SHA)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    row = ledger["cases"]["01-from-02"]
    source = checked_rgb(repo / "01-cold-off.png", ledger["inputs"]["01-cold"]["hashes"]["off"])
    expected = checked_rgb(repo / "01-from-02-expected.png", row["expected_sha"])
    zones = masks(labels["01"]["cold"], labels["screen"], source.shape[:2])
    torch.set_num_threads(4)
    model, vgg, counts = load(BASE / "cifr-intake-v1")
    out.mkdir()
    restored, observed = {}, {}
    for light in LIGHTS:
        key = f"02-{light}"
        observed[light], restored[light] = {}, {}
        for state in STATES:
            observed[light][state] = small(checked_rgb(repo / f"{key}-{state}.png",
                                                      ledger["inputs"][key]["hashes"][state]))
            restored[light][state] = restore(model, vgg, observed[light][state])
    outputs = {"CIFR": {}, "known": {}}
    refs, grades, boards = {}, {}, {}
    for light in LIGHTS:
        refs[light], grades[light] = {}, {}
        outputs["CIFR"][light], outputs["known"][light] = {}, {}
        truth = observed[light]["off"]
        for state in STATES:
            ref = observed[light][state]
            inferred, clip = restored[light][state]
            refs[light][state] = dict(error_to_before=mae(inferred, truth), raw_error_to_before=mae(ref, truth),
                                      change=mae(inferred, ref), clip=clip,
                                      sha256=save(out / f"reference-{light}-{state}.png", inferred))
            grades[light][state] = {}
            for arm, estimate in (("CIFR", inferred), ("known", truth)):
                output, metrics = grade(source, estimate, ref)
                outputs[arm][light][state] = output
                grades[light][state][arm] = dict(fit=metrics,
                    sha256=save(out / f"{arm}-{light}-{state}.png", output))
        np.testing.assert_array_equal(outputs["known"][light]["off"], source)
        boards[light] = panel(out / f"{light}-comparison.png", {
            "target": {"original": source, "Film B": expected},
            "reference": observed[light],
            "inferred original": {state: restored[light][state][0] for state in STATES},
            **{arm: data[light] for arm, data in outputs.items()},
        })
    scores = {arm: score(source, expected, predictions, zones) for arm, predictions in outputs.items()}
    json_to(out / "results.json", dict(parent=BASELINE_SHA, assets=ASSETS, loaded=counts, references=refs,
                                       scores=scores, grades=grades, boards=boards,
                                       peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
