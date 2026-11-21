"""Compare the author's two-image and paired-reference VeraRetouch inputs."""

from argparse import ArgumentParser
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import (
    BASE, STRENGTH, checked_json, checked_rgb, digest, finish_spatial, json_to, panel, save,
)

WEIGHT = "62457c0556a29b07d008774adb5fdd95be96fa8bc8539a034c5822afe7568d58"
EXPECTED_BYTES = 182507490
SOURCE = {
    "model/resnet.py": ("43f86923d63b6cd81a2903e22285c2f725a37d8220aed0b96acd5744e0ef13b8",
                        ("SpatialAttention", "ChannelAttention", "CBAM", "Resnet18Encoder_InputCatMixedCBAM")),
    "model/colormlp_v2.py": ("cb0c31445a78833a6564693fda437d1791e71d8ca5d58cca6e492974f815efcd", ("ConditionalMLPDecoder",)),
    "model/retouch_render.py": ("f6da55822743bf29b358f20c5ba0036b22fff04fe2fd31d3700d2083728c1579", ("RetouchRenderer",)),
    "configs/renderer_config.py": ("940121096e346a3b1ae94ea11b896e521d40e7828758ae2d62c246b4ece192d0", ("load_config",)),
}
DEV = torch.device("cuda")
NAME = "RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM"
ARM_NAMES = ("two_image", "known_before")


def load(intake):
    namespace = dict(torch=torch, nn=nn, F=F, AttrDict=lambda values: SimpleNamespace(**values))
    for name in ("Conv2d", "MaxPool2d", "Flatten", "Linear", "Sequential", "BatchNorm2d", "ReLU", "AdaptiveAvgPool2d"):
        namespace[name] = getattr(nn, name)
    for filename, (sha, used) in SOURCE.items():
        path = intake / filename
        if digest(path) != sha:
            raise ValueError(f"Changed author code: {filename}")
        tree = ast.parse(path.read_bytes(), filename=str(path))
        symbols = [node for node in tree.body if getattr(node, "name", None) in used]
        if len(symbols) != len(used):
            raise ValueError(f"Missing author definitions: {filename}")
        module = ast.Module(body=symbols, type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)
    renderer = namespace["RetouchRenderer"](namespace["load_config"](NAME))
    path = intake / "encoder_renderer.pth"
    if path.stat().st_size != EXPECTED_BYTES or digest(path) != WEIGHT:
        raise ValueError("Changed VeraRetouch weights")
    weights = torch.load(path, weights_only=True, map_location="cpu")["model_state_dict"]
    renderer.load_state_dict(weights, strict=True)
    return renderer.to(DEV).eval(), len(weights)


def target_tensor(image):
    return torch.from_numpy(image.copy()).to(DEV, torch.float32).permute(2, 0, 1)[None].contiguous() / 127.5 - 1


def reference_tensor(image):
    # The author's reference loader resizes the cv2 BGR array directly.
    resized = Image.fromarray(image[..., ::-1]).resize((512, 512), Image.Resampling.BILINEAR)
    return target_tensor(np.asarray(resized))


@torch.inference_mode()
def render(renderer, original, before, after):
    x = target_tensor(original)
    ref_before, ref_after = (reference_tensor(image) for image in (before, after))
    raw = renderer(x, ref_before, ref_after, x.new_ones((1, 3)), 262144)
    if raw.shape != x.shape or not torch.isfinite(raw).all():
        raise ValueError("Invalid VeraRetouch grade")
    output = raw.clamp(0, 1).mul(255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    return finish_spatial(original, output, STRENGTH)


def run(out):
    photos = BASE / "gros-film-b-cross-v2"
    archive = checked_json(photos / "results.json", BASELINE_SHA)
    labels = checked_json(Path(__file__).with_name("gros-portrait-light-regions.json"), REGIONS_SHA)
    case = archive["cases"]["01-from-02"]
    original = checked_rgb(photos / "01-cold-off.png", archive["inputs"]["01-cold"]["hashes"]["off"])
    expected = checked_rgb(photos / "01-from-02-expected.png", case["expected_sha"])
    area = masks(labels["01"]["cold"], labels["screen"], original.shape[:2])
    torch.set_num_threads(4)
    renderer, loaded = load(BASE / "vera-intake-v1")
    out.mkdir()
    predictions = {arm: {} for arm in ARM_NAMES}
    reports, boards = {}, {}
    for light in LIGHTS:
        key = f"02-{light}"
        refs = {state: checked_rgb(photos / f"{key}-{state}.png", archive["inputs"][key]["hashes"][state])
                for state in ("off", "on")}
        reports[light] = {}
        for arm, before in (("two_image", original), ("known_before", refs["off"])):
            frames = {state: render(renderer, original, before, after) for state, after in refs.items()}
            predictions[arm][light] = frames
            reports[light][arm] = {state: save(out / f"{arm}-{light}-{state}.png", picture)
                                    for state, picture in frames.items()}
        boards[light] = panel(out / f"{light}-comparison.png", {
            "target": {"original": original, "Film B": expected},
            "reference": refs,
            **{arm: predictions[arm][light] for arm in ARM_NAMES},
        })
    same = render(renderer, original, original, original)
    report = dict(
        parent=BASELINE_SHA, loaded=loaded, weight=WEIGHT,
        source={file: sha for file, (sha, _) in SOURCE.items()}, media=reports, boards=boards,
        same=dict(sha=save(out / "same.png", same), error=float(np.abs(same.astype(np.int16) - original).mean())),
        scores={arm: score(original, expected, frames, area) for arm, frames in predictions.items()},
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    )
    json_to(out / "results.json", report)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
