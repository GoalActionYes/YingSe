"""Test depth-conditioned lateral light evidence in the prior real room panel."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from probe_multilum_gray_probe import EVIDENCE_SHA, LUMA, SCENES, linear, rgb, valid
from probe_multilum_lateral_cue import PAIRS
from probe_target_mirror_chroma import BASE, checked_rgb, digest


WEIGHT_SHA = "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345"
PRIOR_SHA = "8d9713c9f56a7c5cac2d8db76e5833ea302912d3ef7ba7aef33ac8c7d67941bb"
CODE_REV = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"


def load_model(root):
    library = root / "source/pretrained_models/dpv2"
    manifest = json.loads((root / "depth-source-manifest.json").read_bytes())
    if manifest["revision"] != CODE_REV:
        raise ValueError("Unexpected depth source revision")
    for entry in manifest["files"]:
        if digest(library / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Unexpected depth source: {entry['path']}")
    path = library / "ckpts/depth_anything_v2_vitl.pth"
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHT_SHA:
            raise ValueError("Unexpected depth model weights")
    sys.path.insert(0, str(library.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    state = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().to("cuda")


def correlation(x, y, z):
    residual_x, residual_z = x.copy(), z.copy()
    square = y @ y
    if square > 1e-12:
        residual_x -= (residual_x @ y / square) * y
        residual_z -= (residual_z @ y / square) * y
    length = np.linalg.norm(residual_x) * np.linalg.norm(residual_z)
    return float((residual_x @ residual_z) / length) if length > 1e-12 else None


def cue(depth, brightness, exposure):
    height, width = depth.shape
    cutoff = int(.70 * height)
    depth, brightness, exposure = depth[:cutoff], brightness[:cutoff], exposure[:cutoff]
    low, high = np.quantile(depth, (.05, .95))
    if high <= low:
        return None, 0, 0
    dy, dx = np.gradient((depth.astype(np.float64) - low) / (high - low))
    strength = np.hypot(dx, dy)
    mask = exposure & (strength <= np.quantile(strength, .95))
    coords = np.stack((dx, dy, brightness), axis=-1)
    views = []
    for vertical in np.array_split(np.arange(cutoff), 4):
        for horizontal in np.array_split(np.arange(width), 6):
            ids = np.ix_(vertical, horizontal)
            values = coords[ids][mask[ids]]
            if len(values) >= 128:
                views.append(values - np.median(values, axis=0))
    samples = sum(map(len, views))
    if len(views) < 6 or samples < 1024:
        return None, len(views), samples
    return correlation(*np.concatenate(views).T), len(views), samples


def test_math():
    y, x = np.mgrid[:128, :192].astype(float) / 128
    depth = np.sin(x * 6) + .25 * np.cos(y * 5)
    dy, dx = np.gradient(depth)
    exposure = np.ones(depth.shape, dtype=bool)
    light = 8 * dx + 3 * dy
    left, _, _ = cue(depth, light, exposure)
    right, _, _ = cue(depth, -light + 2, exposure)
    assert left > .99 and right < -.99


def save_depth(path, depth):
    assert np.isfinite(depth).all()
    low, high = np.quantile(depth, (.05, .95))
    scaled = np.clip((depth - low) / max(high - low, 1e-8), 0, 1)
    Image.fromarray(np.rint(scaled * 65535).astype(np.uint16)).save(path)
    return digest(path)


@torch.inference_mode()
def read_signals(model, archive_root, assets, out):
    archive_evidence = Path("docs/research/2026-11-18-multilum-pilot-evidence.json")
    assert digest(archive_evidence) == EVIDENCE_SHA
    archive_sha = json.loads(archive_evidence.read_bytes())["archives"]
    prior_file = assets / "results.json"
    assert digest(prior_file) == PRIOR_SHA
    prior = json.loads(prior_file.read_bytes())
    result = {}
    for scene in SCENES:
        archive_file = archive_root / f"{scene}.zip"
        assert digest(archive_file) == archive_sha[scene]
        frames = {}
        with ZipFile(archive_file) as archive:
            for index in sorted({value for pair in PAIRS for value in pair}):
                original = rgb(archive.read(f"{scene}/dir_{index}_mip2.jpg"), small=True)
                label = f"{scene}-{index:02d}"
                graded = checked_rgb(assets / f"{label}.png", prior["scenes"][scene][str(index)]["on_sha256"])
                frames[str(index)] = {}
                for key, frame in (("off", original), ("on", graded)):
                    depth = model.infer_image(np.ascontiguousarray(frame[..., ::-1]), input_size=518)
                    assert depth.shape == frame.shape[:2]
                    log_y = np.log(np.maximum(linear(frame) @ LUMA, 1e-7))
                    value, tiles, pixels = cue(depth, log_y, valid(frame))
                    sha = save_depth(out / f"{label}-{key}-depth.png", depth)
                    frames[str(index)][key] = {"score": value, "tiles": tiles, "pixels": pixels, "preview_sha256": sha}
        result[scene] = frames
        print(scene, len(frames), flush=True)
    return result, prior["validation"]


def differences(frames, state):
    pairs = []
    for a, b in PAIRS:
        first, second = (frames[str(key)][state]["score"] for key in (a, b))
        pairs.append(second - first if first is not None and second is not None else None)
    return pairs


def compare(results, previous):
    scores = {}
    for scene, other in (SCENES, SCENES[::-1]):
        included = [delta for delta in differences(results[other], "off") if delta is not None]
        sign = float(np.sign(np.median(included))) if included else 0.
        tests = {}
        for state in ("off", "on"):
            deltas = differences(results[scene], state)
            hits = [delta is not None and delta * sign >= .05 for delta in deltas]
            baseline = previous[scene]["spatial"][state]["passed"]
            tests[state] = {"values": deltas, "passed_pairs": sum(hits), "baseline_pairs": baseline,
                            "passed": sum(hits) >= 8 and sum(hits) - baseline >= 2}
        scores[scene] = {"sign": sign, **tests}
    return scores


def run(archive_root, assets, depth_assets, out):
    test_math()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = load_model(depth_assets)
    out.mkdir()
    frames, previous = read_signals(model, archive_root, assets, out)
    scores = compare(frames, previous)
    report = {"weight_sha256": WEIGHT_SHA, "previous_sha256": PRIOR_SHA, "source_sha256": digest(Path(__file__)),
              "scores": scores, "frames": frames,
              "passed": all(row[state]["passed"] for row in scores.values() for state in ("off", "on"))}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("archive_root", "assets", "depth_assets", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
