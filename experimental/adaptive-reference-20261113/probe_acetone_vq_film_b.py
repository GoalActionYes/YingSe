"""Measure round-trip error of the published AceTone LUT codec."""
from argparse import ArgumentParser
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import grid_sample

from probe_arri_sdr_renderability import transforms
from probe_arri_shared_look import converted, metrics, quantize
from probe_gros_film_b_cross import FILM, LIGHTS
from probe_gros_portrait_light_transfer import GROUPS, masks
from probe_target_mirror_chroma import BASE, STRENGTH, checked_rgb, digest, save
from spatial_render import finish_spatial


FILES = {
    "vq.py": "cefaee8ff214c60f65b95feb511a70958a3fa53ad3ae5e1d72ae7ad646b3d372",
    "acetone-vqvae-d64.pt": "115e4e8c147655fe0ae0c7fd494ec6cd3a8ff5a441ce536a886edb1ea556ef92",
}
PARENT_SHA = "4413ee3468586d3ab5e715013c5e86d9c24412781e8385e75d5664b6e50df260"


def load_model(folder):
    for filename, sha in FILES.items():
        if digest(folder / filename) != sha:
            raise ValueError(f"Unexpected upstream asset: {filename}")
    spec = importlib.util.spec_from_file_location("acetone_vq", folder / "vq.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ckpt = torch.load(folder / "acetone-vqvae-d64.pt", map_location="cpu", weights_only=True)
    options = ckpt.get("args", {})
    model = module.VQVAE3DLUT(codebook_size=options.get("codebook_size", 256),
                             embedding_dim=options.get("embedding_dim", 64))
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    return model.eval()


def sample(table, image):
    # PyTorch samples its last spatial dimension first. Table axes are R,G,B.
    lattice = torch.from_numpy(np.ascontiguousarray(table)).permute(3, 0, 1, 2)[None]
    coords = torch.from_numpy(image.astype(np.float32) / 255)[..., [2, 1, 0]]
    values = grid_sample(lattice, coords[None, None] * 2 - 1, align_corners=True, padding_mode="border")
    return values[0, :, 0].permute(1, 2, 0).numpy()


def roundtrip(model, table):
    # Upstream apply_lut accepts RGB axes with stored BGR output channels.
    tensor = torch.from_numpy(np.ascontiguousarray(table[..., ::-1])).permute(3, 0, 1, 2)[None]
    indices = model.encode_indices(tensor)
    reconstructed = model.decode_indices(indices)
    assert tuple(indices.shape) == (1, 4, 4, 4)
    assert int(indices.min()) >= 0 and int(indices.max()) < model.vq.K
    assert torch.isfinite(reconstructed).all()
    return reconstructed[0].permute(1, 2, 3, 0).numpy()[..., ::-1], indices.flatten().tolist()


@torch.inference_mode()
def run(assets, out):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = load_model(assets)
    source = BASE / "gros-film-b-cross-v2"
    cube = BASE / "arri-second-film-assets-v1" / "second-film.cube"
    parent_file = source / "results.json"
    if digest(parent_file) != PARENT_SHA or digest(cube) != FILM:
        raise ValueError("Unexpected Gros experiment inputs")
    parent = json.loads(parent_file.read_bytes())
    zones = json.loads(Path(__file__).with_name("gros-portrait-light-regions.json").read_bytes())
    view, inverse, film = transforms(cube)
    axis = np.linspace(0, 1, 32, dtype=np.float32)
    identity = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    transform = np.clip(converted(converted(converted(identity, inverse), film), view), 0, 1)
    raw = {"off": identity, "on": transform}
    quantized = {name: roundtrip(model, value) for name, value in raw.items()}
    axis_test = np.array([[[19, 65, 201], [250, 38, 129], [0, 255, 12]]], dtype=np.uint8)
    np.testing.assert_array_equal(quantize(sample(identity, axis_test)), axis_test)
    out.mkdir()
    cases = {}
    for person in GROUPS:
        for light in LIGHTS:
            case = f"{person}-{light}"
            hashes = parent["inputs"][case]["hashes"]
            target, styled = (checked_rgb(source / f"{case}-{label}.png", hashes[label]) for label in raw)
            expected = finish_spatial(target, styled, STRENGTH)
            regions = masks(zones[person][light], zones["screen"], target.shape[:2])
            case_report = {}
            for label, tables in (("full_grid", raw), ("vq", {name: value[0] for name, value in quantized.items()})):
                frames = {name: finish_spatial(target, quantize(sample(table, target)), STRENGTH)
                          for name, table in tables.items()}
                if label == "full_grid":
                    np.testing.assert_array_equal(frames["off"], target)
                rows = {key: metrics(target, expected, frames["off"], frames["on"], region)
                        for key, region in regions.items()}
                saved = {name: save(out / f"{case}-{label}-{name}.png", frame) for name, frame in frames.items()}
                accepted = all(row["passed"] and row["off_matches_original_mae"] <= 3 for row in rows.values())
                case_report[label] = {"regions": rows, "outputs": saved, "passed": accepted}
            cases[case] = case_report
    comparator_valid = all(item["full_grid"]["passed"] and all(
        max(row["off_matches_original_mae"], row["on_matches_authored_mae"]) <= 1
        for row in item["full_grid"]["regions"].values()) for item in cases.values())
    report = {"source_sha256": digest(Path(__file__)), "assets": FILES, "parent_sha256": PARENT_SHA,
              "color_table_ids": {key: value[1] for key, value in quantized.items()}, "cases": cases,
              "uncompressed_comparator_valid": comparator_valid,
              "passed": comparator_valid and all(case["vq"]["passed"] for case in cases.values())}
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
