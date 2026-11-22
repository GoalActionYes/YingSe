"""Keep unbounded editor colors through a soft SDR gamut boundary."""
from argparse import ArgumentParser
from hashlib import sha256
from itertools import permutations
import json
import os
from pathlib import Path
import struct

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
from PIL import Image
import torch

from probe_arri_shared_look import metrics
from probe_gros_local_editor_cross import REGIONS, SIZE, areas_and_weight, compare, export_states
from probe_physical_blue_global_fit import display, linear
from probe_playraw_mixed_light_cross_editor import default_states
from probe_playraw_mixed_light_skin import INPUT_HASHES
from probe_playraw_mixed_light_transfer import LUMA, chroma
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel, save
from reference_color_app.lut import apply_mapping
from reference_color_app.pipeline import algorithm_fingerprint, analyze
from spatial_render import SpatialEngine, finish_spatial


PARENT = "3dd7043172870f6d79229cee012f39a0346f18b4c4dfc4e261dd6d1c4f83bb76"
KNEE = .80
LIMIT = .99
CHEEKS = ("cheek_left", "cheek_right")
SCREENS = ("screen_left", "screen_right")


def read_linear_exr(path):
    with path.open("rb") as file:
        header = file.read(4096)
    if not header.startswith(b"v/1\x01"):
        raise ValueError(f"Invalid EXR: {path}")
    for label in (b"channels\0chlist\0", b"chromaticities\0chromaticities\0"):
        offset = header.index(label) + len(label)
        size, = struct.unpack_from("<I", header, offset)
        payload = header[offset + 4:offset + 4 + size]
        if label.startswith(b"channels"):
            for name in (b"B", b"G", b"R"):
                channel, width = payload.split(b"\0", 1)
                if channel != name or struct.unpack_from("<I", width)[0] != 2:
                    raise ValueError(f"Expected 32-bit float RGB channels: {path}")
                payload = width[16:]
            if payload != b"\0":
                raise ValueError(f"Unexpected EXR channels: {path}")
        else:
            # darktable writes ICC PCS chromaticities, adapted to D50.
            expected = [.64844048, .33085853, .32117227, .59787625, .15589646, .06605524, .34570292, .35853860]
            np.testing.assert_allclose(struct.unpack("<8f", payload), expected, rtol=0, atol=1e-5)
    pixels = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.dtype != np.float32 or pixels.shape != (SIZE[1], SIZE[0], 3):
        raise ValueError(f"Unexpected linear image format: {path}")
    if not np.isfinite(pixels).all():
        raise ValueError(f"Nonfinite linear image: {path}")
    return pixels[..., ::-1].astype(np.float64)


def compress(rgb):
    y = (rgb @ LUMA)[..., None]
    if np.any((y < -1e-12) | (y > 1 + 1e-12)):
        raise ValueError("Edited linear luma is outside SDR")
    center = np.clip(y, 0, 1)
    offset = rgb - center
    available = np.where(offset >= 0, 1 - center, center)
    if np.any((available == 0) & (np.abs(offset) > 1e-12)):
        raise ValueError("No chroma headroom at edited luma")
    distance = np.divide(np.abs(offset), available, out=np.zeros_like(offset), where=available > 0).max(axis=-1)
    active = distance > KNEE
    mapped_distance = KNEE + (LIMIT - KNEE) * (-np.expm1(-(distance[active] - KNEE) / (LIMIT - KNEE)))
    result = rgb.copy()
    result[active] = center[active] + offset[active] * (mapped_distance / distance[active])[..., None]
    max_error = float(np.abs((result @ LUMA) - y[..., 0]).max())
    if max_error > 1e-12 or result.min() < -1e-12 or result.max() > 1 + 1e-12:
        raise ValueError("Gamut or linear luma invariant failed")
    return result, max_error, int(active.sum())


def summarize(original, expected, edited, regions):
    delta = np.abs(chroma(edited) - chroma(original)).mean(axis=-1)
    new_endpoints = np.any(((edited == 0) & (original > 0)) | ((edited == 255) & (original < 255)), axis=-1)
    area_stats = {}
    for name, mask in regions.items():
        record = metrics(original, expected, original, edited, mask)
        area_stats[name] = {**record, "pixels": int(mask.sum()), "chroma_median": float(np.median(delta[mask])),
                            "endpoint_fraction": float(new_endpoints[mask].mean())}
    return dict(fraction=float(new_endpoints.mean()), regions=area_stats)


def prepare(cli, source, settings, folder, regions, alpha, archived):
    old = {}
    for state, fingerprint in archived["hashes"].items():
        path = source / f"{state}.png"
        if digest(path) != fingerprint or (source / f"{state}.xmp").read_bytes() != settings[state]:
            raise ValueError(f"Changed frozen frame or author settings: {path}")
        with Image.open(path) as image:
            old[state] = np.array(image)
    with Image.open(source / "input.tif") as image:
        np.testing.assert_array_equal(np.asarray(image), old["off"])
    folder.mkdir(parents=True)
    exports = export_states(cli, source / "input.tif", settings.values(), folder, floating=True)
    frames = {state: read_linear_exr(path) for state, path in exports.items()}
    np.testing.assert_array_equal(display(frames["off"]), old["off"])
    selected = alpha > 0
    want = linear(old["off"][selected]) + alpha[selected, None] * (frames["on"] - frames["off"])[selected]
    luma = want @ LUMA
    valid = (luma >= -1e-12) & (luma <= 1 + 1e-12)
    invalid_luma = int((~valid).sum())
    rolloff = np.clip(want, 0, 1)
    rolloff[valid], luma_error, mapped_count = compress(want[valid])
    clip, smooth = old["off"].copy(), old["off"].copy()
    clip[selected], smooth[selected] = display(want), display(rolloff)
    old_blue = selected & (old["off"][..., 2] > 0) & (old["on"][..., 2] == 0)
    float_blue = frames["on"][..., 2][old_blue]
    measures = {name: summarize(old["off"], old["on"], image, regions)
                for name, image in (("prior", old["on"]), ("clip", clip), ("smooth", smooth))}
    soft = measures["smooth"]
    admitted = (invalid_luma == 0 and soft["fraction"] <= .02
                and all(soft["regions"][name]["pixels"] >= 128 for name in regions)
                and all(soft["regions"][name]["chroma_median"] >= 2
                        and .75 <= soft["regions"][name]["projection"] <= 1.25 for name in CHEEKS)
                and all(soft["regions"][name]["on_matches_authored_mae"] == 0 for name in SCREENS))
    np.testing.assert_array_equal(smooth[~selected], old["off"][~selected])
    output = {"off": old["off"], "on": smooth, "clip": clip}
    hashes = {state: save(folder / f"{state}.png", image) for state, image in output.items()}
    receipt = dict(qualified=admitted, max_luma_error=luma_error, mapped_pixels=mapped_count, hashes=hashes,
                   invalid_luma_pixels=invalid_luma, min_luma=float(luma.min()), max_luma=float(luma.max()),
                   stats=measures, old_blue_floor=int(old_blue.sum()), float_blue_negative=int((float_blue < 0).sum()),
                   panel=panel(folder / "comparison.png", {"sRGB": {"off": old["off"], "prior": old["on"],
                               "float clip": clip, "float smooth": smooth}}))
    return {"off": old["off"], "on": smooth}, receipt


def run(base, output):
    if any(not p.is_absolute() or not p.as_posix().isascii() for p in (base, output)):
        raise ValueError("Use absolute ASCII paths for the editor and OpenCV")
    source = base / "gros-local-editor-cross-v2"
    parent = checked_json(source / "results.json", PARENT)
    regions_path = Path(__file__).with_name("gros-local-editor-cross-regions.json")
    regions_bytes = regions_path.read_bytes().replace(b"\r\n", b"\n")
    if sha256(regions_bytes).hexdigest() != REGIONS:
        raise ValueError("Changed frozen regions")
    coords = json.loads(regions_bytes)
    root = base / "playraw-mixed-light-skin-v1"
    author, cli = root / "author.xmp", root / "runtime/bin/darktable-cli.exe"
    if digest(author) != INPUT_HASHES["author.xmp"] or digest(cli) != INPUT_HASHES["darktable-cli.exe"]:
        raise ValueError("Changed author settings or editor")
    settings = dict(zip(("off", "on"), default_states(author.read_bytes()), strict=True))
    output.mkdir()
    photos, regions, admission = {}, {}, {}
    for person, archived in parent["admission"].items():
        regions[person], alpha = areas_and_weight(coords[person], coords["screen"])
        photos[person], admission[person] = {}, {}
        for light, row in archived.items():
            photos[person][light], admission[person][light] = prepare(
                cli, source / person / light, settings, output / person / light, regions[person], alpha, row)
        print(person, {light: row["qualified"] for light, row in admission[person].items()}, flush=True)
    qualified = [person for person, rows in admission.items() if all(row["qualified"] for row in rows.values())]
    cases, identity = {}, {}
    if len(qualified) >= 2:
        torch.set_num_threads(4)
        engine = SpatialEngine("multiref-v1")
        if engine.checkpoint_sha != CHECKPOINT:
            raise ValueError("Changed model checkpoint")
        for person in qualified:
            rgb = photos[person]["cold"]["off"]
            path = output / person / "cold" / "off.png"
            np.testing.assert_array_equal(rgb, apply_mapping(rgb, analyze(path, path), STRENGTH))
            np.testing.assert_array_equal(rgb, finish_spatial(rgb, engine.prepare(rgb, rgb).apply(rgb), STRENGTH))
            identity[person] = {"stable": True, "spatial13": True}
        for subject, donor in permutations(qualified, 2):
            cases[f"{subject}-from-{donor}"] = compare(output, subject, donor, photos, regions[subject], engine)
            print(f"Rendered {subject} from {donor}", flush=True)
    json_to(output / "results.json", dict(parent_sha256=PARENT, admission=admission, qualified=qualified,
            cases=cases, identity=identity, checkpoint=CHECKPOINT, stable_fingerprint=algorithm_fingerprint(),
            strength=STRENGTH, protocol_sha256=digest(Path(__file__).with_name("gros-local-editor-float-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
