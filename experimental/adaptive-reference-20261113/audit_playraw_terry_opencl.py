"""Verify the native author switch and locate clipping in the four fixed OpenCL renders."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageCms

from probe_playraw_flash_room import LUMA, RAW, XMP, darktable_off, read
from probe_playraw_terry_opencl import DT50
from probe_target_mirror_chroma import digest, json_to, panel, save


REGIONS = "ca20d48d7e5652a27d750eecd46afed903524010bacb69450a2e950c2f4e1134"
STATES = ("off", "off-repeat", "on", "on-repeat")
STEP = re.compile(r"process\s+(CPU|CL\d+)\s+\[export\]\s+(\w+)\s+\d+")


def load_render(folder, state, author):
    receipt = json.loads((folder / "render.json").read_bytes())
    path = folder / "image.tif"
    log = folder / "darktable.log"
    enabled = state.startswith("on")
    expected = author if enabled else darktable_off(author)
    if ((folder / "settings.xmp").read_bytes() != expected or receipt["settings_sha256"] != sha256(expected).hexdigest()
            or receipt["exit_code"] != "0x00000000" or not receipt["render_completed"]
            or (receipt["author_xmp_sha256"], receipt["raw_sha256"], receipt["renderer_sha256"]) != (XMP, RAW, DT50)
            or digest(path) != receipt["output"]["tiff_sha256"] or digest(log) != receipt["log_sha256"]):
        raise ValueError(f"Unverified renderer output: {state}")
    trace = log.read_text(encoding="utf-8", errors="replace")
    modules = STEP.findall(trace)
    mosaic = [line for line in trace.splitlines() if "demosaic inplace" in line]
    if (len(mosaic) != 1 or "CL0 [export]" not in mosaic[0] or mosaic[0].count("6240x4156 scale=1.0000") != 2
            or [backend for backend, name in modules if name == "demosaic"] != ["CL0"]
            or "demosaic approx zoom" in trace or "NVIDIA GeForce RTX 4060 Laptop GPU" not in trace):
        raise ValueError(f"Demosaic did not use the native OpenCL path: {state}")
    if (("CL0", "colorzones") in modules) != enabled:
        raise ValueError(f"Unexpected Color Zones backend or state: {state}")
    rgb, mask, identity = read(path, enabled)
    with Image.open(path) as image:
        if tuple(image.tag_v2[258]) != (8, 8, 8) or image.size != (1024, 682):
            raise ValueError("Changed pixel format")
        profile = image.info["icc_profile"]
        profile_name = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(BytesIO(profile))).strip()
        if profile_name.casefold() != "srgb":
            raise ValueError("Changed output color space")
        # The ICC creation timestamp is not part of the color transform.
        profile_transform = sha256(profile[:24] + profile[36:]).hexdigest()
        mask_name = None
        if enabled:
            image.seek(1)
            mask_name = image.tag_v2._tagdata[285].rstrip(b"\0").decode("utf-8")
            if mask_name.strip().casefold() not in ("色彩空间", "color zones"):
                raise ValueError(f"Unexpected rendered mask: {mask_name}")
        elif image.n_frames != 1:
            raise ValueError("Unexpected mask in disabled render")
    if identity["pixels_sha256"] != receipt["output"]["pixels_sha256"]:
        raise ValueError("Changed rendered pixels")
    ledger = dict(pixel_sha256=identity["pixels_sha256"], tiff_sha256=digest(path),
                  icc_sha256=identity["icc_sha256"], profile_transform=profile_transform,
                  mask_sha256=identity["mask_sha256"], mask_label=mask_name,
                  settings_sha256=receipt["settings_sha256"], log_sha256=receipt["log_sha256"],
                  demosaic_path="OpenCL full 6240x4156",
                  common_backends=[step for step in modules if step[1] != "colorzones"])
    return rgb, mask, ledger


def measurements(original, edited, mask, coordinates, out):
    delta = edited.astype(np.int16) - original
    warmth = (delta[..., 0] - delta[..., 2]) / 2
    y_delta = delta @ LUMA
    endpoints_low = (edited == 0) & (original > 0)
    endpoints_high = (edited == 255) & (original < 255)
    endpoint = (endpoints_low | endpoints_high).any(axis=2)
    regions = {}
    for name, (left, top, right, bottom) in coordinates.items():
        if not (0 <= left < right <= original.shape[1] and 0 <= top < bottom <= original.shape[0]):
            raise ValueError(f"Region outside the frozen planning frame: {name}")
        area = np.s_[top:bottom, left:right]
        regions[name] = dict(warmth_median=float(np.median(warmth[area])),
                             abs_warmth_p95=float(np.percentile(np.abs(warmth[area]), 95)),
                             luma_median=float(np.median(y_delta[area])),
                             luma_abs_p95=float(np.percentile(np.abs(y_delta[area]), 95)),
                             rgb_mae=float(np.abs(delta[area]).mean()),
                             pixel_identical=bool(np.array_equal(original[area], edited[area])),
                             mask_median=float(np.median(mask[area])),
                             endpoint_pixels=int(endpoint[area].sum()))
    marked = np.where(endpoint[:, :, None], np.array([255, 35, 105], np.uint8), original // 3)
    board = panel(out / "comparison.png", {
        "source": {"Color Zones off": original, "Color Zones on": edited},
        "audit": {"authored mask": np.repeat(mask[:, :, None], 3, axis=2), "new endpoints": marked},
    })
    boxes = {}
    y, x = np.nonzero(endpoint)
    if len(y):
        boxes["extent"] = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
    clipped = dict(fraction=float(endpoint.mean()), pixels=int(endpoint.sum()),
                   new_zero=endpoints_low.sum(axis=(0, 1)).tolist(), new_full=endpoints_high.sum(axis=(0, 1)).tolist(),
                   new_zero_original_medians=[float(np.median(original[:, :, i][endpoints_low[:, :, i]]))
                                              if endpoints_low[:, :, i].any() else None for i in range(3)],
                   in_author_mask=int((endpoint & (mask > 0)).sum()), **boxes)
    return regions, clipped, board


def run(base, source, out):
    author = (source / "terry.xmp").read_bytes()
    if sha256(author).hexdigest() != XMP or digest(source / "DSCF8379.raf") != RAW:
        raise ValueError("Changed frozen author input")
    registry = Path(__file__).with_name("playraw-terry-opencl-regions.json")
    if digest(registry) != REGIONS:
        raise ValueError("Changed preregistered measurement boxes")
    plan = json.loads(registry.read_bytes())
    records = {state: load_render(base / state, state, author) for state in STATES}
    original, _, planning = records["off"]
    edited, mask, _ = records["on"]
    if planning["pixel_sha256"] != plan["off_pixels_sha256"]:
        raise ValueError("Scoring unregistered original")
    identities = {state: row[2] for state, row in records.items()}
    if len({row["profile_transform"] for row in identities.values()}) != 1 or any(
            row["common_backends"] != planning["common_backends"] for row in identities.values()):
        raise ValueError("Profile transform or shared processing path changed")
    repeat = {state: int(np.abs(records[state][0].astype(np.int16) - records[state + "-repeat"][0]).max())
              for state in ("off", "on")}
    mask_repeated = bool(np.array_equal(mask, records["on-repeat"][1]))
    out.mkdir()
    regions, endpoints, board = measurements(original, edited, mask, plan["rectangles"], out)
    qualified = (regions["wanted"]["warmth_median"] >= 2 and regions["protected"]["abs_warmth_p95"] <= 1.5
                 and endpoints["fraction"] <= .02 and max(repeat.values()) <= 1 and mask_repeated)
    frames = {key: save(out / (key + ".png"), records[key][0]) for key in ("off", "on")}
    code = Path(__file__)
    json_to(out / "results.json", dict(qualified=qualified, frames=frames, board=board,
            repeats=repeat, repeated_mask=mask_repeated, identities=identities, endpoints=endpoints,
            regions=regions, regions_sha256=REGIONS, script_sha256=digest(code),
            protocol_sha256=digest(code.with_name("playraw-terry-opencl-protocol.md"))))
    print(f"Qualified: {qualified}, new endpoints: {endpoints['pixels']} ({100 * endpoints['fraction']:.3f}%)")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("base", "source", "out"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
