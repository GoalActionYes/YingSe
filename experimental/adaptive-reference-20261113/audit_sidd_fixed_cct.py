"""Verify the two fixed SIDD bursts against the archive exposure index."""
from argparse import ArgumentParser
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import zlib

import numpy as np
from PIL import Image
import rawpy

from fetch_sidd_fixed_cct import ARCHIVES, LIMIT, REVISION
from probe_synclight_real_pixels import registration, shot, srgb8


PROTOCOL = "d1eadce654c1c07da6171f65a9b408a43d38aa2b5395fd544d766ea52676dafe"
EXPECTED = {"0028": 5500, "0033": 3200}


def read_camera(path):
    with Image.open(path) as image:
        tags = image.getexif()
        camera = tags.get_ifd(34665)
        shutter = camera[33434]
        return dict(make=tags[271], model=tags[272], orientation=tags[274],
                    shutter=[shutter.numerator, shutter.denominator], iso=camera[34855],
                    aperture=float(camera[33437]), focal_mm=float(camera[37386]),
                    wb_mode=camera.get(41987), original_time=camera.get(36867),
                    as_shot_neutral=[float(value) for value in tags[50728]],
                    baseline_exposure=float(tags[50730]))


def verified_paths(folder, receipt):
    if (receipt["revision"] != REVISION or receipt["bytes_transferred"] > LIMIT
            or set(receipt["scenes"]) != set(EXPECTED)):
        raise ValueError("Changed SIDD archive receipt")
    for instance, source in receipt["scenes"].items():
        if (source["archive_bytes"], source["linked_sha256"], source["delivery_etag"]) != ARCHIVES[instance]:
            raise ValueError(f"Archive identity mismatch: {instance}")
        names = [item["filename"] for item in source["members"]]
        if len(names) != 2 or len(set(names)) != 2:
            raise ValueError(f"Missing fixed SIDD frames: {instance}")
        for item in source["members"]:
            filename = item["filename"]
            if Path(filename).name != filename or not filename.startswith(f"{instance}_IMG_"):
                raise ValueError("Invalid SIDD filename")
            path = folder / filename
            data = path.read_bytes()
            if (len(data) != item["bytes"] or sha256(data).hexdigest() != item["sha256"]
                    or f"{zlib.crc32(data):08x}" != item["crc32"]):
                raise ValueError(f"Changed DNG: {filename}")
            yield instance, path


def sensor_medians(raw):
    pixels = raw.raw_image_visible
    # Use the visible sensor samples before white balance, demosaicing or color conversion.
    return [float(np.median(pixels[y::2, x::2])) for y, x in ((0, 0), (0, 1), (1, 0), (1, 1))]


def audit(source, out):
    script = Path(__file__)
    protocol = script.with_name("sidd-fixed-cct-intake-protocol.md")
    if sha256(protocol.read_bytes()).hexdigest() != PROTOCOL or rawpy.__version__ != "0.28.0":
        raise ValueError("Unexpected SIDD protocol or LibRaw wrapper")
    receipt_bytes = (source / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    sources = list(verified_paths(source, receipt))
    with rawpy.imread(str(sources[0][1])) as anchor:
        fixed_wb = list(anchor.camera_whitebalance)

    out.mkdir()
    frames, audits = {}, {}
    for instance, path in sources:
        rgb, geometry, clip = shot(rawpy, path, fixed_wb)
        preview = srgb8(rgb)
        rgb_path = out / f"{path.stem}.png"
        Image.fromarray(preview).save(rgb_path)
        camera = read_camera(path)
        with rawpy.imread(str(path)) as raw:
            counts = sensor_medians(raw)
        index_match = (camera["model"] == "iPhone 7" and camera["iso"] == 100
                       and 160 * camera["shutter"][0] == camera["shutter"][1])
        frames[path.stem] = instance, rgb, preview.astype(np.int16)
        audits[path.stem] = dict(instance=instance, camera=camera, fixed_exposure_index_match=index_match,
                                 geometry=geometry, render_clipping=clip, pre_demosaic_medians_rggb=counts,
                                 preview_pixels_sha256=sha256(preview.tobytes()).hexdigest())

    comparisons = {}
    for (a, (inst_a, linear_a, first)), (b, (inst_b, linear_b, second)) in combinations(frames.items(), 2):
        delta = second - first
        comparisons[f"{a}--{b}"] = dict(same_archive=inst_a == inst_b, alignment=registration(linear_a, linear_b),
                                         mean_abs_rgb8=float(np.abs(delta).mean()),
                                         median_signed_rgb8=np.median(delta, axis=(0, 1)).tolist(),
                                         mean_signed_rgb8=delta.mean(axis=(0, 1)).tolist())
    result = dict(protocol_sha256=PROTOCOL, audit_sha256=sha256(script.read_bytes()).hexdigest(),
                  receipt_sha256=sha256(receipt_bytes).hexdigest(), archive=receipt,
                  expected_index=dict(shot_seconds=[1, 160], iso=100, cct_kelvin=EXPECTED),
                  decoder=rawpy.__version__, rawpy_libraw=list(rawpy.libraw_version),
                  fixed_wb=fixed_wb, frames=audits, comparisons=comparisons,
                  exposure_index_matches=sum(row["fixed_exposure_index_match"] for row in audits.values()))
    with (out / "results.json").open("w", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(f"{result['exposure_index_matches']}/4 frames agree with the index", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    audit(args.source, args.out)
