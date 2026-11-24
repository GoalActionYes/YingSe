"""Render one SFU plastic-1 capture using the fixed 4100 K display scale."""

from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from tarfile import open as open_tar
import json

import numpy as np
from PIL import Image

from probe_skin_spectrum_white import checksum
from render_ribeira_daylight import display_rgb, shared_exposure


LIGHTS = ("3500", "4100", "4700")
SCENE = "specular_8_bit/plastic-1/plastic-1_solux-"
WAVELENGTHS = np.arange(380, 781, 4)
ARCHIVE_SHA = "67d47f4eae5b4708d69e4ed496a9ddd8e572fc26a5d1f1962309a2e77dce2820"
SENSOR_SHA = "9eaa31aa471eae2c47b7488eb87f6b3021d332db86fa8773b5a98e3e5bf17b03"
CALIBRATION_SHA = "b857e063c9bdd18db18acf7098679dc8ef204904a4a9ef5c467b736d89b2334f"
OBSERVER_SHA = "fa663e3535a7e0763a745993a1f0a192eb0275ac46ad2d1befd7626841e713c1"


def fit_camera_xyz(camera, xyz):
    y = xyz[:, 1:2]
    if not np.all(y > 0):
        raise ValueError("Calibration requires positive reference luminance")
    rgb_to_xyz, _, rank, _ = np.linalg.lstsq(camera / y, xyz / y, rcond=None)
    if rank != 3 or not np.isfinite(rgb_to_xyz).all():
        raise ValueError("Calibration does not resolve all three channels")
    return rgb_to_xyz


def run(archive_path, sensor_path, spectra_path, observer_path, light, out, anchor=None):
    if (light == "4100") != (anchor is None):
        raise ValueError("Pass --anchor for other lights only, using the prior 4100 K output")
    expected = dict(archive=ARCHIVE_SHA, sensor=SENSOR_SHA, spectra=CALIBRATION_SHA, observer=OBSERVER_SHA)
    paths = dict(archive=archive_path, sensor=sensor_path, spectra=spectra_path, observer=observer_path)
    if any(checksum(paths[key]) != value for key, value in expected.items()):
        raise ValueError("Input identity differs from the frozen SFU release or calibration")
    table = np.loadtxt(observer_path, delimiter=",")
    cmf = table[np.searchsorted(table[:, 0], WAVELENGTHS)]
    np.testing.assert_array_equal(cmf[:, 0], WAVELENGTHS)
    observer = cmf[:, 1:]
    sensor = np.loadtxt(sensor_path).reshape(3, 101).T
    spectra = np.loadtxt(spectra_path).reshape(598, 101)
    camera_to_xyz = fit_camera_xyz(spectra @ sensor, spectra @ observer)
    white = {}
    with open_tar(archive_path, "r:gz") as archive:
        for setting in LIGHTS:
            with archive.extractfile(f"{SCENE}{setting}.rgb") as stream:
                camera_white = np.loadtxt(stream)
            with archive.extractfile(f"{SCENE}{setting}.illum") as stream:
                radiance = np.loadtxt(stream)
            from_spectrum = radiance @ observer
            from_camera = camera_white @ camera_to_xyz
            white[setting] = dict(
                camera_rgb=camera_white.tolist(), predicted_rgb=(radiance @ sensor).tolist(),
                spectral_xyz=from_spectrum.tolist(), mapped_camera_xyz=from_camera.tolist(),
            )
        with archive.extractfile(f"{SCENE}{light}.tif") as stream:
            source = stream.read()
    with Image.open(BytesIO(source)) as image:
        if image.format != "TIFF" or image.mode != "RGB" or image.size != (637, 468):
            raise ValueError("Unexpected SFU frame format")
        if image.tag_v2.get(258) != (8, 8, 8) or image.info.get("icc_profile"):
            raise ValueError("Unexpected SFU bit depth or color management")
        camera_pixels = np.array(image)
    xyz = camera_pixels @ camera_to_xyz
    anchor_hash = None
    if anchor is None:
        exposure = shared_exposure(xyz)
    else:
        receipt = json.loads((anchor / "render.json").read_text(encoding="utf-8"))
        anchor_hash = checksum(anchor / "xyz.npy")
        if (receipt["light"], receipt["source_hashes"], receipt["xyz_sha256"]) != ("4100", expected, anchor_hash):
            raise ValueError("Wrong 4100 K anchor")
        exposure = shared_exposure(np.load(anchor / "xyz.npy", allow_pickle=False))
        if receipt["exposure"] != exposure:
            raise ValueError("Unexpected 4100 K display scale")
    rgb, below, above = display_rgb(xyz, exposure)
    out.mkdir()
    (out / "capture.tif").write_bytes(source)
    Image.fromarray(rgb).save(out / "srgb.png")
    np.save(out / "xyz.npy", xyz, allow_pickle=False)
    np.save(out / "out-of-gamut.npy", below | above, allow_pickle=False)
    receipt = dict(
        light=light, source_hashes=expected, white=white, exposure=exposure, anchor_xyz_sha256=anchor_hash,
        camera_to_xyz=camera_to_xyz.tolist(), capture_sha256=checksum(out / "capture.tif"),
        rgb_sha256=checksum(out / "srgb.png"), xyz_sha256=checksum(out / "xyz.npy"),
        gamut_mask_sha256=checksum(out / "out-of-gamut.npy"),
    )
    (out / "render.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("archive_path", "sensor_path", "spectra_path", "observer_path"):
        parser.add_argument(field, type=Path)
    parser.add_argument("light", choices=LIGHTS)
    parser.add_argument("out", type=Path)
    parser.add_argument("--anchor", type=Path)
    run(**vars(parser.parse_args()))
