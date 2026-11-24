"""Render a frozen Ribeira capture in the common exposure defined by 15:03."""

from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
from PIL import Image

from probe_skin_spectrum_white import CMF, RGB_FROM_XYZ, SOURCES, checksum, to_srgb


BANDS = np.arange(400, 721, 10)
SHAPE = (1024, 1344, len(BANDS))
RELEASES = {
    "1503": (344_191_256, "d188f9b6bffcce3a6d2f91918fec76eb"),
    "1505": (344_081_358, "e7cb0d9af9958a801a74de3305e8b5d9"),
}


def read_xyz(source, frame, cmf):
    from scipy.io import loadmat

    size, md5 = RELEASES[frame]
    if source.stat().st_size != size or checksum(source, "md5") != md5:
        raise ValueError(f"Archive does not match the frozen {frame} release")
    if checksum(cmf, "md5") != SOURCES[CMF]:
        raise ValueError("Unexpected CIE 1931 2-degree observer table")
    data = np.loadtxt(cmf, delimiter=",")
    rows = data[np.searchsorted(data[:, 0], BANDS)]
    np.testing.assert_array_equal(rows[:, 0], BANDS)
    stem = f"Ribeira_Hotel_{frame}"
    with ZipFile(source) as archive, archive.open(f"{stem}/{stem}.mat") as member:
        radiance = loadmat(member, variable_names=["hsi"], verify_compressed_data_integrity=True)["hsi"]
        if member.read(1):
            raise ValueError(f"Unexpected data after hsi in capture {frame}")
    if radiance.shape != SHAPE or radiance.dtype.kind != "f" or not np.isfinite(radiance).all():
        raise ValueError(f"Unexpected spectral array in capture {frame}")
    return radiance @ (rows[:, 1:] * 10)


def shared_exposure(xyz):
    y = xyz[..., 1]
    positive = y[np.isfinite(y) & (y > 0)]
    if not positive.size:
        raise ValueError("The 15:03 reference has no positive finite Y")
    return float(.85 / np.quantile(positive, .99))


def display_rgb(xyz, exposure):
    linear = (xyz @ RGB_FROM_XYZ.T) * exposure
    return to_srgb(linear), linear < 0, linear > 1


def run(source, cmf, frame, out, anchor=None):
    if frame == "1505" and anchor is None:
        raise ValueError("Pass the prior 15:03 render directory as --anchor")
    if frame == "1503" and anchor is not None:
        raise ValueError("Only the 15:05 render takes --anchor")
    xyz = read_xyz(source, frame, cmf)
    cmf_sha = checksum(cmf)
    if anchor is None:
        exposure = shared_exposure(xyz)
        anchor_sha = None
    else:
        receipt = json.loads((anchor / "render.json").read_text(encoding="utf-8"))
        anchor_sha = checksum(anchor / "xyz.npy")
        if (receipt["source_md5"], receipt["observer_sha256"], receipt["xyz_sha256"]) != (
                RELEASES["1503"][1], cmf_sha, anchor_sha):
            raise ValueError("Invalid 15:03 exposure anchor")
        exposure = shared_exposure(np.load(anchor / "xyz.npy", allow_pickle=False))
        if exposure != receipt["exposure"]:
            raise ValueError("Exposure does not match the 15:03 render")
    rgb, below, above = display_rgb(xyz, exposure)
    out.mkdir()
    Image.fromarray(rgb).save(out / "srgb.png")
    np.save(out / "xyz.npy", xyz, allow_pickle=False)
    np.save(out / "out-of-gamut.npy", below | above, allow_pickle=False)
    evidence = dict(
        frame=frame, source_md5=RELEASES[frame][1], bands_nm=BANDS.tolist(), observer_sha256=cmf_sha,
        exposure=exposure, anchor_xyz_sha256=anchor_sha, xyz_sha256=checksum(out / "xyz.npy"),
        rgb_sha256=checksum(out / "srgb.png"), gamut_mask_sha256=checksum(out / "out-of-gamut.npy"),
        below_zero=below.sum(axis=(0, 1)).tolist(), above_one=above.sum(axis=(0, 1)).tolist(),
    )
    (out / "render.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cmf", type=Path)
    parser.add_argument("frame", choices=RELEASES)
    parser.add_argument("out", type=Path)
    parser.add_argument("--anchor", type=Path)
    run(**vars(parser.parse_args()))
