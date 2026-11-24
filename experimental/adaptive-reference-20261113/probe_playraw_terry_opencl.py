"""Render a fixed author's XMP with darktable 5.0.0's OpenCL backend."""
from argparse import ArgumentParser
from hashlib import sha256
import os
from pathlib import Path
import subprocess

from probe_playraw_flash_room import RAW, XMP, darktable_off, read
from probe_target_mirror_chroma import digest, json_to


DT50 = "411578d4a3d073ec5ad711ef351a8888d4793dd21f7d66e87a07021ace912f90"


def read_16(path, with_mask):
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(path) as tiff:
        if tiff.n_frames != (2 if with_mask else 1):
            raise ValueError("Unexpected 16-bit TIFF page count")
        profile = tiff.info["icc_profile"]
        label = None
        size = (tiff.height, tiff.width, 3)
        for page in range(tiff.n_frames):
            tiff.seek(page)
            if (tuple(tiff.tag_v2[258]) != (16, 16, 16)
                    or tuple(tiff.tag_v2.get(339, (1, 1, 1))) != (1, 1, 1)
                    or tiff.tag_v2.get(274, 1) != 1 or tiff.tag_v2[262] != 2):
                raise ValueError("Expected top-left RGB16 unsigned TIFF samples")
            if page:
                label = tiff.tag_v2._tagdata[285].rstrip(b"\0").decode("utf-8")
                if label.strip().casefold() not in ("色彩空间", "color zones"):
                    raise ValueError("Unexpected 16-bit mask label")
    success, pages = cv2.imdecodemulti(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if not success or len(pages) != (2 if with_mask else 1) or any(
            image.dtype != np.uint16 or image.shape != size for image in pages):
        raise ValueError("Decoder did not retain full RGB16 precision")
    rgb = np.ascontiguousarray(pages[0][:, :, ::-1])
    mask = None
    if with_mask:
        if not (pages[1] == pages[1][..., :1]).all():
            raise ValueError("Mask is not grayscale")
        mask = pages[1][..., 0].copy()
    return rgb, mask, dict(icc_sha256=sha256(profile).hexdigest(), pixels_sha256=sha256(rgb.tobytes()).hexdigest(),
                           mask_label=label, mask_sha256=sha256(mask.tobytes()).hexdigest() if with_mask else None)


def render(source, cli, out, state, cache_dir=None, bits=8):
    paths = (source, cli, out) if cache_dir is None else (source, cli, out, cache_dir)
    if any(not p.is_absolute() or not str(p).isascii() for p in paths):
        raise ValueError("Use an isolated ASCII drive path")
    raw = source / "DSCF8379.raf"
    xmp = source / "terry.xmp"
    if (digest(raw), digest(xmp), digest(cli)) != (RAW, XMP, DT50):
        raise ValueError("Changed raw, author XMP or darktable binary")
    original = xmp.read_bytes()
    out.mkdir()
    config = out / "config"
    config.mkdir()
    cache = out / "cache" if cache_dir is None else cache_dir
    if cache_dir is None:
        cache.mkdir()
    elif not cache.is_dir():
        raise ValueError("Expected existing isolated cache")
    profile = out / "settings.xmp"
    profile.write_bytes(darktable_off(original) if state == "off" else original)
    destination = out / "image.tif"
    options = [
        "--width", "1024", "--height", "1024", "--hq", "true", "--apply-custom-presets", "false",
        "--icc-type", "srgb", "--export_masks", str(state == "on").lower(), "--core",
        "--configdir", config.as_posix(), "--cachedir", cache.as_posix(), "--library", ":memory:",
        "-t", "2", "-d", "opencl", "-d", "pipe", "--conf", "opencl=TRUE",
        "--conf", "write_sidecar_files=never", "--conf", f"plugins/imageio/format/tiff/bpp={bits}",
    ]
    command = [p.as_posix() for p in (cli, raw, profile, destination)] + options
    env = {key: value for key, value in os.environ.items() if key != "OMP_NUM_THREADS"}
    logfile = out / "darktable.log"
    with logfile.open("wb") as log:
        done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=130, cwd=cli.parent, env=env)
    lines = logfile.read_text(encoding="utf-8", errors="replace").splitlines()
    relevant = [line.strip() for line in lines if "demosaic" in line.lower()]
    rendered = bool(done.returncode == 0 and destination.is_file())
    identity = {}
    if rendered:
        reader = read if bits == 8 else read_16
        rgb, _, identity = reader(destination, state == "on")
        if max(rgb.shape[:2]) != 1024:
            raise ValueError("Changed export dimensions")
        identity = dict(identity, size=[rgb.shape[1], rgb.shape[0]], tiff_sha256=digest(destination))
    receipt = dict(state=state, render_completed=rendered, exit_code=f"0x{done.returncode & 0xffffffff:08X}",
                   raw_sha256=RAW, author_xmp_sha256=XMP, renderer_sha256=DT50,
                   settings_sha256=digest(profile), output=identity, log_sha256=digest(logfile),
                   demosaic_lines=relevant[-16:])
    json_to(out / "render.json", receipt)
    print(state, receipt["exit_code"], rendered, flush=True)
    for line in relevant[-5:]:
        print(line, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("source", "cli", "out"):
        parser.add_argument(key, type=Path)
    parser.add_argument("state", choices=("off", "on"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--bits", type=int, choices=(8, 16), default=8)
    render(**vars(parser.parse_args()))
