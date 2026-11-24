"""Render a fixed author's XMP with darktable 5.0.0's OpenCL backend."""
from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess

from probe_playraw_flash_room import RAW, XMP, darktable_off, read
from probe_target_mirror_chroma import digest, json_to


DT50 = "411578d4a3d073ec5ad711ef351a8888d4793dd21f7d66e87a07021ace912f90"


def render(source, cli, out, state, cache_dir=None):
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
        "--conf", "write_sidecar_files=never", "--conf", "plugins/imageio/format/tiff/bpp=8",
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
        rgb, _, identity = read(destination, state == "on")
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
    render(**vars(parser.parse_args()))
