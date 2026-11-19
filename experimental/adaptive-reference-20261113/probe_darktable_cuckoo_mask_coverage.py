"""Export the author's mask and test the same edit at full coverage."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import subprocess
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from audit_playraw_sky_xmp import DT, RDF, payload
from render_rawtherapee_sky_pair import check

RAW_SHA = "7fbb1b6b9509e3ebb81aa9e76354c09ff1bf3002f465ef0364fcab6f8595304d"
CLI_SHA = "4db9777f5fee0800d8fbb6f18673635263182e7f2783ed4162a9dfb9ecd73a3a"
XMP_SHA = "578f776ad24571aded0878d286319b85c4f290b398ff18141f277858607332bb"
ON_SHA = "c17743f5eb6c81411f76e6c0258b2706fedea7491fe1980bd3214660b0c077f8"
OFF_SHA = "4104a99302814307ad560dfa7d2e9c1a6c2860af13d16db3a3d843f0880b0da0"
REGIONS = Path(__file__).with_name("darktable-cuckoo-legacy-regions.json")
LUMA = np.array([2126, 7152, 722])
MASK_NAMES = {"color balance rgb", "rgb 色彩平衡"}


def make_uniform(data):
    def step(content):
        row, = [node for node in ElementTree.fromstring(content).iter(RDF + "li")
                if node.get(DT + "operation") == "colorbalancergb"]
        if (row.get(DT + "num"), row.get(DT + "enabled"), row.get(DT + "blendop_version")) != ("10", "1", "11"):
            raise ValueError("Unexpected color balance step")
        return row.attrib

    original = step(data)
    token = original[DT + "blendop_params"]
    blob = payload(token)
    if len(blob) != 420 or struct.unpack_from("<I", blob)[0] != 5 or struct.unpack_from("<f", blob, 16)[0] != 100:
        raise ValueError("Unexpected author mask configuration")
    updated = struct.pack("<I", 1) + blob[4:]
    attribute = b'darktable:blendop_params="' + token.encode() + b'"'
    if data.count(attribute) != 1:
        raise ValueError("Expected one unique blend setting")
    answer = data.replace(attribute, b'darktable:blendop_params="' + updated.hex().encode() + b'"', 1)
    revised = step(answer)
    if revised.pop(DT + "blendop_params") != updated.hex() or original.pop(DT + "blendop_params") != token or revised != original:
        raise ValueError("Non-mask attributes changed")
    return answer


def render(cli, raw, author, out):
    for path in (cli, raw, author, out):
        if not path.is_absolute() or not path.as_posix().isascii():
            raise ValueError("Use absolute ASCII workspace paths")
    check(cli, CLI_SHA)
    check(raw, RAW_SHA)
    author_data = check(author, XMP_SHA)
    out.mkdir()
    uniform = out / "uniform.xmp"
    uniform.write_bytes(make_uniform(author_data))
    for label, sidecar, export_masks in (("author", author, "true"), ("uniform", uniform, "false")):
        output = out / f"{label}.tif"
        config, cache = out / f"config-{label}", out / f"cache-{label}"
        config.mkdir()
        cache.mkdir()
        args = [cli, raw, sidecar, output]
        args = [p.as_posix() for p in args] + [
            "--width", "1024", "--height", "1024", "--hq", "true", "--apply-custom-presets", "false",
            "--icc-type", "srgb", "--export_masks", export_masks, "--core", "--configdir", config.as_posix(),
            "--cachedir", cache.as_posix(), "--disable-opencl", "-t", "2", "--conf", "write_sidecar_files=never",
            "--conf", "plugins/imageio/format/tiff/bpp=8",
        ]
        with (out / f"{label}.log").open("wb") as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240, cwd=cli.parent)
        if not output.is_file():
            raise RuntimeError(f"No TIFF output for {label}")
        print(f"Exported {label}", flush=True)


def tiff(path, masked):
    with Image.open(path) as image:
        if image.format != "TIFF" or image.mode != "RGB" or image.size != (1024, 683):
            raise ValueError(f"Incompatible image: {path.name}")
        photo = np.array(image)
        if not masked:
            if image.n_frames != 1:
                raise ValueError("Unexpected pages in full-coverage control")
            return photo, None, []
        names, masks = [], []
        for index in range(1, image.n_frames):
            image.seek(index)
            name = str(image.tag_v2.get(285, ""))
            try:
                name = name.encode("latin1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass  # A decoder may already return the correct Unicode name.
            names.append(name)
            if name.casefold().strip() in MASK_NAMES:
                if image.mode != "RGB" or image.size != (1024, 683) or image.tag_v2.get(258) != (8, 8, 8):
                    raise ValueError("Unexpected RGB mask encoding")
                pixels = np.array(image)
                if not np.all(pixels == pixels[..., :1]):
                    raise ValueError("Mask channels differ")
                masks.append(pixels[..., 0])
        if len(masks) != 1:
            raise ValueError(f"Cannot identify a unique color balance mask: {names}")
        return photo, masks[0], names


def chroma_change(photo, original):
    delta = photo.astype(np.int64) - original
    light = (delta @ LUMA)[..., None]
    return np.abs(10000 * delta[..., [0, 2]] - light).mean(axis=-1) / 10000


def audit(archive, current, report):
    names = ("off", "on")
    pngs = []
    for name, expected in zip(names, (OFF_SHA, ON_SHA), strict=True):
        path = archive / f"{name}.png"
        check(path, expected)
        with Image.open(path) as image:
            pngs.append(np.array(image))
    original, known_on = pngs
    authored, mask, pages = tiff(current / "author.tif", True)
    uniform, _, _ = tiff(current / "uniform.tif", False)
    control = float(np.abs(authored.astype(np.int16) - known_on).mean())
    if control > .1:
        raise ValueError(f"TIFF did not reproduce the known author PNG: {control}")
    regions = json.loads(REGIONS.read_bytes()) | {"whole": [0, 0, 1024, 683]}
    signed = {"author": chroma_change(authored, original), "uniform": chroma_change(uniform, original)}
    measured = {}
    for name, (x, y, x1, y1) in regions.items():
        area = np.s_[y:y1, x:x1]
        measured[name] = {"weight_median": float(np.median(mask[area]) / 255),
                          "weight_at_least_half": float((mask[area] >= 128).mean()),
                          **{arm: float(np.median(values[area])) for arm, values in signed.items()}}
    chest = measured["chest"]
    result = {"pages": pages, "author_tiff_png_mae": control, "regions": measured,
              "mask_supported_explanation": chest["weight_median"] < .1 and chest["uniform"] >= 2,
              "uniform_environment_protected": all(measured[name]["uniform"] <= 1.5
                                                   for name in ("light_material", "green_background")),
              "hashes": {name: sha256((current / name).read_bytes()).hexdigest()
                         for name in ("author.tif", "uniform.tif", "uniform.xmp")}}
    with report.open("x", encoding="utf-8") as file:
        file.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name, args in (("render", ("cli", "raw", "author", "out")), ("audit", ("archive", "current", "report"))):
        sub = subs.add_parser(name)
        for arg in args:
            sub.add_argument(arg, type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    (render if command == "render" else audit)(**args)
