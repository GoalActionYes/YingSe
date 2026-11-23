"""Reproduce fixed human local edit switches on one mixed-light raw photo."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import os
import re
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

from audit_playraw_sky_xmp import DT, RDF, payload
from probe_playraw_mixed_light_skin import export
from render_rawtherapee_sky_pair import check, disabled, render


RAW = "d75f10c550d281e6fb9aa3b91c23a4b83ec8feafd2f10b3e37966e7c6e9c9049"
XMP = "db18485e080aac1e1e63b038adc39323130a18edb244bd23e799243bfd52457f"
PP3 = "e719c120855b13c777299608f36040c4dc5f4d6d83ad9a38fefeb4e63941526f"
DT_CLI = "3b0883da74c21d7deb83de8cc60a1d496d87b67c6f4b7d38c9c86a6f9aaaaacb"
RT_CLI = "9172a03896f0200daf621909b024f6cd65a24837cad56d0748730b2fa6d2fb6a"
LUMA = np.array([2126, 7152, 722]) / 10000


def darktable_off(source):
    rows = [node for node in ET.fromstring(source).iter(RDF+"li") if node.get(DT+"operation") == "colorzones"]
    row, = rows
    if (row.get(DT+"num"), row.get(DT+"enabled"), row.get(DT+"blendop_version")) != ("16", "1", "13"):
        raise ValueError("Unexpected author's color zones history")
    if int.from_bytes(payload(row.get(DT+"blendop_params"))[:4], "little") != 3:
        raise ValueError("Expected author's drawn mask")
    tag, = (item for item in re.finditer(rb"<rdf:li\s+[^>]*>", source) if b'darktable:num="16"' in item[0])
    replacement = tag[0].replace(b'darktable:enabled="1"', b'darktable:enabled="0"')
    off = source[:tag.start()] + replacement + source[tag.end():]
    if sum(a != b for a,b in zip(source,off,strict=True)) != 1:
        raise ValueError("Expected only one changed byte")
    return off


def render_state(source, cli, out, editor, state):
    if any(not p.is_absolute() or not p.as_posix().isascii() for p in (source,cli,out)):
        raise ValueError("Expected absolute ASCII paths for editor runtime")
    raw = source/"DSCF8379.raf"
    check(raw, RAW)
    out.mkdir()
    if editor == "darktable":
        check(cli, DT_CLI)
        original = check(source/"terry.xmp", XMP)
        profile = out/"settings.xmp"
        profile.write_bytes(darktable_off(original) if state == "off" else original)
        env = {key:value for key,value in os.environ.items() if key != "OMP_NUM_THREADS"}
        export(cli, raw, profile, out, "image", state == "on", env=env)
    else:
        check(cli, RT_CLI)
        original = check(source/"martin.pp3", PP3)
        profile = out/"settings.pp3"
        profile.write_bytes(disabled(original) if state == "off" else original)
        env = dict(os.environ, RT_SETTINGS=str(out/"rt-settings"), RT_CACHE=str(out/"rt-cache"), OMP_NUM_THREADS="1")
        render(cli, raw, profile, out/"image.png", env)


def read(path, with_mask):
    with Image.open(path) as file:
        if file.mode != "RGB" or not file.info.get("icc_profile"):
            raise ValueError("Expected RGB and an embedded profile")
        rgb = np.array(file)
        icc = sha256(file.info["icc_profile"]).hexdigest()
        mask, label = None, None
        if with_mask:
            if file.n_frames != 2:
                raise ValueError("Expected exactly one rendered mask")
            file.seek(1)
            label = str(file.tag_v2.get(285, ""))
            try:
                label = label.encode("latin1").decode("utf-8")
            except (UnicodeEncodeError,UnicodeDecodeError):
                pass
            mask = np.array(file)
            if mask.ndim == 3:
                if not (mask[..., :1] == mask).all():
                    raise ValueError("Mask channels differ")
                mask = mask[..., 0]
            if mask.shape != rgb.shape[:2] or mask.dtype != np.uint8:
                raise ValueError("Unexpected mask geometry or depth")
        return rgb, mask, dict(icc_sha256=icc, pixels_sha256=sha256(rgb.tobytes()).hexdigest(),
                               mask_label=label, mask_sha256=sha256(mask.tobytes()).hexdigest() if mask is not None else None)


def audit(base, out, editor, regions):
    frames, receipts = {}, {}
    for label in ("off", "on", "off-repeat", "on-repeat"):
        masked = editor == "darktable" and label.startswith("on")
        path = base/label/("image.tif" if editor == "darktable" else "image.png")
        img, mask, receipt = read(path, masked)
        frames[label] = img, mask
        receipts[label] = receipt
    before, _ = frames["off"]
    after, mask = frames["on"]
    if before.shape != after.shape or len({info["icc_sha256"] for info in receipts.values()}) != 1:
        raise ValueError("Changed image geometry or profile")
    repeat = {}
    for state in ("off", "on"):
        original, original_mask = frames[state]
        repeated, repeated_mask = frames[state+"-repeat"]
        count = np.abs(original.astype(np.int16)-repeated)
        repeat[state] = dict(max=int(count.max()), changed_values=int(np.count_nonzero(count)),
                             mask_equal=np.array_equal(original_mask,repeated_mask))
    delta = after.astype(float) - before
    brightness = delta @ LUMA
    chroma = np.abs(delta[..., [0,2]] - brightness[..., None]).mean(axis=-1)
    rgb_delta = np.abs(delta).mean(axis=-1)
    if editor == "darktable":
        zones = {"wanted":mask>=204,"protected":mask==0}
    else:
        cfg = json.loads(regions.read_bytes())
        if cfg["off_pixels_sha256"] != receipts["off"]["pixels_sha256"]:
            raise ValueError("Scoring regions were fixed on a different image")
        zones = {}
        for name,(x0,y0,x1,y1) in cfg["rectangles"].items():
            if not (0 <= x0 < x1 <= before.shape[1] and 0 <= y0 < y1 <= before.shape[0]):
                raise ValueError("Scoring box outside the rendered image")
            rect = np.zeros(before.shape[:2],dtype=bool)
            rect[y0:y1,x0:x1] = True
            zones[name] = rect
        if not set(zones) >= {"wanted","protected"}:
            raise ValueError("Expected wanted and protected boxes")
    effect = chroma if editor == "darktable" else brightness
    measurements = {}
    for name,zone in zones.items():
        change = effect[zone]
        measurements[name] = dict(pixels=int(zone.sum()), median=float(np.median(change)) if change.size else None,
            primary_abs_p95=float(np.percentile(np.abs(change),95)) if change.size else None,
            rgb8_change_p95=float(np.percentile(rgb_delta[zone],95)) if change.size else None)
    wanted = measurements["wanted"]
    protected = [value for key,value in measurements.items() if key != "wanted"]
    qualified = (min(value["pixels"] for value in measurements.values()) >= 128 and wanted["median"] >= 2
                 and max(value["primary_abs_p95"] for value in protected) <= 1.5)
    out.open("x",encoding="utf-8").write(json.dumps(dict(editor=editor, image_shape=before.shape, receipts=receipts,
        repeat=repeat, regions=measurements, qualified=qualified), indent=2, allow_nan=False)+"\n")
    print(editor, qualified, measurements, repeat)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    output = commands.add_parser("render")
    for key in ("source", "cli", "out"):
        output.add_argument(key,type=Path)
    output.add_argument("editor",choices=("darktable","rawtherapee"))
    output.add_argument("state",choices=("off","on"))
    score = commands.add_parser("audit")
    score.add_argument("base",type=Path)
    score.add_argument("out",type=Path)
    score.add_argument("editor",choices=("darktable","rawtherapee"))
    score.add_argument("--regions",type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    (render_state if action == "render" else audit)(**args)
