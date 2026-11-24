"""Check frozen fixed-daylight LSMI inputs against the actual local warmth edit."""
from argparse import ArgumentParser
from pathlib import Path
import os

import numpy as np
from PIL import Image

from fetch_lsmi_daylight_switch import INDEX, LIGHTS, SCENES
from probe_chromamill_blue_white_warmth import aligned
from probe_inretouch_registration import gradient, shifts
from probe_playraw_flash_room import PP3, RT_CLI
from probe_shared_authored_tone_admission import support
from probe_shared_local_warmth import WARMTH_FIELDS, prepare, rectangles
from probe_target_mirror_chroma import checked_json, digest, json_to
from render_rawtherapee_sky_pair import check
from triplet_color import to_srgb


PROTOCOL = "9aac01cf42f7e2b079b420d3cb8604653af877de58f5a9e1594f9b9f8d75e192"
REGIONS = "52539be479e54c371b742a8cdd4478c07e1bcabf7e624a1ade40c482e1d7f6b9"
RECEIPT = "d0f980fcb2331ed917723b577e62357c94354c5246de9d2f6fb46e423c7c0c74"


def capture(path, source):
    if digest(path) != source["sha256"]:
        raise ValueError(f"Unexpected LSMI image: {path}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (920, 614):
            raise ValueError(f"Unexpected LSMI resolution: {path}")
        rgb, info = to_srgb(image)
        tags = image.getexif().get_ifd(34665)
        info["capture_exif"] = {key: str(tags[tag]) if tag in tags else None
                                for key, tag in (("shutter", 33434), ("aperture", 33437), ("iso", 34855),
                                                 ("white_balance", 41987))}
    return np.array(rgb), info


def physical_shift(first, second, area):
    x, y = first[area], second[area]
    valid = support(x) & support(y)
    shift = np.median(y[valid], axis=0) - np.median(x[valid], axis=0) if valid.any() else None
    return dict(coverage=float(valid.mean()), median_rgb8=shift.tolist() if shift is not None else None)


def run(base, cli, out):
    code = Path(__file__)
    layout = code.with_name("lsmi-daylight-switch-warmth-regions.json")
    if digest(code.with_name("lsmi-daylight-switch-warmth-protocol.md")) != PROTOCOL:
        raise ValueError("Changed LSMI protocol")
    region_ledger = checked_json(layout, REGIONS)
    if region_ledger["anchor"] != "12" or region_ledger["shape"] != [614, 920] or set(region_ledger["scenes"]) != set(SCENES):
        raise ValueError("Unexpected LSMI anchor layout")
    check(cli, RT_CLI)
    template = check(base / "playraw-flash-room-intake-v1/martin.pp3", PP3)
    profile = cli.parent / "iccprofiles/output/RTv4_sRGB.icc"
    source = base / "srgb-lsmi-nikon-daylight-switch-input-v2"
    receipts = checked_json(source / "receipt.json", RECEIPT)
    if receipts["index_sha256"] != INDEX or set(receipts["files"]) != {f"{key}-{tag}" for key in SCENES for tag in LIGHTS}:
        raise ValueError("Unexpected LSMI image cohort")
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "rt-cache"), OMP_NUM_THREADS="1")
    out.mkdir()
    reports = {}
    for key in SCENES:
        scene = region_ledger["scenes"][key]
        if scene["anchor_sha256"] != receipts["files"][f"{key}-12"]["sha256"]:
            raise ValueError(f"Unexpected region anchor: {key}")
        if scene.get("skip"):
            reports[key] = dict(skip=scene["skip"])
            continue
        frames, image_info = {}, {}
        for lamp in LIGHTS:
            member = f"{key}-{lamp}"
            frames[lamp], image_info[lamp] = capture(source / f"{member}.png", receipts["files"][member])
        gradients = {name: gradient(frame) for name, frame in frames.items()}
        anchors = {name: shifts(gradients["12"], g)[0] for name, g in gradients.items()}
        bounds = {name: scene[name] for name in ("wall_left", "protect")}
        guards = rectangles(frames["12"].shape, bounds)
        edited = prepare(cli, template, profile.read_bytes(), env, out / key, frames, bounds,
                         {name: aligned(row) for name, row in anchors.items()})
        x0, y0, x1, y1 = edited["extent"]
        yy, xx = np.ogrid[:frames["12"].shape[0], :frames["12"].shape[1]]
        guards["outside"] = ~((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1))
        for label, row in edited["directions"].items():
            a, b = label.split("-to-")
            shift = shifts(gradients[a], gradients[b])[0]
            row.update(physical_reference_guards={name: physical_shift(frames[a], frames[b], guards[name])
                                                  for name in ("protect", "outside")},
                       source_kind="replacement" if "123" not in (a, b) else "addition_or_removal",
                       pair_alignment=shift, provisional=row["passed"] and aligned(shift) and "123" not in (a, b))
        reports[key] = dict(material=scene["material"], image_info=image_info, against_12=anchors, **edited)
    result = dict(index=INDEX, receipt=RECEIPT, script=digest(code), protocol=PROTOCOL,
                  regions=REGIONS, editor=RT_CLI, profile=digest(profile), template=PP3, fields=WARMTH_FIELDS,
                  reports=reports)
    json_to(out / "results.json", result)
    candidates = [direction for report in reports.values() for direction,row in report.get("directions",{}).items()
                  if row["provisional"]]
    print("Provisional auxiliary-light matches:", len(candidates), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "cli", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
