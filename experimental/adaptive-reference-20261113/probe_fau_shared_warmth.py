"""Check fixed authored warmth against FAU's physical two-lamp captures."""
from argparse import ArgumentParser
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tarfile

import numpy as np
from PIL import Image

from audit_fau_two_light_intake import ARCHIVE_SHA, LAB, sha
from probe_playraw_flash_room import PP3, RT_CLI
from probe_shared_local_warmth import WARMTH_FIELDS, prepare, rectangles
from probe_target_mirror_chroma import digest, json_to
from render_rawtherapee_sky_pair import check


REGIONS = "4f170cd9bf33164d99d30e3440f4720a8039831fad3d1e0456a8153fe55881e3"
NAME = re.compile(r"(?P<group>[a-z0-9]+)_(?P<light>[wrb]_[wrb])\.png")


def read_archive(path, layout):
    if digest(path) != ARCHIVE_SHA:
        raise ValueError("Changed frozen FAU archive")
    with tarfile.open(path, "r:gz") as archive:
        entries = [row for row in archive.getmembers() if row.isfile()]
        names = {item.name for item in entries}
        if len(names) != len(entries):
            raise ValueError("Ambiguous archive member names")
        prefixes = (LAB + "/srgb8bit/", LAB + "/masks/")
        payloads = {item.name: archive.extractfile(item).read() for item in entries if item.name.startswith(prefixes)}
        groups = {key: {} for key in layout}
        for name in payloads:
            if not name.startswith(LAB + "/srgb8bit/"):
                continue
            match = NAME.fullmatch(Path(name).name)
            if match is None or match["group"] not in groups:
                raise ValueError("Changed frozen scene set")
            group, light = match["group"], match["light"]
            preview, valid = (payloads[f"{LAB}/{role}/{group}_{light}.png"] for role in ("srgb8bit", "masks"))
            with Image.open(BytesIO(preview)) as rgb, Image.open(BytesIO(valid)) as mask:
                if rgb.mode != "RGB" or mask.mode != "L" or rgb.size != (452, 260) or mask.size != rgb.size:
                    raise ValueError("Changed FAU preview/mask format")
                frame, flags = np.array(rgb), np.array(mask)
            if not np.isin(flags, [0, 255]).all():
                raise ValueError("Expected published binary invalid-pixel mask")
            groups[group][light] = dict(rgb=frame, valid=flags == 255,
                                        rgb_sha256=sha(preview), mask_sha256=sha(valid))
    if sum(len(values) for values in groups.values()) != 58 or any(
            len(rows) not in (4, 6) or "w_b" not in rows for rows in groups.values()):
        raise ValueError("Changed FAU lighting cohort")
    return groups


def verify_direction(rows, edit_region, result):
    for label, original in result["directions"].items():
        first, second = label.split("-to-")
        held_position = sum(a == b for a, b in zip(first.split("_"), second.split("_"), strict=True)) == 1
        support = rows[first]["valid"][edit_region] & rows[second]["valid"][edit_region]
        mask_coverage = float(support.mean())
        original.update(one_lamp_changed=held_position, fau_mask_common_coverage=mask_coverage,
                        in_scope_matched=original["passed"] and held_position and mask_coverage >= .8)


def run(base, cli, out):
    check(cli, RT_CLI)
    script = Path(__file__)
    ledger = script.with_name("fau-shared-warmth-regions.json")
    protocol = script.with_name("fau-shared-warmth-protocol.md")
    if digest(ledger) != REGIONS:
        raise ValueError("Changed fixed scene areas")
    lock = json.loads(ledger.read_bytes())
    layout = lock["groups"]
    if lock["anchor_light"] != "w_b" or lock["shape"] != [260, 452]:
        raise ValueError("Unexpected layout geometry")
    archive = base / "fau-two-light-intake-v1" / "benchmark_mirf.tar.gz"
    scenes = read_archive(archive, layout)
    template = check(base / "playraw-flash-room-intake-v1/martin.pp3", PP3)
    profile = (cli.parent / "iccprofiles/output/RTv4_sRGB.icc").read_bytes()
    out.mkdir()
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "rt-cache"), OMP_NUM_THREADS="1")
    reports = {}
    for group, rows in scenes.items():
        bounds = {name: layout[group][name] for name in ("wall_left", "protect")}
        valid = rectangles(tuple(lock["shape"]), bounds)
        frames = {name: row["rgb"] for name, row in sorted(rows.items())}
        folder = out / "fau" / group
        rendered = prepare(cli, template, profile, env, folder, frames, bounds, dict.fromkeys(frames, True))
        verify_direction(rows, valid["wall_left"], rendered)
        rendered["material"] = layout[group]["material"]
        rendered["input_ids"] = {key: {name: row[name] for name in ("rgb_sha256", "mask_sha256")}
                                 for key, row in rows.items()}
        rendered["in_scope_matched"] = sum(row["in_scope_matched"] for row in rendered["directions"].values())
        reports[group] = rendered
    directions = [row for group in reports.values() for row in group["directions"].values()]
    if len(directions) != 282:
        raise ValueError("Changed number of physical comparison directions")
    result = dict(groups=reports, archive_sha256=ARCHIVE_SHA, editor_sha256=RT_CLI, template_sha256=PP3,
                  profile_sha256=sha(profile), editor_fields=WARMTH_FIELDS, regions_sha256=REGIONS,
                  protocol_sha256=digest(protocol), script_sha256=digest(script),
                  total_frames=58, native_positive=sum(group["totals"]["admitted"] for group in reports.values()),
                  total_directions=len(directions), standard_matches=sum(row["passed"] for row in directions),
                  single_lamp_directions=sum(row["one_lamp_changed"] for row in directions),
                  strict_directions=sum(row["in_scope_matched"] for row in directions))
    json_to(out / "results.json", result)
    print({"native_positive": result["native_positive"], "original_matches": result["standard_matches"],
           "in_scope_matches": result["strict_directions"]}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("base", "cli", "out"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
