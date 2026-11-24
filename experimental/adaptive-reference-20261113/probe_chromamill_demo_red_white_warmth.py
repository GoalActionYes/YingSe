"""Compare actual red/white CHROMA-MILL demo captures with fixed local warmth."""
from argparse import ArgumentParser
from pathlib import Path
import os

from fetch_chromamill_single_light_demo import LEVELS, LIGHTS, REVISION, SCENES, TOTAL, TREE
from probe_chromamill_blue_white_warmth import aligned
from probe_chromamill_capture_cross import decode_rgb8
from probe_inretouch_registration import gradient, shifts
from probe_playraw_flash_room import PP3, RT_CLI
from probe_shared_local_warmth import WARMTH_FIELDS, prepare
from probe_target_mirror_chroma import checked_json, digest, json_to
from render_rawtherapee_sky_pair import check


PROTOCOL = "68fcb2acf9c44869e0b3cacb93ed2ca81bbd0755b84382dfc441080f89b235c1"
RECEIPT = "9de16489ad34535c3a14c69ccea6e5293f9e94f8643c2f441a084c4d73589def"
AREAS = {
    "Test13": {"wall_left": (321, 85, 348, 104), "protect": (83, 232, 120, 256)},
    "Test15": {"wall_left": (342, 133, 368, 147), "protect": (43, 170, 64, 198)},
    "Test17": {"wall_left": (5, 32, 25, 45), "protect": (275, 100, 305, 113)},
}


def run(base, cli, out):
    script = Path(__file__)
    if digest(script.with_name("chromamill-demo-red-white-warmth-protocol.md")) != PROTOCOL:
        raise ValueError("Changed demo warmth protocol")
    check(cli, RT_CLI)
    template = check(base / "playraw-flash-room-intake-v1/martin.pp3", PP3)
    profile = cli.parent / "iccprofiles/output/RTv4_sRGB.icc"
    source = base / "chromamill-single-light-demo-v1"
    receipt = checked_json(source / "receipt.json", RECEIPT)
    if (receipt["revision"], receipt["tree"], receipt["asset_bytes"], len(receipt["files"])) != (
            REVISION, TREE, TOTAL, 66):
        raise ValueError("Changed source demo")
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "rt-cache"), OMP_NUM_THREADS="1")
    out.mkdir()
    reports = {}
    for name in SCENES:
        frames = {}
        for light in LIGHTS:
            for level in LEVELS:
                tag = f"{light}{level:02d}"
                asset = f"{name}/{light}_L{level:02d}"
                path = source / f"{asset}.png"
                row = receipt["files"][asset]
                if path.stat().st_size != row["bytes"] or digest(path) != row["png_sha256"]:
                    raise ValueError(f"Changed checked demo capture: {asset}")
                frame, decode = decode_rgb8(path)
                if frame.shape != (341, 512, 3) or decode != {"shape": [400, 600, 3], "dtype": "uint8"}:
                    raise ValueError(f"Changed decoded demo capture: {asset}")
                frames[tag] = frame
        ref_grad = gradient(frames["white11"])
        anchor = {tag: shifts(ref_grad, gradient(frame))[0] for tag, frame in frames.items()}
        material = prepare(cli, template, profile.read_bytes(), env, out / name, frames, AREAS[name],
                           {tag: aligned(row) for tag, row in anchor.items()})
        for label, row in material["directions"].items():
            before, after = label.split("-to-")
            same_level = before[-2:] == after[-2:]
            changed_light = before[:-2] != after[:-2]
            row.update(same_level=same_level, light_color_changed=changed_light,
                       tentative_match=row["passed"] and changed_light)
        reports[name] = dict(**material, against_white11=anchor)
    directions = [row for group in reports.values() for row in group["directions"].values()]
    count = sum(row["light_color_changed"] for row in directions)
    if count != 726:
        raise ValueError("Changed fixed demo direction count")
    result = dict(revision=REVISION, capture_receipt=RECEIPT, script=digest(script), protocol=PROTOCOL,
                  editor=RT_CLI, profile=digest(profile), template=PP3, fields=WARMTH_FIELDS,
                  frame_count=sum(len(group["frames"]) for group in reports.values()),
                  frames_admitted=sum(group["totals"]["admitted"] for group in reports.values()),
                  all_directions=len(directions), cross_color_directions=count,
                  tentative_matches=sum(row["tentative_match"] for row in directions), groups=reports)
    json_to(out / "results.json", result)
    print({key: result[key] for key in ("frame_count", "frames_admitted", "all_directions", "cross_color_directions",
                                         "tentative_matches")}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "cli", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
