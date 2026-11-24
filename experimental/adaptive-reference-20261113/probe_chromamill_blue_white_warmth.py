"""Check fixed wall-warmth edits against cached CHROMA-MILL captures."""
from argparse import ArgumentParser
from pathlib import Path
import os

from probe_chromamill_camera_pairs import verified
from probe_chromamill_capture_cross import LEVELS, REVISION, SCENES, decode_rgb8, source_members
from probe_chromamill_fixed_material import BOXES
from probe_inretouch_registration import gradient, shifts
from probe_playraw_flash_room import PP3, RT_CLI
from probe_shared_local_warmth import WARMTH_FIELDS, prepare
from probe_target_mirror_chroma import checked_json, digest, json_to
from render_rawtherapee_sky_pair import check


PROTOCOL = "180943537859b7d3a0821c24d2cdd96708909d12a66f6bdc47db39801af8ed00"
SOURCE = "121951adebd16c5b73b68e067bf3523b2794cec582a851664e7243cd584eb04b"
AREAS = {
    scene: dict(zip(("wall_left", "table2", "table3", guard), BOXES[scene], strict=True))
    for scene, guard in (("Scene1_a", "bottle"), ("Scene2_a", "fabric"))
}
LIGHTS = ("white", "blue")


def aligned(row):
    return row["zero"] >= .60 and row["peak_gain"] <= .02 and row["dy"] == row["dx"] == 0


def run(base, cli, out):
    script = Path(__file__)
    if digest(script.with_name("chromamill-blue-white-warmth-protocol.md")) != PROTOCOL:
        raise ValueError("Changed CHROMA-MILL warmth protocol")
    check(cli, RT_CLI)
    template = check(base / "playraw-flash-room-intake-v1/martin.pp3", PP3)
    profile = cli.parent / "iccprofiles/output/RTv4_sRGB.icc"
    source = base / "chromamill-cross-assets-v1"
    identity = checked_json(source / "receipt.json", SOURCE)
    if identity["revision"] != REVISION or set(identity["files"]) != set(source_members()):
        raise ValueError("Changed published white/blue cohort")
    for name, expected in source_members().items():
        if identity["files"][name]["path"] != expected:
            raise ValueError(f"Changed published source: {name}")
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "rt-cache"), OMP_NUM_THREADS="1")
    out.mkdir()
    reports = {}
    for name in SCENES:
        frames, files = {}, {}
        for light in LIGHTS:
            for level in LEVELS:
                tag = f"{light}{level}"
                key = f"{name}-{light}-{level}"
                expected = identity["files"][key]
                path = verified(source / f"{key}.png", expected)
                frame, decode = decode_rgb8(path)
                if frame.shape != (341, 512, 3):
                    raise ValueError(f"Changed registered dimensions: {key}")
                frames[tag] = frame
                files[tag] = dict(published=expected, decode=decode)
        gradients = {tag: gradient(frame) for tag, frame in frames.items()}
        anchor = {tag: shifts(gradients["white241"], value)[0] for tag, value in gradients.items()}
        material = prepare(cli, template, profile.read_bytes(), env, out / name, frames, AREAS[name],
                           {tag: aligned(row) for tag, row in anchor.items()})
        for label, row in material["directions"].items():
            before, after = label.split("-to-")
            match = shifts(gradients[before], gradients[after])[0]
            color_changed = before.rstrip("0123456789") != after.rstrip("0123456789")
            row.update(registration=match, raw_light_color_changed=color_changed,
                       admitted_color_match=row["passed"] and color_changed and aligned(match))
        reports[name] = dict(**material, files=files, against_white241=anchor)
    directions = [row for group in reports.values() for row in group["directions"].values()]
    result = dict(revision=REVISION, original_receipt=SOURCE, script=digest(script), protocol=PROTOCOL,
                  editor=RT_CLI, profile=digest(profile), template=PP3, fields=WARMTH_FIELDS,
                  frame_count=sum(len(group["frames"]) for group in reports.values()),
                  frames_admitted=sum(group["totals"]["admitted"] for group in reports.values()),
                  all_directions=len(directions), cross_color_directions=sum(row["raw_light_color_changed"] for row in directions),
                  admitted_color_matches=sum(row["admitted_color_match"] for row in directions),
                  groups=reports)
    json_to(out / "results.json", result)
    print({key: result[key] for key in ("frame_count", "frames_admitted", "all_directions", "cross_color_directions",
                                         "admitted_color_matches")}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("base", "cli", "out"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
