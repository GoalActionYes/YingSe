"""Measure matched physical and authored lightening on two capture sources."""
from argparse import ArgumentParser
from itertools import permutations
import os
from pathlib import Path

import numpy as np
from PIL import Image

from probe_gros_author_tone_switch import relocate
from probe_gros_local_editor_channel import frozen_json, read_frame
from probe_gros_local_editor_cross import REGIONS, areas_and_weight
from probe_gros_mixed_light_transfer import LEVELS
from probe_mid_matched_wall_cast import box
from probe_playraw_flash_room import LUMA, PP3, RT_CLI, read
from probe_playraw_flash_room_workbench import added_endpoints
from probe_target_mirror_chroma import checked_json, digest, json_to, panel
from render_rawtherapee_sky_pair import check, disabled, render


ROOM_SHA = "766c8e5df51edfc4c11d4d1b1e4cce2b20ea2fbadb226057b69ba7eeecf9408d"
PORTRAIT_SHA = "6b66892e5966ab36920c89da11696e1a2067bb43fee13db4b608b5c22c596cd5"
ROOMS = ("14n_copyroom1", "everett_living2")
LAMPS = ("0", "4", "8", "12", "14", "16", "20", "24")
STATES = ("off", "on")


def support(rgb):
    return ((rgb >= 4) & (rgb <= 251)).all(axis=-1)


def measurements(source, off, on, edit, protected):
    identity = np.abs(off.astype(np.int16) - source)
    delta = on.astype(np.int16) - off
    change = float(np.median(delta[edit] @ LUMA))
    protected_p95 = {key: float(np.quantile(np.abs(delta[area]).mean(axis=1), .95))
                     for key, area in protected.items()}
    effective = float(support(off[edit]).mean())
    endpoints = added_endpoints(off, on)
    passed = (effective >= .8 and identity.max() <= 2 and identity.mean() <= .25 and change >= 3
              and max(protected_p95.values()) <= 1.5 and endpoints <= .02)
    return dict(admitted=bool(passed), coverage=effective, off_max=int(identity.max()), off_mae=float(identity.mean()),
                luma_median=change, protected_p95=protected_p95, added_endpoint_fraction=endpoints)


def compare(photos, area, admissible):
    rows = {}
    for a, b in permutations(photos, 2):
        baseline = photos[a]["off"][area]
        physical = photos[b]["off"][area]
        edit = photos[a]["on"][area]
        valid = support(baseline) & support(physical) & support(edit)
        coverage = float(valid.mean())
        capture, authored = np.zeros(3), np.zeros(3)
        if valid.any():
            before, after_light, after_edit = [np.median(frame[valid], axis=0) for frame in (baseline, physical, edit)]
            capture, authored = after_light - before, after_edit - before
        length_l, length_e = np.linalg.norm(capture), np.linalg.norm(authored)
        cosine = float(capture @ authored / (length_l * length_e)) if length_l and length_e else None
        ratio = float(length_e / length_l) if length_l else None
        checks = dict(valid_coverage=coverage >= .8, both_material_states=admissible[a] and admissible[b],
                      physical_signal=bool(np.abs(capture).mean() >= 3), edited_signal=bool(np.abs(authored).mean() >= 3),
                      same_direction=cosine is not None and cosine >= .90,
                      comparable_amplitude=ratio is not None and .75 <= ratio <= 1.25)
        rows[f"{a}-to-{b}"] = dict(coverage=coverage, capture=capture.tolist(), edit=authored.tolist(),
                                    cosine=cosine, ratio=ratio, checks=checks, passed=all(checks.values()))
    return rows


def room_photos(base, room, ledger):
    for light in LAMPS:
        directory = "mid-new-room-ranges-v1" if light in ("14", "24") else "mid-gray-anchor-new-jpg-v1"
        path = base / directory / f"{room}-{light}.jpg"
        yield light, read_frame(path, ledger["members"][light]["sha256"])


def room_profile(author, rectangle, shape):
    left, top, right, bottom = rectangle
    width, height = shape[1], shape[0]
    cx, cy = (left + right) / 2, (top + bottom) / 2
    rx, ry = 1.5 * (right - left), 2 * (bottom - top)
    coords = dict(cheek_right=[left * 384 / width, top * 256 / height, right * 384 / width, bottom * 256 / height],
                  ellipse=[0, 0, 2 * rx * 384 / width, 2 * ry * 256 / height])
    profile, fields = relocate(author, coords)
    return profile, fields, (cx - rx * 2, cy - ry * 2, cx + rx * 2, cy + ry * 2)


def room_switch(base, cli, output, author, room, ledger):
    folder = output / room
    folder.mkdir()
    env = dict(os.environ, RT_SETTINGS=str(folder / "rt-settings"), RT_CACHE=str(folder / "rt-cache"), OMP_NUM_THREADS="1")
    icc = (cli.parent / "iccprofiles/output/RTv4_sRGB.icc").read_bytes()
    photos = dict(room_photos(base, room, ledger))
    image = photos[LAMPS[0]]
    boxes = ledger["boxes"]
    profile, fields, extent = room_profile(author, boxes["wall_left"], image.shape[:2])
    edit = box(image, boxes["wall_left"])
    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    x0, y0, x1, y1 = extent
    protected = {key: box(image, value) for key, value in boxes.items() if key != "wall_left"}
    protected["outside"] = ~((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1))
    pp3 = {state: folder / f"{state}.pp3" for state in STATES}
    pp3["on"].write_bytes(profile)
    pp3["off"].write_bytes(disabled(profile))
    traces, prepared = {}, {}
    for light, source in photos.items():
        original = folder / f"{light}-input.png"
        Image.fromarray(source).save(original, icc_profile=icc)
        prepared[light], files = {}, {}
        for state in STATES:
            path = folder / f"{light}-{state}.png"
            render(cli, original, pp3[state], path, env)
            rgb, _, info = read(path, False)
            if rgb.shape != image.shape:
                raise ValueError("Editor changed room geometry")
            prepared[light][state] = rgb
            files[state] = info
        if files["off"]["icc_sha256"] != files["on"]["icc_sha256"]:
            raise ValueError("Editor changed color profile")
        metrics = measurements(source, *(prepared[light][tag] for tag in STATES), edit, protected)
        traces[light] = dict(metrics=metrics, files=files)
    cases = compare(prepared, edit, {light: row["metrics"]["admitted"] for light, row in traces.items()})
    print(room, sum(row["metrics"]["admitted"] for row in traces.values()), sum(p["passed"] for p in cases.values()), flush=True)
    return dict(geometry=fields, extent=extent, pp3_sha={k: digest(v) for k, v in pp3.items()},
                renders=traces, comparisons=cases,
                board=panel(folder / "comparison.png", prepared))


def portraits(base):
    archive = base / "gros-author-tone-switch-v1"
    parent = checked_json(archive / "results.json", PORTRAIT_SHA)
    regions = frozen_json(Path(__file__).with_name("gros-local-editor-cross-regions.json"), REGIONS)
    result = {}
    for person, info in parent["receipts"].items():
        area = areas_and_weight(regions[person], regions["screen"])[0]["cheek_right"]
        photos = {}
        for level in LEVELS:
            photos[level] = {}
            for state in STATES:
                rgb, _, identity = read(archive / person / level / f"editor-{state}.png", False)
                source = info["cases"][level]
                if identity["pixels_sha256"] != source["output_pixels"][state] or identity["icc_sha256"] != source["icc_sha256"]:
                    raise ValueError("Changed frozen person output")
                photos[level][state] = rgb
        result[person] = compare(photos, area, {level: info["cases"][level]["admission"]["passed"] for level in LEVELS})
    return result


def run(base, cli, output):
    check(cli, RT_CLI)
    author = check(base / "playraw-flash-room-intake-v1" / "martin.pp3", PP3)
    ledger = checked_json(base / "mid-gray-anchor-v1.json", ROOM_SHA)["scenes"]
    output.mkdir()
    rooms = {name: room_switch(base, cli, output, author, name, ledger[name]) for name in ROOMS}
    people = portraits(base)
    searched = dict(rooms=sum(v["passed"] for x in rooms.values() for v in x["comparisons"].values()),
                    portraits=sum(v["passed"] for x in people.values() for v in x.values()))
    origin = Path(__file__)
    json_to(output / "results.json", dict(rooms=rooms, portraits=people, matched=searched,
            room_source=ROOM_SHA, portrait_source=PORTRAIT_SHA, author=PP3, cli=RT_CLI,
            script_sha=digest(origin), protocol_sha=digest(origin.with_name("shared-authored-tone-admission-protocol.md"))))
    print(searched, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("base", "cli", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
