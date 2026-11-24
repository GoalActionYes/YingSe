"""Make controlled local-warmth positives on frozen physical light sequences."""
from argparse import ArgumentParser
from hashlib import sha256
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from probe_cross_source_wall_lightening_workbench import OUTDOOR_SHA, frozen_scenes, load
from probe_gros_local_editor_channel import frozen_json, read_frame
from probe_gros_local_editor_cross import REGIONS
from probe_matched_wall_lightening_workbench import PARENT
from probe_playraw_flash_room import LUMA, PP3, RT_CLI, read
from probe_playraw_flash_room_workbench import added_endpoints
from probe_shared_authored_tone_admission import ROOM_SHA, compare, room_profile, support
from probe_target_mirror_chroma import checked_json, digest, json_to, panel
from render_rawtherapee_sky_pair import check, disabled, render


PORTRAITS = "gros-local-editor-channel-v2"
PORTRAITS_SHA = "0198b52a85999a5f04a7939742712dec26c1e29d31cdca96db32d08425beb0c7"
PROTOCOL = "99d904e285ad4b93acbcc8373edecf4abe9c50efa8f5f413dcbf9aed88630db0"
WARMTH_FIELDS = dict(Lightness_0=0, labgridALow_0=1600, labgridAHigh_0=1600,
                     labgridBLow_0=2800, labgridBHigh_0=2800)


def warm_profile(local, rectangle, shape):
    original, location, extent = room_profile(local, rectangle, shape)
    if not {b"Strengthgrid_0=30", b"gridMethod_0=one"}.issubset(original.splitlines()):
        raise ValueError("Changed editor normalization")
    result = original
    for key, value in WARMTH_FIELDS.items():
        field = key.encode() + b"="
        result, count = re.subn(rb"(?m)^" + field + rb"[^\r\n]*", field + str(value).encode(), result)
        if count != 1:
            raise ValueError(f"Missing unique editor setting: {key}")
    return result, location, extent


def rectangles(shape, named_bounds):
    y, x = np.ogrid[:shape[0], :shape[1]]
    result = {}
    for key, (x0, y0, x1, y1) in named_bounds.items():
        result[key] = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        if not result[key].any():
            raise ValueError(f"Empty registered area: {key}")
    return result


def scene_sequences(base):
    for name, scene in frozen_scenes(base).items():
        family = "wildrelight" if scene["anchor"] == "time0" else "mit"
        frames = {time: load(scene, f"{time}-off")[1] for time in scene["frames"]}
        registered = {time: row.get("registered", True) for time, row in scene["frames"].items()}
        yield family, name, frames, scene["boxes"], registered

    intake = checked_json(base / PORTRAITS / "results.json", PORTRAITS_SHA)["intake"]
    regions = frozen_json(Path(__file__).with_name("gros-local-editor-cross-regions.json"), REGIONS)
    for subject, receipt in intake.items():
        frames = {light: read_frame(base / PORTRAITS / subject / light / "off.png", receipt[light]["hashes"]["off"])
                  for light in ("cold", "warm")}
        height, width = frames["cold"].shape[:2]
        bounds = {"wall_left": regions[subject]["cheek_right"], "other_cheek": regions[subject]["cheek_left"],
                  **regions["screen"]}
        boxes = {name: (x0 * width / 384, y0 * height / 256, x1 * width / 384, y1 * height / 256)
                 for name, (x0, y0, x1, y1) in bounds.items()}
        yield "gros", subject, frames, boxes, {"cold": True, "warm": True}


def admission(original, off, on, edit, guards):
    identity = np.abs(off.astype(np.int16) - original)
    delta = on.astype(np.int16) - off
    pixels = delta[edit]
    warmth = float(np.median(pixels[:, 0] - pixels[:, 2]))
    luma = float(np.median(pixels @ LUMA))
    coverage = float(support(off[edit]).mean())
    protected = {name: float(np.quantile(np.abs(delta[area]).mean(axis=-1), .95))
                 for name, area in guards.items()}
    endpoints = added_endpoints(off, on)
    passed = (identity.max() <= 2 and identity.mean() <= .25 and coverage >= .8 and warmth >= 3
              and abs(luma) <= 3 and max(protected.values()) <= 1.5 and endpoints <= .02)
    return dict(admitted=bool(passed), off_mae=float(identity.mean()), off_max=int(identity.max()),
                coverage=coverage, warmth=warmth, luma=luma, protected_p95=protected,
                endpoints=endpoints, edited_rgb_medians=np.median(pixels, axis=0).tolist())


def prepare(cli, template, profile_bytes, env, folder, frames, boxes, registered):
    folder.mkdir(parents=True)
    original = next(iter(frames.values()))
    profile, fields, extent = warm_profile(template, boxes["wall_left"], original.shape[:2])
    masks = rectangles(original.shape, {**boxes, "extent": extent})
    masks["outside"] = ~masks.pop("extent")
    edit = masks.pop("wall_left")
    if not masks["outside"].any():
        raise ValueError("No protected pixels outside the local edit")
    paths = {state: folder / f"{state}.pp3" for state in ("off", "on")}
    paths["on"].write_bytes(profile)
    paths["off"].write_bytes(disabled(profile))
    photos, receipt = {}, {}
    for light, source in frames.items():
        if source.shape != original.shape or source.dtype != np.uint8:
            raise ValueError("Changed frozen display geometry")
        input_path = folder / f"{light}-input.png"
        Image.fromarray(source).save(input_path, icc_profile=profile_bytes)
        observed, files = {}, {}
        for state, pp3 in paths.items():
            output = folder / f"{light}-{state}.png"
            render(cli, input_path, pp3, output, env)
            observed[state], _, files[state] = read(output, False)
            if observed[state].shape != source.shape or files[state]["icc_sha256"] != sha256(profile_bytes).hexdigest():
                raise ValueError("Changed output dimensions or embedded profile")
        photos[light] = observed
        receipt[light] = dict(metrics=admission(source, observed["off"], observed["on"], edit, masks),
                              registered=registered[light], files=files, source_pixels=sha256(source.tobytes()).hexdigest())
    admissible = {light: row["metrics"]["admitted"] and row["registered"] for light, row in receipt.items()}
    matches = compare(photos, edit, admissible)
    counts = dict(admitted=sum(admissible.values()), matched=sum(pair["passed"] for pair in matches.values()))
    board = panel(folder / "comparison.png", photos)
    print(folder.parent.name, folder.name, counts, flush=True)
    return dict(boxes=boxes, extent=extent, geometry=fields, profiles={key: digest(path) for key, path in paths.items()},
                frames=receipt, directions=matches, totals=counts, panel=board)


def run(base, cli, out):
    check(cli, RT_CLI)
    code = Path(__file__)
    if digest(code.with_name("shared-local-warmth-protocol.md")) != PROTOCOL:
        raise ValueError("Changed study protocol")
    template = check(base / "playraw-flash-room-intake-v1/martin.pp3", PP3)
    icc = (cli.parent / "iccprofiles/output/RTv4_sRGB.icc").read_bytes()
    out.mkdir()
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "rt-cache"), OMP_NUM_THREADS="1")
    results = {}
    for family, name, frames, boxes, registered in scene_sequences(base):
        results.setdefault(family, {})[name] = prepare(cli, template, icc, env, out / family / name,
                                                      frames, boxes, registered)
    json_to(out / "results.json", dict(parents=dict(mit=PARENT, outdoor=OUTDOOR_SHA, portraits=PORTRAITS_SHA,
                 rooms=ROOM_SHA, portrait_regions=REGIONS), template=PP3, cli=RT_CLI, fields=WARMTH_FIELDS,
                 protocol=PROTOCOL, script=digest(code), results=results))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("base", "cli", "out"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
