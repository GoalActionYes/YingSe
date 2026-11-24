"""Cross matching authored lightening with additional real outdoor time steps."""
from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
from PIL import Image

from probe_mid_matched_wall_cast import box
from probe_playraw_flash_room import LUMA, PP3, RT_CLI, read
from probe_shared_authored_tone_admission import compare, measurements, room_profile
from probe_target_mirror_chroma import checked_json, digest, json_to, panel
from probe_wildrelight_authored_darken import PROTECTED, REGIONS, SOURCE, masks
from probe_wildrelight_wall_cross import SCENES, image, prepared_rgb
from render_rawtherapee_sky_pair import check, disabled, render


RELEASE = "ce129428bb925df0f7aa490651d69951d3ca3c6a4ef5b9f5e2bd6abbbddb3116"
OLD_STEPS = ("time0", "time5")
NEW_STEPS = ("time1", "time2")
STATES = ("off", "on")


def scene_inputs(base, ledger, new_root, release, name, edit_box):
    saved = base / "wildrelight-wall-prepared-v1"
    raw = base / "wildrelight-small-v1"
    folder = f"small-aligned/{name}"
    metadata = checked_json(raw / folder / "meta.json", ledger["inputs"][folder + "/meta.json"])
    scene = ledger["scenes"][name]
    timestamps = {row["time"]: row for row in metadata["photos"]}
    if metadata["scene"] != name or metadata["alignment_method"] != scene["alignment"]:
        raise ValueError("Changed captured scene metadata")
    paths = {}
    for level in OLD_STEPS:
        item = f"{folder}/photo/{level}_hdr.exr"
        path = raw / item
        check(path, ledger["inputs"][item])
        paths[level] = path
    for item in release["files"]:
        if item["scene"] == name:
            path = new_root / item["path"]
            if path.stat().st_size != item["bytes"]:
                raise ValueError("Unexpected download size")
            check(path, item["sha256"])
            paths[item["time"]] = path
    photos = {tag: prepared_rgb(image(paths[tag]), scene["scale"]) for tag in (*OLD_STEPS, *NEW_STEPS)}
    for level in OLD_STEPS:
        path = saved / f"{name}-{level}.png"
        check(path, scene["hashes"][level])
        with Image.open(path) as saved_image:
            np.testing.assert_array_equal(photos[level], np.array(saved_image))
    if len({f.shape for f in photos.values()}) != 1:
        raise ValueError("Varying outdoor geometry")
    x0, y0, x1, y1 = edit_box
    gradients = {tag: np.diff(photos[tag][y0:y1, x0:x1] @ LUMA, axis=1).ravel() for tag in photos}
    metadata_rows = {tag: timestamps[tag] for tag in photos}
    if (len({r["shooting_time"] for r in metadata_rows.values()}) != len(photos)
            or any(r["ref_shutter"] != metadata["ref_shutter"] for r in metadata_rows.values())):
        raise ValueError("Changed exposure preparation and chronological interpretation")
    gradients = {tag: float(np.corrcoef(gradients["time0"], g)[0, 1]) for tag, g in gradients.items()}
    return photos, gradients, {tag: dict(shooting_time=r["shooting_time"], exposure_count=r["num_images"])
                                for tag, r in metadata_rows.items()}


def render_scene(out, cli, author, capture, bounds):
    originals, gradients, timing = capture
    out.mkdir()
    profile, geo, extent = room_profile(author, bounds["edit"], originals["time0"].shape[:2])
    pp3 = {state: out / f"{state}.pp3" for state in STATES}
    pp3["on"].write_bytes(profile)
    pp3["off"].write_bytes(disabled(profile))
    icc = (cli.parent / "iccprofiles/output/RTv4_sRGB.icc").read_bytes()
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "rt-cache"), OMP_NUM_THREADS="1")
    prepared, traces = {}, {}
    for time, src in originals.items():
        pic = out / f"{time}-input.png"
        Image.fromarray(src).save(pic, icc_profile=icc)
        output, files = {}, {}
        for state in STATES:
            path = out / f"{time}-{state}.png"
            render(cli, pic, pp3[state], path, env)
            output[state], _, files[state] = read(path, False)
            if output[state].shape != src.shape:
                raise ValueError("Unexpected editor dimensions")
        if files["off"]["icc_sha256"] != files["on"]["icc_sha256"]:
            raise ValueError("Different on/off color profiles")
        area = masks(src.shape, bounds["edit"], bounds["protect"], extent)
        values = measurements(src, output["off"], output["on"], area.pop("edit"), area)
        traces[time] = dict(metrics=values, gradient=gradients[time], metadata=timing[time], files=files,
                            registered=bool(time in OLD_STEPS or gradients[time] >= .70))
        prepared[time] = output
    approved = {time: item["metrics"]["admitted"] and item["registered"] for time, item in traces.items()}
    ratios = compare(prepared, box(originals["time0"], bounds["edit"]), approved)
    print(out.name, "eligible", [time for time in approved if approved[time]], "matched",
          [item for item in ratios if ratios[item]["passed"]], flush=True)
    return dict(rectangles=bounds, extent=extent, geometry=geo, cases=traces, comparison=ratios,
                pp3={state:digest(path) for state,path in pp3.items()},
                panel=panel(out / "comparison.png", prepared))


def run(base, cli, new_root, out):
    code = Path(__file__)
    release = checked_json(code.with_name("wildrelight-shared-tone-release.json"), RELEASE)
    original = checked_json(base / "wildrelight-wall-prepared-v1/inputs.json", SOURCE)
    bounds = checked_json(code.with_name("wildrelight-wall-cross-regions.json"), REGIONS)
    check(cli, RT_CLI)
    source = check(base / "playraw-flash-room-intake-v1/martin.pp3", PP3)
    out.mkdir()
    results = {}
    for name in SCENES:
        limits = dict(edit=bounds[name]["neutral"], protect=PROTECTED[name])
        capture = scene_inputs(base, original, new_root, release, name, limits["edit"])
        results[name] = render_scene(out / name, cli, source, capture, limits)
    matches = {scene: sum(row["passed"] for row in data["comparison"].values()) for scene,data in results.items()}
    json_to(out / "results.json", dict(source=SOURCE, regions=REGIONS, release=RELEASE, cli=RT_CLI, author=PP3,
            script_sha=digest(code), protocol_sha=digest(code.with_name("wildrelight-shared-tone-protocol.md")),
            scenes=results, matched=matches))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("base", "cli", "new_root", "out"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
