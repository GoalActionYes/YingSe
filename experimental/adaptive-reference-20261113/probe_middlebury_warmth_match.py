"""Check a frozen local-warmth edit against same-exposure Middlebury lighting."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from probe_playraw_flash_room import LUMA, PP3, RT_CLI
from probe_shared_local_warmth import PROTOCOL as WARMTH_PROTOCOL, prepare
from probe_target_mirror_chroma import digest, json_to
from render_rawtherapee_sky_pair import check


PROTOCOL = "22dfcdc28e787f15798967b997cb8875b2cb644d97e57a80c706913d9d252d44"
SCENES = {
    "Art": dict(size=4699854, archive="6a7c6f5fa27f41b1a092246859ea54b08a4125399e2bd7a36e8048fff719d0c0",
                default="087f18c9c27d0a4facd5e07df89ab7f04a4653207d7e8ad6dc08fc138160c15f",
                edit=(131, 119, 159, 152), protected=(367, 286, 390, 306)),
    "Dolls": dict(size=5098953, archive="c17029e0a0cecfe85e5aac09cf54348574e1a28b29f0a0997dacf22b08d32152",
                  default="fdfc295f9270e1a26d5ec6705030107541178d1b72bb699909044c27367500b1",
                  edit=(414, 146, 439, 163), protected=(35, 79, 70, 102)),
}


def rgb(png):
    with Image.open(BytesIO(png)) as image:
        if image.mode != "RGB" or image.size != (463, 370) or image.info.get("icc_profile"):
            raise ValueError("Changed fixed-view RGB encoding")
        return np.array(image)


def load_full(base, source, scene, spec):
    small = base / "middlebury-stereo-grade-input-v1" / f"{scene}-2views.zip"
    large = source / f"{scene}-2views-3illum-3exp.zip"
    if large.stat().st_size != spec["size"]:
        raise ValueError(f"Changed archive size: {scene}")
    with ZipFile(BytesIO(check(small, spec["default"]))) as original:
        reference = rgb(original.read(f"{scene}/view1.png"))
    frames, details = {}, {}
    with ZipFile(BytesIO(check(large, spec["archive"]))) as photos:
        for lamp in range(1, 4):
            for bracket in range(3):
                directory = f"{scene}/Illum{lamp}/Exp{bracket}"
                name = f"L{lamp}E{bracket}"
                png = photos.read(directory + "/view1.png")
                exposure = photos.read(directory + "/exposure_ms.txt")
                ms = int(exposure.decode("ascii").strip())
                if ms <= 0:
                    raise ValueError(f"Invalid capture exposure: {directory}")
                frames[name] = rgb(png)
                details[name] = dict(light=lamp, exposure_index=bracket, exposure_ms=ms,
                                     view=1, png_sha256=sha256(png).hexdigest(),
                                     exposure_sha256=sha256(exposure).hexdigest())
    np.testing.assert_array_equal(reference, frames["L1E1"])
    x0, y0, x1, y1 = spec["edit"]
    anchor = np.diff(reference[y0:y1, x0:x1] @ LUMA, axis=1).ravel()
    for name, frame in frames.items():
        gradient = np.diff(frame[y0:y1, x0:x1] @ LUMA, axis=1).ravel()
        corr = float(np.corrcoef(anchor, gradient)[0, 1]) if np.ptp(gradient) else None
        details[name].update(gradient=corr, registered=corr is not None and corr >= .70)
    return frames, details


def run(base, source, cli, out):
    check(cli, RT_CLI)
    code = Path(__file__)
    if (digest(code.with_name("middlebury-warmth-match-protocol.md")) != PROTOCOL
            or digest(code.with_name("shared-local-warmth-protocol.md")) != WARMTH_PROTOCOL):
        raise ValueError("Changed frozen study protocol")
    editor = code.with_name("probe_shared_local_warmth.py")
    template = check(base / "playraw-flash-room-intake-v1/martin.pp3", PP3)
    icc = (cli.parent / "iccprofiles/output/RTv4_sRGB.icc").read_bytes()
    out.mkdir()
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "rt-cache"), OMP_NUM_THREADS="1")
    reports = {}
    for scene, spec in SCENES.items():
        frames, captures = load_full(base, source, scene, spec)
        report = prepare(cli, template, icc, env, out / scene, frames,
                         {"wall_left": spec["edit"], "protect": spec["protected"]},
                         {name: data["registered"] for name, data in captures.items()})
        for key, pair in report["directions"].items():
            a, b = (captures[name] for name in key.split("-to-"))
            eligible = a["exposure_ms"] == b["exposure_ms"] and a["light"] != b["light"]
            pair["same_exposure_different_light"] = eligible
            pair["passed_physical"] = eligible and pair["passed"]
        comparisons = list(report["directions"].values())
        report["totals"] = dict(
            eligible_edit_frames=sum(frame["metrics"]["admitted"] for frame in report["frames"].values()),
            registered_frames=sum(frame["registered"] for frame in report["frames"].values()),
            same_exposure_different_light=sum(row["same_exposure_different_light"] for row in comparisons),
            matched=sum(row["passed_physical"] for row in comparisons))
        reports[scene] = dict(spec=spec, captures=captures, **report)
        print(scene, reports[scene]["totals"], flush=True)
    json_to(out / "results.json", dict(scenes=reports, editor_template=PP3, cli=RT_CLI, protocol=PROTOCOL,
                                      script=digest(code), warmth_protocol=WARMTH_PROTOCOL, warmth_script=digest(editor)))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("base", "source", "cli", "out"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
