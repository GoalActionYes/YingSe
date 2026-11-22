"""Preserve channel differences in an authored SDR test edit and run both baselines."""
from argparse import ArgumentParser
from contextlib import contextmanager
from hashlib import sha256
from itertools import permutations
import io
import json
from pathlib import Path
import threading
from time import monotonic, sleep
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import torch

from probe_gros_local_editor_cross import REGIONS, areas_and_weight, score_outputs
from probe_gros_local_editor_float import CHEEKS, SCREENS, read_linear_exr, summarize
from probe_physical_blue_global_fit import display, linear
from probe_playraw_mixed_light_transfer import LUMA
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to, panel, save
from reference_color_app.pipeline import algorithm_fingerprint
from reference_color_app.server import make_server
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench


FLOAT_REPORT = "747226ea325afe0207525397f2ca51829366d8e64f5bfbf4ee12eb6228a0f18e"
SDR_REPORT = "3dd7043172870f6d79229cee012f39a0346f18b4c4dfc4e261dd6d1c4f83bb76"
EXR_SHA = "1da4f3afab371033dacaa2f135e3e808380ee5d05c57009e76dacd3e115c62e8"
SCALE = 8.


def frozen_json(path, fingerprint):
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if sha256(data).hexdigest() != fingerprint:
        raise ValueError(f"Frozen input changed: {path}")
    return json.loads(data)


def read_frame(path, fingerprint):
    if digest(path) != fingerprint:
        raise ValueError(f"Frozen PNG changed: {path}")
    with Image.open(path) as image:
        return np.array(image)


def signed_code(linear_rgb):
    magnitude = np.abs(linear_rgb)
    transfer = np.where(magnitude <= .0031308, magnitude * 12.92, 1.055 * magnitude ** (1 / 2.4) - .055)
    return np.copysign(255 * transfer, linear_rgb)


def capacity(value):
    lifted = 1 + SCALE * np.logaddexp(0., (value - 1) / SCALE)
    return 254 - SCALE * np.logaddexp(0., (254 - lifted) / SCALE)


def map_channels(original, intended):
    origin = original.astype(np.float64)
    shift = capacity(signed_code(intended)) - capacity(origin)
    return np.rint(np.clip(origin + shift, 0, 255)).astype(np.uint8)


def verify_math():
    code = np.arange(256, dtype=np.uint8)
    np.testing.assert_array_equal(map_channels(code, linear(code)), code)
    span = capacity(np.arange(-30, 286, dtype=float))
    if np.any(np.diff(span) <= 0) or span.min() <= 1 or span.max() >= 254:
        raise ValueError("Channel mapping did not stay ordered and bounded")


def prepare(project, output, subject, light, hashes, frozen, raw, regions, alpha):
    relative = f"{subject}/{light}"
    source = project / "gros-local-editor-float-v2" / relative
    before = read_frame(source / "off.png", frozen["hashes"]["off"])
    clipped = read_frame(source / "clip.png", frozen["hashes"]["clip"])
    compressed = read_frame(source / "on.png", frozen["hashes"]["on"])
    earlier = read_frame(project / "gros-local-editor-cross-v2" / relative / "on.png", raw["hashes"]["on"])
    rgb = {}
    for name in ("off", "on"):
        frame = f"editor-{name}.exr"
        path = source / frame
        if digest(path) != hashes[f"{relative}/{frame}"]:
            raise ValueError(f"Frozen EXR changed: {path}")
        rgb[name] = read_linear_exr(path)
    np.testing.assert_array_equal(display(rgb["off"]), before)
    visible = alpha > 0
    intended = linear(before[visible]) + alpha[visible, None] * (rgb["on"] - rgb["off"])[visible]
    np.testing.assert_array_equal(display(intended), clipped[visible])
    edited = before.copy()
    edited[visible] = map_channels(before[visible], intended)
    np.testing.assert_array_equal(edited[~visible], before[~visible])
    score = summarize(before, earlier, edited, regions)
    old_floor = (earlier[..., 2] == 0) & (before[..., 2] > 0)
    blue = edited[..., 2][old_floor]
    percentiles = np.percentile(blue, [5, 95]).tolist() if blue.size else None
    detail = dict(pixels=int(blue.size), levels=int(np.unique(blue).size), percentiles=percentiles)
    detail["retained"] = (detail["levels"] >= 3 and percentiles is not None and percentiles[1] - percentiles[0] >= 2)
    difference = (edited.astype(np.float64) - clipped) @ LUMA
    luma = {name: float(np.abs(difference[regions[name]]).mean()) for name in CHEEKS}
    eligible = (score["fraction"] <= .02
                and all(score["regions"][name]["pixels"] >= 128 for name in regions)
                and all(score["regions"][name]["chroma_median"] >= 2
                        and .75 <= score["regions"][name]["projection"] <= 1.25
                        and luma[name] <= 3 for name in CHEEKS)
                and all(score["regions"][name]["on_matches_authored_mae"] == 0 for name in SCREENS)
                and (light == "cold" or detail["retained"]))
    destination = output / relative
    destination.mkdir(parents=True)
    frames = dict(off=before, on=edited)
    receipt = dict(qualified=eligible, score=score, formerly_zero_blue=detail, luma_mae_against_clip=luma,
                   hashes={name: save(destination / f"{name}.png", image) for name, image in frames.items()},
                   panel=panel(destination / "comparison.png", {"SDR": {"original": before, "old edit": earlier,
                       "clip": clipped, "chroma rolloff": compressed, "channel rolloff": edited}}))
    return frames, receipt


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(server, path, body=None, name=None):
    headers = {"Content-Type": "application/json"}
    if name is not None:
        headers = {"Content-Type": "application/octet-stream", "X-Filename": quote(name)}
    elif body is not None:
        body = json.dumps(body).encode("utf-8")
    url = f"http://127.0.0.1:{server.server_port}{path}"
    with urlopen(Request(url, data=body, headers=headers), timeout=60) as response:
        return json.load(response) if response.headers.get_content_type() == "application/json" else response.read()


def actual_png(server, target, reference, destination):
    mapping = request(server, "/api/analyze", {"target_id": target, "reference_id": reference})["mapping_id"]
    job = request(server, "/api/export", dict(mapping_id=mapping, media_id=target, strength=STRENGTH, kind="png"))
    deadline = monotonic() + 90
    while job["status"] in ("queued", "running"):
        if monotonic() > deadline:
            raise TimeoutError("Existing PNG export did not finish")
        sleep(.05)
        job = request(server, "/api/jobs/" + job["id"])
    if job["status"] != "done":
        raise RuntimeError(job)
    png = request(server, job["result_url"])
    destination.write_bytes(png)
    with Image.open(io.BytesIO(png)) as picture:
        rgb = np.array(picture.convert("RGB"))
    return rgb, sha256(png).hexdigest()


def compare_workbenches(out, qualified, photos, areas):
    torch.set_num_threads(4)
    stable = make_server(out / "stable-data")
    spatial = make_spatial_workbench(out / "spatial-data", enhanced=True)
    with serving(stable), serving(spatial):
        if spatial.store.engine.checkpoint_sha != CHECKPOINT:
            raise ValueError("Changed model fingerprint")
        servers = {"stable": stable, "spatial13": spatial}
        assets = {name: {} for name in servers}
        identity, cases = {}, {}
        for subject in qualified:
            original = photos[subject]["cold"]["off"]
            identity[subject] = {}
            for model, server in servers.items():
                for light, pair in photos[subject].items():
                    for state in pair:
                        path = out / subject / light / f"{state}.png"
                        assets[model][subject, light, state] = request(server, "/api/media", path.read_bytes(), path.name)["id"]
                key = subject, "cold", "off"
                own = assets[model][key]
                checked, fingerprint = actual_png(server, own, own, out / f"{subject}-{model}-same.png")
                np.testing.assert_array_equal(checked, original)
                identity[subject][model] = fingerprint
        for subject, donor in permutations(qualified, 2):
            slug = f"{subject}-from-{donor}"
            target = photos[subject]["cold"]["off"]
            expected = finish_spatial(target, photos[subject]["cold"]["on"], STRENGTH)
            frames, hashes = {}, {}
            for model, server in servers.items():
                frames[model] = {}
                owned = assets[model]
                for light, pair in photos[donor].items():
                    frames[model][light] = {}
                    for state in pair:
                        suffix = f"{model}-{light}-{state}"
                        frames[model][light][state], hashes[suffix] = actual_png(server, owned[subject, "cold", "off"],
                            owned[donor, light, state], out / f"{slug}-{suffix}.png")
            scores = score_outputs(target, expected, frames, areas[subject])
            flat = {name: {f"{light}-{state}": rgb for light, pair in pictures.items() for state, rgb in pair.items()}
                    for name, pictures in ({"reference": photos[donor]} | frames).items()}
            cases[slug] = dict(scores=scores, complete_pass={name: all(row["passed"] for row in result.values())
                                      for name, result in scores.items()}, hashes=hashes,
                                wanted=save(out / f"{slug}-expected.png", expected),
                                panel=panel(out / f"{slug}-comparison.png", {"target": {"input": target, "expected": expected}} | flat))
            print(f"Exported {subject} from {donor}", flush=True)
    return identity, cases

def run(project, output):
    verify_math()
    parent = checked_json(project / "gros-local-editor-float-v2/results.json", FLOAT_REPORT)
    old = checked_json(project / "gros-local-editor-cross-v2/results.json", SDR_REPORT)
    script = Path(__file__)
    hashes = frozen_json(script.with_name("gros-local-editor-channel-exr.json"), EXR_SHA)
    coords = frozen_json(script.with_name("gros-local-editor-cross-regions.json"), REGIONS)
    output.mkdir()
    photos, areas, intake = {}, {}, {}
    for subject, lights in parent["admission"].items():
        areas[subject], alpha = areas_and_weight(coords[subject], coords["screen"])
        photos[subject], intake[subject] = {}, {}
        for light, record in lights.items():
            photos[subject][light], intake[subject][light] = prepare(
                project, output, subject, light, hashes, record, old["admission"][subject][light], areas[subject], alpha)
        print(subject, {light: view["qualified"] for light, view in intake[subject].items()}, flush=True)
    qualified = [name for name, lights in intake.items() if all(view["qualified"] for view in lights.values())]
    identity, cases = {}, {}
    if len(qualified) >= 2:
        identity, cases = compare_workbenches(output, qualified, photos, areas)
    json_to(output / "results.json", dict(parent=FLOAT_REPORT, source_exr_manifest=EXR_SHA, qualified=qualified,
            intake=intake, cases=cases, identity=identity, stable_fingerprint=algorithm_fingerprint(),
            spatial_checkpoint=CHECKPOINT, strength=STRENGTH, script_sha256=digest(script),
            protocol_sha256=digest(script.with_name("gros-local-editor-channel-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    run(**vars(parser.parse_args()))
