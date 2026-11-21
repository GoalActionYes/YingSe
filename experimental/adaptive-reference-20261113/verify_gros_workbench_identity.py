"""Verify the Gros development benchmark through the 13 workbench HTTP API."""

from argparse import ArgumentParser
import base64
import hashlib
import io
import json
from pathlib import Path
import threading
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, PngImagePlugin
import torch

from probe_gros_film_b_cross import ARCHIVE, GROUPS, LIGHTS, masks, score
from probe_target_mirror_chroma import BASE, CHECKPOINT, STRENGTH, checked_json, checked_rgb, digest, json_to
from reference_color_app.adjustments import transform
from spatial_workbench import make_spatial_workbench

PARENT = "4413ee3468586d3ab5e715013c5e86d9c24412781e8385e75d5664b6e50df260"
MANUAL = {"basic": {"brightness": 12}}


def rgb(png):
    with Image.open(io.BytesIO(png)) as image:
        return np.asarray(image.convert("RGB"))


def preview_rgb(uri):
    return rgb(base64.b64decode(uri.split(",", 1)[1]))


def deviation(output, expected):
    difference = np.abs(output.astype(np.int16) - expected)
    return {"max": int(difference.max()), "mae": float(difference.mean())}


def exercise(out, server, parent, root, regions):
    origin = f"http://127.0.0.1:{server.server_port}"

    def request(path, data=None, upload_name=None):
        headers = {"Content-Type": "application/json"}
        if upload_name:
            headers = {"Content-Type": "application/octet-stream", "X-Filename": quote(upload_name)}
        elif data is not None:
            data = json.dumps(data).encode("utf-8")
        with urlopen(Request(origin + path, data=data, headers=headers), timeout=30) as response:
            return json.load(response) if response.headers.get_content_type() == "application/json" else response.read()

    def upload(path):
        return request("/api/media", path.read_bytes(), path.name)["id"]

    def analyze(target, reference):
        response = request("/api/analyze", {"target_id": target, "reference_id": reference})
        mapping = response['mapping_id']
        return mapping, server.store.get_mapping_record(mapping)['diagnostics']

    def export(mapping, target, name, settings=None):
        job = request("/api/export", dict(mapping_id=mapping, media_id=target, strength=STRENGTH,
                                          kind="png", adjustments=settings))
        deadline = time.monotonic() + 90
        while job["status"] in ("queued", "running"):
            if time.monotonic() >= deadline:
                raise TimeoutError("PNG export did not finish")
            time.sleep(.05)
            job = request("/api/jobs/" + job["id"])
        if job["status"] != "done":
            raise RuntimeError(job)
        data = request(job["result_url"])
        (out / f"{name}.png").write_bytes(data)
        return rgb(data), hashlib.sha256(data).hexdigest()

    assert request("/api/health")["variant"] == "multiref-temporal-v1"
    source_ids = {}
    originals = {}
    for key, states in parent["inputs"].items():
        for state, expected_hash in states["hashes"].items():
            name = f"{key}-{state}"
            path = root / f"{name}.png"
            originals[name] = checked_rgb(path, expected_hash)
            source_ids[name] = upload(path)
    controls = {}
    for person in GROUPS:
        name = f"{person}-cold-off"
        original = originals[name]
        duplicate = out / f"{person}-separately-encoded.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("Description", "Different file bytes, same RGB")
        Image.fromarray(original).save(duplicate, pnginfo=info)
        assert digest(duplicate) != digest(root / f"{name}.png")
        target = source_ids[name]
        mapping, diagnostics = analyze(target, upload(duplicate))
        assert diagnostics["exact_identity"]
        normal = request("/api/preview", dict(mapping_id=mapping, strength=STRENGTH))
        basis = request("/api/preview", dict(mapping_id=mapping, strength=STRENGTH, include_basis=True))
        for frame in (normal["before"], normal["after"], basis["before"], basis["after"], basis["basis"]["full"]):
            np.testing.assert_array_equal(preview_rgb(frame), original)
        picture, plain_sha = export(mapping, target, f"{person}-same-normal")
        manual, manual_sha = export(mapping, target, f"{person}-same-manual", MANUAL)
        expected = np.rint(transform(original.astype(np.float32) / 255, MANUAL) * 255).astype(np.uint8)
        np.testing.assert_array_equal(picture, original)
        np.testing.assert_array_equal(manual, expected)
        assert not np.array_equal(manual, original)
        controls[person] = dict(exact_identity=True, same_preview_and_basis=True, plain_sha=plain_sha,
                                manual_sha=manual_sha, manual_delta=deviation(manual, original))

    cases = {}
    for name, row in parent["cases"].items():
        person, donor = name.split("-from-")
        target_name = f"{person}-cold-off"
        target = source_ids[target_name]
        original = originals[target_name]
        expected = checked_rgb(root / f"{name}-expected.png", row["expected_sha"])
        outputs = {}
        comparison = {}
        for light in LIGHTS:
            outputs[light] = {}
            for state in ("off", "on"):
                suffix = f"{light}-{state}"
                mapping, diagnostics = analyze(target, source_ids[f"{donor}-{suffix}"])
                actual, sha = export(mapping, target, f"{name}-{suffix}")
                frozen = checked_rgb(root / f"{name}-13-{suffix}.png", row["hashes"][f"13-{suffix}"])
                outputs[light][state] = actual
                comparison[suffix] = dict(sha256=sha, from_frozen=deviation(actual, frozen),
                                          exact_identity=diagnostics["exact_identity"],
                                          aligned=diagnostics["certified_aligned"]["accepted"])
        zones = masks(regions[person]["cold"], regions["screen"], original.shape[:2])
        cases[name] = dict(comparison=comparison, scores=score(original, expected, outputs, zones))
    return controls, cases


def main(out):
    root = BASE / "gros-film-b-cross-v2"
    parent = checked_json(root / "results.json", PARENT)
    source = checked_json(BASE / "gros-portrait-light-transfer-v1" / "results.json", ARCHIVE)
    labels = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(labels) != source["regions_sha256"] or parent["strength"] != STRENGTH:
        raise ValueError("Changed registration or strength")
    regions = json.loads(labels.read_bytes())
    out.mkdir()
    torch.set_num_threads(4)
    server = make_spatial_workbench(out / "data", enhanced=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if server.store.engine.checkpoint_sha != CHECKPOINT:
            raise ValueError("Changed checkpoint")
        controls, cases = exercise(out, server, parent, root, regions)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    json_to(out / "results.json", dict(parent=PARENT, build=server.build_tag, checkpoint=CHECKPOINT,
                                       strength=STRENGTH, controls=controls, cases=cases))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    main(parser.parse_args().out)
