"""Verify image replay through the actual candidate preview and export routes."""
from argparse import ArgumentParser
import base64
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import monotonic, sleep
from urllib.error import HTTPError

import numpy as np
from PIL import Image

from probe_gros_local_editor_channel import request, serving
from probe_same_capture_detail_replay import BASE, negatives, positives, score
from probe_target_mirror_chroma import STRENGTH, digest, json_to
from same_capture_workbench import make_candidate
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench


VIDEO_SHA = "66b424854ec21a91280cdba42c403288b68be030a119f5b486163804652c28d9"


def image(data):
    with Image.open(BytesIO(data)) as parsed:
        return np.array(parsed.convert("RGB"))


def upload(server, value, name):
    if isinstance(value, np.ndarray):
        buffer = BytesIO()
        Image.fromarray(value).save(buffer, format="PNG")
        value = buffer.getvalue()
    return request(server, "/api/media", value, name)["id"]


def analyze(server, target, ref):
    return request(server, "/api/analyze", dict(target_id=target, reference_id=ref))["mapping_id"]


def export(server, mapping, target, path, strength=STRENGTH, adjustments=None):
    body = dict(mapping_id=mapping, media_id=target, strength=strength, kind=path.suffix[1:])
    if adjustments is not None:
        body["adjustments"] = adjustments
    reply = request(server, "/api/export", body)
    deadline = monotonic() + 90
    while reply["status"] in ("queued", "running"):
        if monotonic() > deadline:
            raise TimeoutError(f"Export failed to finish: {path.name}")
        sleep(.05)
        reply = request(server, "/api/jobs/" + reply["id"])
    if reply["status"] != "done":
        raise RuntimeError(reply)
    value = request(server, reply["result_url"])
    path.write_bytes(value)
    return sha256(value).hexdigest()


def preview(server, mapping, strength, adjustments, interactive):
    args = dict(mapping_id=mapping, strength=strength, adjustments=adjustments, interactive=interactive,
                include_basis=True)
    result = request(server, "/api/preview", args)
    return [image(base64.b64decode(value.split(",", 1)[1]))
            for value in (result["before"], result["after"], result["basis"]["full"])]


def expected_preview(full, dimensions):
    picture = Image.fromarray(full)
    picture.thumbnail((dimensions[1], dimensions[0]), Image.Resampling.BILINEAR)
    return np.array(picture)


def run(out):
    out.mkdir()
    with serving(make_candidate(out / "candidate")) as enabled, serving(
            make_spatial_workbench(out / "stock", enhanced=True)) as stock:
        builds = dict(candidate=enabled.build_tag, stock=stock.build_tag)
        positive, rejected = {}, {}
        restored = None
        adjustments = {"basic": {"brightness": 9, "tint": -6}}
        for name, original, references, _, expected, areas in positives(out):
            target = upload(enabled, original, name + "-target.png")
            results, hashes = {}, {}
            for mode, reference in references.items():
                source = upload(enabled, reference, name + "-" + mode + ".png")
                mapping = analyze(enabled, target, source)
                path = out / f"{name}-{mode}.png"
                hashes[mode] = export(enabled, mapping, target, path)
                results[mode] = image(path.read_bytes())
                if mode == "on" and name == "skin-full":
                    restored = dict(project=request(enabled, "/api/projects", dict(mapping_id=mapping, strength=STRENGTH,
                                    output_settings={"kind": "png", "adjustments": adjustments}))["project_id"],
                                    mapping=mapping, target=target)
                    path = out / "skin-full-adjusted.png"
                    restored["sha256"] = export(enabled, mapping, target, path, adjustments=adjustments)
                    np.testing.assert_array_equal(image(path.read_bytes()), finish_spatial(original, reference, STRENGTH, adjustments))
                    raw_preview, adjusted_preview, basis = preview(enabled, mapping, STRENGTH, adjustments, True)
                    np.testing.assert_array_equal(adjusted_preview, finish_spatial(raw_preview, basis, STRENGTH, adjustments))
                    np.testing.assert_array_equal(basis, expected_preview(reference, raw_preview.shape))
                    neutral, ungraded, _ = preview(enabled, mapping, 0., None, True)
                    np.testing.assert_array_equal(neutral, ungraded)
                    try:
                        request(enabled, "/api/export", dict(mapping_id=mapping, media_id=target, strength=STRENGTH, kind="cube"))
                    except HTTPError as exc:
                        assert exc.code == 400
                    else:
                        raise AssertionError("Spatial candidate exported a regular CUBE")
            outcome = score(name, original, expected, results, areas)
            assert outcome["passed"] and np.array_equal(results["off"], original), name
            positive[name] = dict(outcome=outcome, exports=hashes)
            print("positive", name, flush=True)
        for name, original, reference in negatives():
            results = {}
            for label, server in (("stock", stock), ("candidate", enabled)):
                target = upload(server, original, name + "-input.png")
                source = upload(server, reference, name + "-reference.png")
                mapping = analyze(server, target, source)
                path = out / f"{name}-{label}.png"
                checksum = export(server, mapping, target, path)
                results[label] = dict(sha256=checksum, accepted=bool(server.store.get_mapping_record(mapping)["diagnostics"]
                                           .get("same_capture_replay", {}).get("accepted")))
            exact = results["stock"]["sha256"] == results["candidate"]["sha256"]
            accepted = results["candidate"]["accepted"]
            if name != "jpeg-flash":
                assert exact and not accepted, name
            elif accepted:
                recorded = image((out/f"{name}-candidate.png").read_bytes())
                np.testing.assert_array_equal(recorded, finish_spatial(original, reference, STRENGTH))
            else:
                assert exact
            rejected[name] = dict(results=results, exact=exact)
        movie = BASE / "natural-video/http-multiref-v1/native-short.mp4"
        if digest(movie) != VIDEO_SHA:
            raise ValueError("Changed frozen video control")
        with movie.open("rb") as fixture:
            movie_bytes = fixture.read()
        videos = {}
        for label, server in (("stock", stock), ("candidate", enabled)):
            target = upload(server, movie_bytes, "motion.mp4")
            source = upload(server, original, "motion-ref.png")
            mapping = analyze(server, target, source)
            videos[label] = export(server, mapping, target, out / f"motion-{label}.mp4")
        assert videos["candidate"] == videos["stock"]
    with serving(make_candidate(out / "candidate")) as restarted:
        project = request(restarted, "/api/projects/" + restored["project"])
        assert project["mapping_id"] == restored["mapping"] and not project["needs_analysis"]
        assert export(restarted, project["mapping_id"], restored["target"], out / "restored-adjusted.png",
                      adjustments=project["output_settings"]["adjustments"]) == restored["sha256"]
    json_to(out / "results.json", dict(positive=positive, negative=rejected, restored_sha=restored["sha256"],
            video=videos, builds=builds, script=digest(Path(__file__)),
            protocol=digest(Path(__file__).with_name("same-capture-workbench-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
