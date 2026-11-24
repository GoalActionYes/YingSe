"""Exercise resized reference matching through the real image endpoints."""
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image
import torch

from probe_gros_local_editor_channel import request, serving
from probe_same_capture_detail_replay import negatives, positives, score
from probe_target_mirror_chroma import STRENGTH, digest, json_to, panel
from same_capture_workbench import make_candidate
from spatial_render import finish_spatial
from spatial_workbench import make_spatial_workbench
from verify_same_capture_workbench import analyze, expected_preview, export, image, preview, upload


SCALES = (("half", 1, 2), ("three-quarters", 3, 4))
FILTERS = ("area", "lanczos")
ADJUSTMENTS = {"basic": {"brightness": 9, "tint": -6}}


def shrink(rgb, numerator, denominator, filter_name):
    height, width = rgb.shape[:2]
    size = (width * numerator // denominator, height * numerator // denominator)
    if filter_name == "area":
        return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return np.array(Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS))


def jpeg(rgb):
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=80, subsampling=2)
    return buffer.getvalue()


def export_pair(server, target_id, reference, out, label, suffix=".png"):
    ref = upload(server, reference, label + suffix)
    mapping = analyze(server, target_id, ref)
    filename = out / (label + ".png")
    checksum = export(server, mapping, target_id, filename)
    meta = server.store.get_mapping_record(mapping)["diagnostics"].get("same_capture_replay", {})
    return mapping, image(filename.read_bytes()), {"sha256": checksum, "replay": meta,
                                                   "reference_sha256": server.store.get_media(ref)["sha256"]}


def check_preview(server, mapping, target_id, out, original, replay):
    raw, adjusted, basis = preview(server, mapping, STRENGTH, ADJUSTMENTS, True)
    np.testing.assert_array_equal(adjusted, finish_spatial(raw, basis, STRENGTH, ADJUSTMENTS))
    np.testing.assert_array_equal(basis, expected_preview(replay, raw.shape))
    before, after, _ = preview(server, mapping, 0., None, False)
    np.testing.assert_array_equal(before, after)
    full = out / "resized-adjusted.png"
    checksum = export(server, mapping, target_id, full, adjustments=ADJUSTMENTS)
    np.testing.assert_array_equal(image(full.read_bytes()), finish_spatial(original, replay, STRENGTH, ADJUSTMENTS))
    project = request(server, "/api/projects", dict(mapping_id=mapping, strength=STRENGTH,
        output_settings={"kind": "png", "adjustments": ADJUSTMENTS}))["project_id"]
    return dict(project=project, mapping=mapping, target=target_id, sha256=checksum)


def run(out):
    torch.set_num_threads(4)
    out.mkdir()
    rows, rejected = {}, {}
    restored = None
    with serving(make_candidate(out / "candidate")) as candidate, serving(
            make_spatial_workbench(out / "stock", enhanced=True)) as stock:
        builds = {"candidate": candidate.build_tag, "stock": stock.build_tag}
        negative_inputs = []
        for name, original, refs, _, expected, zones in positives(out):
            if not name.endswith("-full"):
                continue
            target = {key: upload(server, original, name + "-target.png")
                      for key, server in (("stock", stock), ("candidate", candidate))}
            for scale, numerator, denominator in SCALES:
                for filter_name in FILTERS:
                    label = f"{name}-{scale}-{filter_name}"
                    frames, diagnostics = {}, {}
                    for key, server in (("stock", stock), ("candidate", candidate)):
                        frames[key], diagnostics[key] = {}, {}
                        for state, rgb in refs.items():
                            low = shrink(rgb, numerator, denominator, filter_name)
                            mapping, frames[key][state], diagnostics[key][state] = export_pair(
                                server, target[key], low, out, f"{label}-{key}-{state}")
                            if key == "candidate" and state == "on" and label == "flash-full-half-area":
                                basis, _ = server.store._basis(server.store.get_mapping_record(mapping))
                                if basis is not None:
                                    restored = check_preview(server, mapping, target[key], out, original, basis)
                    scores = {key: score(name, original, expected, result, zones)
                              for key, result in frames.items()}
                    same_off = bool(np.array_equal(frames["candidate"]["off"], original))
                    accepted = all(entry["replay"].get("accepted") for entry in diagnostics["candidate"].values())
                    rows[label] = dict(scores=scores, diagnostics=diagnostics, off_exact=same_off,
                                       passed=bool(accepted and same_off and scores["candidate"]["passed"]),
                                       panel=panel(out / f"{label}-comparison.png",
                                                   {"inputs": {"target": original, "intended": expected}, **frames}))
                    print(label, rows[label]["passed"], flush=True)
            compressed = jpeg(shrink(refs["on"], 1, 2, "area"))
            _, _, a = export_pair(stock, target["stock"], compressed, out, name + "-jpeg-stock", ".jpg")
            _, _, b = export_pair(candidate, target["candidate"], compressed, out, name + "-jpeg-candidate", ".jpg")
            rejected[name + "-jpeg"] = dict(stock=a, candidate=b)
            if name == "flash-full":
                width = original.shape[1]
                moved = original.copy()
                moved[:, width // 2:] = np.roll(original[:, width // 2:], width // 12, axis=1)
                negative_inputs.append(("shifted-half", original, moved))
        negative_inputs.extend(item for item in negatives() if item[0] != "jpeg-flash")
        for name, target, reference in negative_inputs:
            target_ids = {key: upload(server, target, name + "-target.png")
                          for key, server in (("stock", stock), ("candidate", candidate))}
            for scale, numerator, denominator in SCALES:
                for filter_name in FILTERS:
                    low = shrink(reference, numerator, denominator, filter_name)
                    label = f"{name}-{scale}-{filter_name}"
                    row = {}
                    for key, server in (("stock", stock), ("candidate", candidate)):
                        _, _, row[key] = export_pair(server, target_ids[key], low, out, f"{label}-{key}")
                    rejected[label] = row
    if restored is not None:
        with serving(make_candidate(out / "candidate")) as restarted:
            project = request(restarted, "/api/projects/" + restored["project"])
            assert project["mapping_id"] == restored["mapping"] and not project["needs_analysis"]
            assert export(restarted, project["mapping_id"], restored["target"], out / "resized-restored.png",
                          adjustments=project["output_settings"]["adjustments"]) == restored["sha256"]
    rejects_ok = len(rejected) == 70 and all(
        cell["stock"]["sha256"] == cell["candidate"]["sha256"] and not cell["candidate"]["replay"].get("accepted")
        for cell in rejected.values())
    improved = sum(row["passed"] and not row["scores"]["stock"]["passed"] for row in rows.values())
    passed = len(rows) == 8 and all(row["passed"] for row in rows.values()) and rejects_ok and improved and restored
    json_to(out / "results.json", dict(positive=rows, rejected=rejected, restored=restored and restored["sha256"],
            improved=improved, passed=bool(passed), builds=builds, script=digest(Path(__file__)),
            libraries={"opencv": cv2.__version__, "pillow": PIL.__version__},
            protocol=digest(Path(__file__).with_name("same-capture-resize-protocol.md"))))
    if not passed:
        raise SystemExit("Resized replay did not pass the frozen criteria")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
