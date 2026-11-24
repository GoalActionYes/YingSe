"""Replay matching local warmth over two frozen CL3AN white-light captures."""

from argparse import ArgumentParser
import json
from pathlib import Path

from PIL import Image
import numpy as np

from probe_cl3an_real_light_global_fit import source_hash
from probe_hdm_hdr_display import json_to
from warm_capture_preflight import RESEARCH_ROOT, registration, run as preflight


EXPERIMENT = Path(__file__).absolute().parent
PARENT_EVIDENCE = EXPERIMENT.parents[1] / "docs/research/2026-11-21-cl3an-pointwise-bound-evidence.json"
CHROMATICITY = (0.3127, 0.329, 0.64, 0.33, 0.3, 0.6, 0.15, 0.06)


def verify_and_resize(source, expected, prepared):
    if source_hash(source) != expected:
        raise ValueError(f"Unexpected source identity: {source.name}")
    with Image.open(source) as image:
        if (image.format, image.mode, image.size) != ("PNG", "RGB", (6000, 4000)):
            raise ValueError(f"Unexpected source geometry: {source.name}")
        if image.info.get("gamma") != .45455 or image.info.get("chromaticity") != CHROMATICITY:
            raise ValueError(f"Unexpected source display metadata: {source.name}")
        image.thumbnail((1200, 1200), Image.Resampling.BOX)
        image.save(prepared)
        return np.array(image)


def run(base, out):
    base, out = base.absolute(), out.absolute()
    source = base / "cl3an-real-light-input"
    parent = json.loads(PARENT_EVIDENCE.read_bytes())
    geometry_path = EXPERIMENT / "cl3an-neutral-warmth-regions.json"
    geometry = json.loads(geometry_path.read_bytes())
    out.mkdir()
    record = out / "capture.md"
    record.write_bytes(b"CL3AN fixed Canon R6 Mark II camera, 1/60 s, f/11, ISO 100, 6400 K white balance, "
                       b"50 mm. Ambient GT to first direct white SH_IN. Characterization from paper Table 1. "
                       b"Two distinct light rigs; no individual EXIF in release PNG.\n")
    cases = {}
    for scene, regions in geometry.items():
        identity = parent[scene]["source_sha256"]
        first, second = (out / f"{scene}-{kind}.png" for kind in ("ambient", "direct"))
        ambient = verify_and_resize(source / f"{scene}_GT.png", identity["ambient"], first)
        direct = verify_and_resize(source / f"{scene}_1_SH_IN.png", identity["white"], second)
        result = {
            "native_hashes": {key: identity[key] for key in ("ambient", "white")},
            "prepared_hashes": [source_hash(path) for path in (first, second)],
            "alignment": registration(ambient, direct),
        }
        if result["alignment"]["passed"]:
            grade = preflight(first, second, record, out / scene, regions["edit"], [regions["guard"]], base)
            result.update(frames={name: value["metrics"] for name, value in grade["frames"].items()},
                          directions=grade["directions"], matches=grade["matching_directions"],
                          report_sha256=source_hash(out / scene / "results.json"))
        cases[scene] = result
    json_to(out / "results.json", {
        "prior_evidence": source_hash(PARENT_EVIDENCE),
        "protocol": source_hash(EXPERIMENT / "cl3an-neutral-warmth-protocol.md"),
        "regions": source_hash(geometry_path),
        "cases": cases,
    })


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--base", type=Path, default=RESEARCH_ROOT)
    run(**vars(parser.parse_args()))
