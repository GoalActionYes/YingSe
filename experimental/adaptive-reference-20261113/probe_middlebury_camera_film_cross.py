"""Cross a known Film A edit with two cameras photographing the same chart."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile


URL = "https://vision.middlebury.edu/color/data/checker/zip/checker24s-RAW-JPG.zip"
SIZE = 42_359_108
ETAG = '"sha256-420534c8d56cfdd896241b2e96ed34e69df8e6de6a519e56489fe7de7826df74"'
MEMBERS = {"canon": ("CanonEOS1DsMarkII", 93439), "nikon": ("NikonD70", 101162)}
PARENT_SHA = "e09286501decd0d41cf727bce645d9920298da166f6bdaf4bab20385d27d730e"


class CameraArchive(io.RawIOBase):
    def __init__(self, session):
        self.session = session
        self.position = 0
        self.bytes = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, SIZE)[whence] + offset
        if not 0 <= self.position <= SIZE:
            raise ValueError("Out-of-bounds ZIP seek")
        return self.position

    def read(self, count=-1):
        length = SIZE - self.position
        count = length if count < 0 else min(count, length)
        if count == 0:
            return b""
        if self.bytes + count > 1024**2:
            raise ValueError("ZIP range budget exceeded")
        start, end = self.position, self.position + count - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        with self.session.get(URL, headers=headers, stream=True, timeout=(8, 18)) as response:
            if (response.status_code != 206 or response.headers.get("ETag") != ETAG
                    or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}"):
                raise ValueError(f"Unverified archive range {start}–{end}, status {response.status_code}")
            body = response.raw.read(count + 1)
        if len(body) != count:
            raise ValueError("Truncated archive range")
        self.position += count
        self.bytes += count
        return body


def fetch(out):
    import requests

    output, files = {}, {}
    with requests.Session() as session:
        reader = CameraArchive(session)
        with zipfile.ZipFile(reader) as archive:
            for key, (camera, size) in MEMBERS.items():
                path = f"checker24s/{camera}/wb1i1e3-jpg.png"
                member = archive.getinfo(path)
                if archive.namelist().count(path) != 1 or member.file_size != size or member.flag_bits & 1:
                    raise ValueError(f"Unexpected archive member {path}")
                data = archive.read(member)
                output[key] = data
                files[key] = {"member": path, "bytes": size, "crc32": f"{member.CRC:08x}",
                              "sha256": sha256(data).hexdigest()}
    out.mkdir()
    for key, data in output.items():
        (out / f"{key}.png").write_bytes(data)
    receipt = {"archive": URL, "size": SIZE, "etag": ETAG, "files": files, "range_bytes": reader.bytes}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def run(source, base, out):
    import numpy as np
    from PIL import Image, ImageDraw
    import torch

    from background_intervention import target_regions
    from probe_arri_film_b_dose import verified
    from probe_arri_sdr_renderability import transforms
    from probe_arri_shared_look import LOOK_SHA, converted, digest, metrics, quantize
    from probe_arri_two_looks import CHECKPOINT, chart_masks
    from probe_hdm_hdr_display import json_to
    from probe_hdm_hdr_equal_luminance import verified_json
    from probe_hdm_poker_material_roi import REGIONS
    from probe_psbench_background_desaturation import REC709
    from spatial_render import ROOT, SpatialEngine, finish_spatial
    from triplet_color import to_srgb

    data = json.loads((source / "receipt.json").read_bytes())
    parent = base / "arri-two-looks-v2"
    existing = verified_json(parent / "results.json", PARENT_SHA)
    cube = base / "arri-shared-look-assets-v1/film-a.cube"
    if data["etag"] != ETAG or digest(cube) != LOOK_SHA or existing["weight_sha256"] != CHECKPOINT:
        raise ValueError("Frozen lineage changed")
    display, inverse, film = transforms(cube)
    out.mkdir()

    def save(name, rgb):
        path = out / f"{name}.png"
        Image.fromarray(rgb).save(path)
        return digest(path)

    refs, input_score = {}, {}
    for key, row in data["files"].items():
        path = source / f"{key}.png"
        if digest(path) != row["sha256"]:
            raise ValueError(f"Changed {key} image")
        with Image.open(path) as image:
            srgb, profile = to_srgb(image)
            original = np.array(srgb)
        log = converted(original.astype(np.float32) / 255, inverse)
        np.testing.assert_array_equal(quantize(converted(log, display)), original)
        edited = quantize(np.clip(converted(converted(log, film), display), 0, 1))
        style_mae = float(np.abs(edited.astype(float) - original).mean())
        if style_mae < 1:
            raise ValueError("Insufficient known Film A edit")
        refs[key] = (original, edited)
        input_score[key] = {"shape": list(original.shape), "profile": profile["policy"], "film_edit_mae": style_mae,
                            "edited_sha256": save(f"{key}-film", edited)}
    first, second = (refs[key][0] for key in MEMBERS)
    cross_input = float(np.abs(first.astype(float) - second).mean())
    if first.shape != second.shape or cross_input < 1:
        raise ValueError("Missing comparable camera separation")

    torch.set_num_threads(4)
    engine = SpatialEngine("multiref-v1")
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError("Changed 13 weights")
    results = {}
    for scene in ("poker_fullshot", "hdr_testimage"):
        files = existing["preparation"]["film_a"][scene]["images"]
        target, authored = (verified(parent / "film_a" / f"{scene}-{role}.png", files[role])
                            for role in ("off", "on"))
        expected = finish_spatial(target, authored, .8)
        whole = np.ones(target.shape[:2], dtype=bool)
        masks = {"whole": whole, "neutral": np.logical_and(*(target_regions(rgb, ~whole)["neutral_proxy"]
                                                             for rgb in (target, authored)))}
        if scene == "hdr_testimage":
            masks.update({key: value for key, value in chart_masks(ROOT, {"off": target, "on": authored}).items()
                          if key.startswith("skin_")})
        else:
            for key in ("cloth_left", "cloth_right"):
                area = Image.new("L", (whole.shape[1], whole.shape[0]))
                ImageDraw.Draw(area).polygon(REGIONS[key], fill=255)
                masks[key] = np.asarray(area) != 0
        if len(masks) != 4 or min(area.sum() for area in masks.values()) < 32 or masks["neutral"].sum() < 256:
            raise ValueError("Insufficient frozen measurement support")
        outputs, hashes = {}, {}
        for camera, pictures in refs.items():
            outputs[camera], hashes[camera] = {}, {}
            for role, reference in zip(("off", "on"), pictures, strict=True):
                grade = finish_spatial(target, engine.prepare(target, reference).apply(target), .8)
                outputs[camera][role] = grade
                hashes[camera][role] = save(f"{scene}-{camera}-{role}", grade)
        scores = {}
        for name, area in masks.items():
            style = {camera: metrics(target, expected, *grades.values(), area) for camera, grades in outputs.items()}
            camera_effect = {}
            for role in ("off", "on"):
                change = outputs["nikon"][role][area].astype(float) - outputs["canon"][role][area]
                y = change @ REC709
                camera_effect[role] = {"rgb_mae": float(np.abs(change).mean()), "luma_mae": float(np.abs(y).mean()),
                                       "chroma_mae": float(np.abs(change - y[:, None]).mean())}
            scores[name] = {"pixels": int(area.sum()), "style": style, "camera_effect": camera_effect}
        passed = (all(arm["passed"] for name in ("whole", "neutral") for arm in scores[name]["style"].values())
                  and all(arm["rgb_mae"] <= 3 for row in scores.values() for arm in row["camera_effect"].values()))
        results[scene] = {"hashes": hashes, "regions": scores, "passed": passed}
    json_to(out / "results.json", {"source": data, "target_ledger_sha256": PARENT_SHA, "look_sha256": LOOK_SHA,
                                   "model_sha256": CHECKPOINT, "inputs": input_score, "input_camera_mae": cross_input,
                                   "results": results, "passed": all(row["passed"] for row in results.values())})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="mode", required=True)
    commands.add_parser("fetch").add_argument("out", type=Path)
    probe = commands.add_parser("run")
    for name in ("source", "base", "out"):
        probe.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    mode = args.pop("mode")
    (fetch if mode == "fetch" else run)(**args)
