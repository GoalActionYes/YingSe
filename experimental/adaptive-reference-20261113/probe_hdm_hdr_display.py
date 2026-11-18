"""Verify temporal candidates and prepare two fixed HdM display RGB controls."""
import argparse
from ftplib import FTP
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw

from fetch_hdm_hdr_paired_stills import remote_credentials

MANIFEST_SHA = "2e4cc59b231c44ddc489842bdbdeb9482930d270b9574ef87b207a38d06bf751"
SOURCE_TO_REC709 = np.array([[1.617523436306807, -.070572740897816, -.021101728042793],
                             [-.537286622188294, 1.334613062330328, -.226953875218266],
                             [-.080236814118512, -.264040321432512, 1.248055603261060]])
REC2020_TO_REC709 = np.array([[1.660491002108435, -.124550474521591, -.018150763354905],
                             [-.587641138788550, 1.132899897125960, -.100578898008007],
                             [-.072849863319885, -.008349422604369, 1.118729661362913]])
LUMA = np.array([.2126, .7152, .0722])
EXPOSURES = {"half": .5, "unit": 1., "double": 2.}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def json_to(path, record):
    path.write_bytes((json.dumps(record, indent=2, allow_nan=False)+"\n").encode("utf-8"))


def original_manifest(folder):
    path = folder/"manifest.json"
    if digest(path) != MANIFEST_SHA:
        raise ValueError("Unexpected frozen HdM archive")
    return json.loads(path.read_bytes())


def fetch_neighbors(source, out, proxy):
    manifest = original_manifest(source)
    host, username, password = remote_credentials(proxy)
    out.mkdir()
    jobs = []
    with FTP(host, timeout=20) as ftp:
        ftp.login(username, password)
        ftp.voidcmd("TYPE I")
        for scene, row in manifest["cases"].items():
            original = row["files"]["original"]
            file = PurePosixPath(original["official_path"])
            for step in (-1, 1):
                frame = original["frame"] + step
                remote = file.with_name(f"{scene}_{frame:06}.exr").as_posix()
                size = ftp.size(remote)
                if size is None or size <= 0:
                    raise ValueError(f"Missing archive neighbor: {remote}")
                jobs.append((scene, step, remote, size))
        if sum(job[3] for job in jobs) > 32*1024**2:
            raise ValueError("Neighbor frame download budget exceeded")
        records = {}
        for scene, step, remote, size in jobs:
            name = f"{scene}-neighbor-{step:+d}.exr"
            dest = out/name
            with dest.open("xb") as handle:
                def write(data):
                    if handle.tell() + len(data) > size:
                        raise ValueError("Archive exceeded reported size")
                    handle.write(data)
                ftp.retrbinary("RETR " + remote, write)
            if dest.stat().st_size != size:
                raise ValueError("Incomplete archive frame")
            records.setdefault(scene, {})[str(step)] = {
                "file": name, "official_path": remote, "sha256": digest(dest), "bytes": size}
    json_to(out/"neighbors.json", records)


def decode_pq(values):
    m1, m2 = 2610/16384, 2523/32
    c1, c2, c3 = 3424/4096, 2413/128, 2392/128
    signal = np.power(values/65535., 1/m2)
    return 10000*np.power(np.maximum(signal-c1, 0)/(c2-c3*signal), 1/m1)


def display(linear):
    y = np.maximum(linear @ LUMA, 0)
    exposed = linear/(100+y[..., None])
    limited = exposed.clip(0, 1)
    encoded = np.where(limited <= .0031308, 12.92*limited, 1.055*limited**(1/2.4)-.055)
    return np.rint(255*encoded).astype(np.uint8), {
        "below_0": float((exposed < 0).mean()), "above_1": float((exposed > 1).mean())}


def prepare(source, neighbors, out):
    import cv2
    from audit_hdm_hdr_still_alignment import gradients, registration, verified_image

    cv2.setNumThreads(1)
    manifest = original_manifest(source)
    nearby = json.loads((neighbors/"neighbors.json").read_bytes())
    out.mkdir()
    cases = {}
    for scene, row in manifest["cases"].items():
        raw, graded = (verified_image(source, row["files"][role]) for role in ("original", "graded"))
        target = gradients(graded)
        hypotheses = {"0": registration(gradients(raw), target)}
        for step, info in nearby[scene].items():
            image = verified_image(neighbors, info)
            hypotheses[step] = registration(gradients(image), target)
        center = hypotheses["0"]
        margin = center["zero"] - max(hypotheses[step]["zero"] for step in ("-1", "1"))
        verified = center["passed"] and margin >= .002
        scene_rgb = cv2.resize(raw[..., ::-1], (512, 288), interpolation=cv2.INTER_AREA) @ SOURCE_TO_REC709
        graded_rgb = cv2.resize(decode_pq(graded[..., ::-1]), (512, 288), interpolation=cv2.INTER_AREA) @ REC2020_TO_REC709
        paths = {}
        for role, linear in (("graded", graded_rgb), *[(key, gain*scene_rgb) for key, gain in EXPOSURES.items()]):
            png = out/f"{scene}-{role}.png"
            pixels, clipping = display(linear)
            Image.fromarray(pixels).save(png)
            paths[role] = {"file": png.name, "sha256": digest(png), **clipping}
        cases[scene] = {"correlations": hypotheses, "margin": margin, "temporal_verified": verified, "images": paths}
    json_to(out/"prepared.json", {"cases": cases, "strict_temporal_requirement": .002,
                                  "neighborhood_source_sha256": digest(neighbors/"neighbors.json")})


def render(prepared, out):
    import torch
    from spatial_render import SpatialEngine, finish_spatial

    torch.set_num_threads(4)
    meta = json.loads((prepared/"prepared.json").read_bytes())
    known = meta["cases"]
    model = SpatialEngine("multiref-v1")
    if model.checkpoint_sha != "d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9":
        raise ValueError("Unexpected frozen checkpoint")
    out.mkdir()
    results = {}
    panels = Image.new("RGB", (4*272, len(known)*(177*len(EXPOSURES))), "#222730")
    labels = ImageDraw.Draw(panels)

    def photo(records, role):
        path = prepared/records[role]["file"]
        if digest(path) != records[role]["sha256"]:
            raise ValueError("Display image changed")
        return np.array(Image.open(path).convert("RGB"))

    for index, scene in enumerate(known):
        donor = next(s for s in known if s != scene)
        guide = photo(known[donor]["images"], "graded")
        target_record = known[scene]
        rows = {}
        for i, exposure in enumerate(EXPOSURES):
            target = photo(target_record["images"], exposure)
            result = finish_spatial(target, model.prepare(target, guide).apply(target), .8)
            output = out/f"{scene}-{exposure}-13.png"
            Image.fromarray(result).save(output)
            expected = photo(target_record["images"], "graded")
            scores = None
            if target_record["temporal_verified"]:
                raw_difference = np.abs(target.astype(np.int16)-expected)
                model_difference = np.abs(result.astype(np.int16)-expected)
                neutral = (np.ptp(target, axis=-1) <= 12) & (target.min(-1) >= 32) & (target.max(-1) <= 240)
                scores = {"whole": [float(x.mean()) for x in (raw_difference, model_difference)],
                          "neutral_pixels": int(neutral.sum()),
                          "neutral": [float(x[neutral].mean()) for x in (raw_difference, model_difference)]
                          if neutral.sum() >= 256 else None}
            rows[exposure] = {"output_sha256": digest(output), "mae": scores}
            top = (index*len(EXPOSURES)+i)*177
            for col, (role, pixels) in enumerate(zip(("input", "reference", "13", "archival grade"),
                                                     (target, guide, result, expected), strict=True)):
                left = col*272 + 8
                labels.text((left, top+5), f"{scene} / {exposure} / {role}", fill="white")
                panels.paste(Image.fromarray(pixels).resize((256, 144), Image.Resampling.LANCZOS), (left, top+23))
        results[scene] = {"donor": donor, "exposures": rows}
    panels.save(out/"comparison.png")
    json_to(out/"results.json", {"prepared_sha256": digest(prepared/"prepared.json"),
                                 "model_sha256": model.checkpoint_sha, "scores": results,
                                 "comparison_sha256": digest(out/"comparison.png")})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("fetch-neighbors", "prepare", "render"))
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--neighbors", type=Path)
    cli.add_argument("--proxy")
    args = cli.parse_args()
    if args.stage == "fetch-neighbors":
        fetch_neighbors(args.source, args.output, args.proxy)
    elif args.stage == "prepare":
        prepare(args.source, args.neighbors, args.output)
    else:
        render(args.source, args.output)
