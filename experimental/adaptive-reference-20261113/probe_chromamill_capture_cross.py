"""Cross physically captured light color and level with an explicit digital look."""

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path


REPO = "https://huggingface.co/datasets/priubrogent/CHROMA-MILL"
REVISION = "fcb84de6a849931cc7d1583ebcdce3ca2006d2fd"
PART = "phone/train/low"
SCENES = ("Scene1_a", "Scene2_a")
LIGHTS = {"white": "3127-y_3290", "blue": "1968-y_1623"}
LEVELS = (213, 241)
LIMIT = 20 * 1024**2


def source_members(scenes=SCENES):
    return {
        f"{scene}-{light}-{level}": f"{PART}/{scene}-F_90-SS_200-x_{xy}-B_{level}.png"
        for scene, (light, xy), level in product(scenes, LIGHTS.items(), LEVELS)
    }


def read_bytes(session, url, size):
    content = bytearray()
    with session.get(url, stream=True, timeout=(10, 30)) as response:
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code} for a frozen CHROMA-MILL request")
        for block in response.iter_content(65536):
            content.extend(block)
            if len(content) > size:
                raise ValueError("Response exceeds the frozen request size")
    return bytes(content)


def fetch(out, scenes=SCENES, budget=LIMIT):
    import requests
    from urllib.parse import urljoin, urlparse

    with requests.Session() as session:
        api = f"https://huggingface.co/api/datasets/priubrogent/CHROMA-MILL/tree/{REVISION}/{PART}"
        wanted = source_members(scenes)
        by_path = {}
        for _ in range(3):
            with session.get(api, stream=True, timeout=(10, 30)) as page:
                if page.status_code != 200:
                    raise RuntimeError(f"HTTP {page.status_code} for the frozen CHROMA-MILL index")
                data = page.raw.read(600_001, decode_content=True)
                if len(data) > 600_000:
                    raise ValueError("Index exceeds its request budget")
                by_path.update({row["path"]: row for row in json.loads(data)})
                link = page.links.get("next")
                if set(wanted.values()) <= by_path.keys() or link is None:
                    break
                api = urljoin(api, link["url"])
                if urlparse(api).netloc != "huggingface.co" or not api.startswith("https://"):
                    raise ValueError("Unexpected CHROMA-MILL index host")
        files = {}
        for name, member in wanted.items():
            row = by_path[member]
            if row["lfs"]["size"] != row["size"]:
                raise ValueError(f"Inconsistent published size: {member}")
            files[name] = {"path": member, "bytes": row["size"], "sha256": row["lfs"]["oid"]}
        if sum(row["bytes"] for row in files.values()) > budget:
            raise ValueError("Selected assets exceed budget")
        out.mkdir(exist_ok=True)
        for name, row in files.items():
            path = out / f"{name}.png"
            if path.exists():
                content = path.read_bytes()
            else:
                content = read_bytes(session, f"{REPO}/resolve/{REVISION}/{row['path']}", row["bytes"])
            if len(content) != row["bytes"] or sha256(content).hexdigest() != row["sha256"]:
                raise ValueError(f"Mismatched published image: {name}")
            if not path.exists():
                path.write_bytes(content)
    receipt = {"revision": REVISION, "files": files, "bytes": sum(row["bytes"] for row in files.values())}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def decode_rgb8(filename):
    import cv2
    import numpy as np
    from probe_arri_shared_look import quantize

    linear = cv2.imdecode(np.frombuffer(filename.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if linear is None or linear.ndim != 3 or linear.shape[2] != 3 or linear.dtype not in (np.uint8, np.uint16):
        raise ValueError("Expected an 8- or 16-bit color PNG")
    height, width = linear.shape[:2]
    scale = min(1, 512 / max(height, width))
    rgb = linear[..., ::-1].astype(np.float32) / np.iinfo(linear.dtype).max
    rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    encoded = quantize(np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055))
    return encoded, {"shape": list(linear.shape), "dtype": str(linear.dtype)}


def run(media, base, out):
    import cv2
    import numpy as np
    from PIL import Image
    import torch

    from background_intervention import target_regions
    from probe_arri_sdr_renderability import transforms
    from probe_arri_shared_look import LOOK_SHA, converted, digest, metrics, quantize
    from probe_arri_two_looks import CHECKPOINT
    from probe_hdm_hdr_display import json_to
    from probe_inretouch_registration import gradient, shifts
    from probe_middlebury_camera_channel_replay import difference
    from spatial_render import SpatialEngine, finish_spatial

    cv2.setNumThreads(1)
    receipt = json.loads((media / "receipt.json").read_bytes())
    if receipt["revision"] != REVISION or {k: r["path"] for k, r in receipt["files"].items()} != source_members():
        raise ValueError("Changed frozen capture group")
    cube = base / "arri-shared-look-assets-v1/film-a.cube"
    if digest(cube) != LOOK_SHA:
        raise ValueError("Changed Film A look")
    view, inverse, look = transforms(cube)
    out.mkdir()

    def save(key, rgb):
        output = out / f"{key}.png"
        Image.fromarray(rgb).save(output)
        return digest(output)

    captures, prep = {}, {}
    for scene in SCENES:
        captures[scene], prep[scene] = {}, {}
        for light, level in product(LIGHTS, LEVELS):
            key = f"{scene}-{light}-{level}"
            filename = media / f"{key}.png"
            spec = receipt["files"][key]
            if filename.stat().st_size != spec["bytes"] or digest(filename) != spec["sha256"]:
                raise ValueError(f"Changed capture: {key}")
            encoded, format_info = decode_rgb8(filename)
            log = converted(encoded.astype(np.float32) / 255, inverse)
            np.testing.assert_array_equal(encoded, quantize(converted(log, view)))
            styled = quantize(converted(converted(log, look), view))
            captures[scene][light, level] = {"off": encoded, "on": styled}
            prep[scene][f"{light}-{level}"] = format_info | {
                "hashes": {state: save(f"{key}-{state}", image)
                           for state, image in captures[scene][light, level].items()},
            }

    checks, masks = {}, {}
    for scene, photos in captures.items():
        target, styled = photos["white", 241].values()
        whole = np.ones(target.shape[:2], dtype=bool)
        neutral = np.logical_and(*(target_regions(image, ~whole)["neutral_proxy"] for image in (target, styled)))
        masks[scene] = {"whole": whole, "neutral": neutral}
        geometries = {tuple(row["shape"]) for row in prep[scene].values()}
        align, differences = {}, {}
        if len(geometries) == 1:
            for (light, level), frames in photos.items():
                align[f"{light}-{level}"] = shifts(gradient(target), gradient(frames["off"]))[0]
            differences = {
                **{f"color-{level}": difference((photos["blue", level]["off"].astype(float)
                                                  - photos["white", level]["off"]).reshape(-1, 3)) for level in LEVELS},
                **{f"level-{light}": difference((photos[light, 241]["off"].astype(float)
                                                  - photos[light, 213]["off"]).reshape(-1, 3)) for light in LIGHTS},
            }
        checks[scene] = {"neutral_pixels": int(neutral.sum()), "alignment": align, "input_effects": differences,
                         "qualified": (len(align) == 4 and neutral.sum() >= 256
                                       and all(row["zero"] >= .60 and row["peak_gain"] <= .02 for row in align.values())
                                       and all(row["rgb"] >= 1 for row in differences.values()))}
    cases = {}
    if all(case["qualified"] for case in checks.values()):
        torch.set_num_threads(4)
        engine = SpatialEngine("multiref-v1")
        if engine.checkpoint_sha != CHECKPOINT:
            raise ValueError("Changed 13 checkpoint")
        for scene, donor in (SCENES, SCENES[::-1]):
            target, authored = captures[scene]["white", 241].values()
            truth = finish_spatial(target, authored, .8)
            views, files = {}, {}
            for (light, level), frames in captures[donor].items():
                views[light, level] = {
                    state: finish_spatial(target, engine.prepare(target, guide).apply(target), .8)
                    for state, guide in frames.items()
                }
                for state, grade in views[light, level].items():
                    tag = f"{light}-{level}-{state}"
                    files[tag] = save(f"{scene}-{tag}-13", grade)
            regions = {}
            for label, area in masks[scene].items():
                factors = {}
                for state in ("off", "on"):
                    w0, w1, b0, b1 = (views[key][state][area].astype(float) for key in product(LIGHTS, LEVELS))
                    factors[state] = {
                        "color-213": difference(b0 - w0), "color-241": difference(b1 - w1),
                        "level-white": difference(w1 - w0), "level-blue": difference(b1 - b0),
                        "interaction": difference(b1 - b0 - w1 + w0),
                    }
                styles = {f"{light}-{level}": metrics(target, truth, *pair.values(), area)
                          for (light, level), pair in views.items()}
                valid = all(data["passed"] for data in styles.values())
                valid &= all(data["rgb"] <= 3 for state in factors.values() for tag, data in state.items()
                             if tag != "interaction")
                regions[label] = {"styles": styles, "factors": factors, "passed": valid}
            cases[scene] = {"reference_scene": donor, "regions": regions, "hashes": files,
                            "passed": all(region["passed"] for region in regions.values())}
    json_to(out / "results.json", {"source": receipt, "model_sha256": CHECKPOINT, "look_sha256": LOOK_SHA,
                                   "prepared": prep, "checks": checks, "cases": cases,
                                   "passed": len(cases) == 2 and all(row["passed"] for row in cases.values())})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="mode", required=True)
    sub.add_parser("fetch").add_argument("out", type=Path)
    probe = sub.add_parser("run")
    for item in ("media", "base", "out"):
        probe.add_argument(item, type=Path)
    args = vars(cli.parse_args())
    mode = args.pop("mode")
    (fetch if mode == "fetch" else run)(**args)
