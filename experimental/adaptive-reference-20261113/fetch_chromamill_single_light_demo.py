"""Fetch fixed CHROMA-MILL single-light demo assets without running JavaScript."""
from argparse import ArgumentParser
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image
import requests


REVISION = "62fe4a3a33cd50ff39d94d20111146e3bf4d2d88"
TREE = "37d1b8992f8fed1e060c0f5dc7d580cc494c9d31"
REPOSITORY = "priubrogent/chroma-mill"
SCENES = ("Test13", "Test15", "Test17")
LIGHTS = ("red", "white")
LEVELS = tuple(range(1, 12))
TOTAL = 14_770_698


def receive(url, limit, proxy):
    with requests.get(url, proxies={"https": proxy} if proxy else None,
                      headers={"Accept-Encoding": "identity"}, stream=True, timeout=(12, 24)) as response:
        response.raise_for_status()
        data = response.raw.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Unexpected size: {url}")
    return data


def index(proxy):
    url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{TREE}?recursive=1"
    metadata = json.loads(receive(url, 600_000, proxy))
    if metadata["sha"] != TREE or metadata["truncated"]:
        raise ValueError("Unexpected demo tree")
    by_path = {row["path"]: row for row in metadata["tree"] if row["type"] == "blob"}
    selection = {key: by_path[f"static/mixer/linear_js/{key}.js"]
                 for scene in SCENES for light in LIGHTS for level in LEVELS
                 for key in (f"{scene}/{light}_L{level:02d}",)}
    if sum(row["size"] for row in selection.values()) != TOTAL:
        raise ValueError("Changed selected demo asset size")
    return selection


def extract(key, source, proxy):
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{source['path']}"
    wrapper = receive(url, source["size"], proxy)
    blob = sha1(f"blob {len(wrapper)}\0".encode() + wrapper).hexdigest()
    if len(wrapper) != source["size"] or blob != source["sha"]:
        raise ValueError(f"Changed demo source: {key}")
    escaped = re.escape(key.encode("ascii"))
    assignment = rb'\s*CM\.captures\[["\']' + escaped + rb'["\']\]\s*=\s*["\']data:image/png;base64,([A-Za-z0-9+/=]+)["\'];\s*'
    content = re.fullmatch(assignment, wrapper)
    if content is None:
        raise ValueError(f"Unexpected demo asset: {key}")
    png = b64decode(content[1], validate=True)
    with Image.open(BytesIO(png)) as image:
        if image.size != (600, 400) or image.mode != "RGB" or image.format != "PNG":
            raise ValueError(f"Unexpected demo PNG geometry: {key}")
        image.verify()
    return png, dict(path=source["path"], wrapper_bytes=len(wrapper), git_blob_sha1=blob,
                     bytes=len(png), png_sha256=sha256(png).hexdigest())


def fetch(out, proxy):
    selection = index(proxy)
    out.mkdir()
    receipt = dict(revision=REVISION, tree=TREE, asset_bytes=TOTAL, files={})
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {key: pool.submit(extract, key, row, proxy) for key, row in selection.items()}
        for key, future in futures.items():
            png, identity = future.result()
            dst = out / f"{key}.png"
            dst.parent.mkdir(exist_ok=True)
            dst.write_bytes(png)
            receipt["files"][key] = identity
            print(key, len(png), flush=True)
    with (out / "receipt.json").open("w", encoding="utf-8", newline="\n") as file:
        json.dump(receipt, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--proxy")
    args = parser.parse_args()
    fetch(args.out, args.proxy)
