"""Read only the fixed LSMI same-preset physical capture members."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
import re
from pathlib import Path
import struct
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib

from PIL import Image

from probe_facd_range import ArchiveWindow


INDEX = "852b32f9c5ac8ce9ba6bea687b4ed21820d93aed62e15ad4824bf2fbf54a4131"
URL = "https://datasets.cvc.uab.cat/color/srgb-lsmi-nikon.zip"
SIZE = 11_401_528_965
ETAG = '"2a7958685-64494a8316c6f"'
SCENES = ("Place916", "Place923", "Place931", "Place938", "Place946", "Place954")
LIGHTS = ("12", "13", "123")
BUDGET = 19_182_394
HEADER_ALLOWANCE = 1024



CATALOG_START = 11_398_119_879
CATALOG_BYTES = 3_408_988
DAYLIGHT = re.compile(r"srgb-lsmi-nikon/input/(Place[0-9]+)_([0-9]+)_D_CS\.png")



def read_range(url, total, etag, start, length):
    end = start + length - 1
    if not 0 <= start <= end < total or length > 4 * 1024**2:
        raise ValueError("Invalid LSMI range")
    headers = {"Range": f"bytes={start}-{end}", "If-Match": etag, "Accept-Encoding": "identity"}
    with urlopen(Request(url, headers=headers), timeout=22) as response:
        if (response.status != 206 or response.headers.get("ETag") != etag
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}"):
            raise ValueError("Unexpected LSMI range response")
        data = response.read(length + 1)
    if len(data) != length:
        raise ValueError("Incomplete LSMI range")
    return data


def catalog(out):
    tail = read_range(URL, SIZE, ETAG, CATALOG_START, SIZE - CATALOG_START)
    directory = {}
    with ZipFile(ArchiveWindow(CATALOG_START, tail)) as archive:
        entries = archive.infolist()
        for member in entries:
            match = DAYLIGHT.fullmatch(member.filename)
            if match is not None:
                name, light = match.groups()
                registry = directory.setdefault(name, {})
                if light in registry:
                    raise ValueError("Duplicate LSMI physical condition")
                registry[light] = dict(path=member.filename, png_bytes=member.file_size,
                                       zip_bytes=member.compress_size, member_offset=member.header_offset,
                                       crc32=f"{member.CRC:08x}", compression=member.compress_type)
    ranges = []
    for first, count in ((SIZE - 22, 22), (SIZE - 42, 20), (SIZE - 98, 56),
                         (CATALOG_START, CATALOG_BYTES)):
        start = first - CATALOG_START
        ranges.append(dict(first=first, last=first + count - 1, bytes=count,
                           sha256=sha256(tail[start:start + count]).hexdigest()))
    result = dict(source=URL, size=SIZE, etag=ETAG, ranges=ranges, index_entries=len(entries),
                  unmasked_daylight_places=len(directory), sets_by_place=directory)
    payload = (json.dumps(result, indent=2) + "\n").encode("utf-8")
    if sha256(payload).hexdigest() != INDEX:
        raise ValueError("Rebuilt archive index differs from the fixed capture index")
    with out.open("xb") as file:
        file.write(payload)


def select(path):
    source = path.read_bytes()
    if sha256(source).hexdigest() != INDEX:
        raise ValueError("Changed LSMI index")
    catalog = json.loads(source)
    if (catalog["source"], catalog["size"], catalog["etag"], catalog["index_entries"]) != (URL, SIZE, ETAG, 24_806):
        raise ValueError("Unexpected LSMI archive")
    eligible = sorted((name for name, settings in catalog["sets_by_place"].items() if set(settings) == set(LIGHTS)),
                      key=lambda name: int(name[5:]))
    if len(eligible) != 39 or tuple(eligible[i * (len(eligible) - 1) // 5] for i in range(6)) != SCENES:
        raise ValueError("Changed deterministic LSMI scene selection")
    selected = {f"{name}-{light}": catalog["sets_by_place"][name][light] for name in SCENES for light in LIGHTS}
    if sum(item["zip_bytes"] + HEADER_ALLOWANCE for item in selected.values()) != BUDGET:
        raise ValueError("Unexpected LSMI transfer size")
    return selected


def from_archive(item):
    block = read_range(URL, SIZE, ETAG, item["member_offset"], item["zip_bytes"] + HEADER_ALLOWANCE)
    header = struct.unpack_from("<4s5H3I2H", block)
    name_size, extra_size = header[-2:]
    start = 30 + name_size + extra_size
    if (header[0] != b"PK\x03\x04" or header[2] & 1 or header[3] != item["compression"]
            or start > HEADER_ALLOWANCE or block[30:30 + name_size].decode("ascii") != item["path"]):
        raise ValueError(f"Unexpected ZIP member layout: {item['path']}")
    data = block[start:start + item["zip_bytes"]]
    if header[3] == ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif header[3] != ZIP_STORED:
        raise ValueError(f"Unsupported compression: {item['path']}")
    if len(data) != item["png_bytes"] or f"{zlib.crc32(data):08x}" != item["crc32"]:
        raise ValueError(f"LSMI member failed integrity validation: {item['path']}")
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (920, 614):
            raise ValueError(f"Unexpected LSMI pixel encoding: {item['path']}")
        image.verify()
    return data


def run(index, out):
    selected = select(index)
    out.mkdir()
    result = dict(index_sha256=INDEX, bytes_transferred=BUDGET, files={})
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {key: pool.submit(from_archive, item) for key, item in selected.items()}
        for key, task in pending.items():
            data = task.result()
            (out / f"{key}.png").write_bytes(data)
            result["files"][key] = dict(**selected[key], sha256=sha256(data).hexdigest())
            print(key, len(data), flush=True)
    with (out / "receipt.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    operations.add_parser("catalog").add_argument("out", type=Path)
    fetch = operations.add_parser("fetch")
    fetch.add_argument("index", type=Path)
    fetch.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.operation == "catalog":
        catalog(args.out)
    else:
        run(args.index, args.out)
