"""Retrieve four fixed SIDD originals without downloading whole camera archives."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import struct
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib


REVISION = "6d0169e354f11ce3d98c1ffa214f9acf9d373155"
ROOT = f"https://huggingface.co/datasets/MhdMohammadi/SIDD-CC-DNG/resolve/{REVISION}/data/train"
ARCHIVES = {
    "0028": (1910004432, "f2b0a9378fffa8de204254968b13aa5097705a014d2e6dc77b70abfd3a6758b8",
             "2fac5ccbca0881b390b4aaae3b9447e11beca716887c2c7f46eb8d00657b4103"),
    "0033": (1968124850, "20362894ab135c2ce9b9d065ebf5f034c632b2097826667f3b1531f3aabef8bd",
             "d7f44e1ffc3c318da13a2c5c168ea5084f0a640566f709c7f26eee67b28c2215"),
}
TAIL = 65536
HEADER_ALLOWANCE = 1024
LIMIT = 64 * 1024**2


def read_range(session, instance, start, count):
    size, linked_hash, cdn_etag = ARCHIVES[instance]
    end = start + count - 1
    if not 0 <= start <= end < size or count > LIMIT:
        raise ValueError("Invalid SIDD archive range")
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    with session.get(f"{ROOT}/{instance}_DNG.zip", headers=headers, stream=True, timeout=(12, 40)) as response:
        upstream = response.history[0].headers if response.history else {}
        if (response.status_code != 206
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}"
                or upstream.get("x-repo-commit") != REVISION
                or upstream.get("x-linked-etag") != f'"{linked_hash}"'
                or response.headers.get("ETag") != f'"{cdn_etag}"'):
            raise ValueError(f"Unverified SIDD response for {instance}: HTTP {response.status_code}")
        data = response.raw.read(count + 1)
    if len(data) != count:
        raise ValueError(f"Truncated SIDD range for {instance}")
    return data


def selected_members(instance, tail):
    pattern = re.compile(fr"{instance}_DNG/{instance}_IMG_\d{{4}}(?:_1)?\.DNG", re.IGNORECASE)
    with ZipFile(BytesIO(tail)) as archive:
        members = sorted((item for item in archive.infolist() if pattern.fullmatch(item.filename)),
                         key=lambda item: item.filename)
    if len(members) != 150 or len({item.filename.casefold() for item in members}) != 150:
        raise ValueError(f"Expected exactly 150 original DNGs in {instance}")
    return [members[0], members[75]]


def extract_member(raw, item):
    signature, _, flags, codec, _, _, _, _, _, name_size, extra_size = struct.unpack_from("<4s5H3I2H", raw)
    length = 30 + name_size + extra_size
    name = raw[30:30 + name_size].decode("ascii")
    if (signature != b"PK\x03\x04" or flags & 1 or codec != item.compress_type
            or name != item.filename or length > HEADER_ALLOWANCE):
        raise ValueError(f"Unexpected SIDD member header: {item.filename}")
    payload = raw[length:length + item.compress_size]
    if codec == ZIP_DEFLATED:
        payload = zlib.decompress(payload, -15)
    elif codec != ZIP_STORED:
        raise ValueError(f"Unsupported SIDD member compression: {codec}")
    if len(payload) != item.file_size or zlib.crc32(payload) != item.CRC:
        raise ValueError(f"SIDD member failed integrity checks: {item.filename}")
    return payload


def fetch(output, proxy):
    import requests

    with requests.Session() as session:
        if proxy:
            session.proxies.update({"https": proxy})
        tails = {instance: read_range(session, instance, size - TAIL, TAIL)
                 for instance, (size, _, _) in ARCHIVES.items()}
        selected = {instance: selected_members(instance, tail) for instance, tail in tails.items()}
        planned = len(tails) * TAIL + sum(item.compress_size + HEADER_ALLOWANCE
                                         for items in selected.values() for item in items)
        if planned > LIMIT:
            raise ValueError(f"Selected SIDD member ranges exceed {LIMIT} bytes")
        output.mkdir()
        result = {"revision": REVISION, "bytes_transferred": planned, "scenes": {}}
        for instance, members in selected.items():
            size, linked_hash, cdn_etag = ARCHIVES[instance]
            receipt = dict(archive_bytes=size, linked_sha256=linked_hash, delivery_etag=cdn_etag,
                           tail_sha256=sha256(tails[instance]).hexdigest(), members=[])
            for item in members:
                start = item.header_offset + size - TAIL
                count = item.compress_size + HEADER_ALLOWANCE
                payload = extract_member(read_range(session, instance, start, count), item)
                filename = Path(item.filename).name
                (output / filename).write_bytes(payload)
                receipt["members"].append(dict(filename=filename, bytes=len(payload),
                                               sha256=sha256(payload).hexdigest(), crc32=f"{item.CRC:08x}",
                                               requested_range=[start, start + count - 1]))
                print(instance, filename, len(payload), flush=True)
            result["scenes"][instance] = receipt
    with (output / "receipt.json").open("w", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--proxy")
    args = parser.parse_args()
    fetch(args.output, args.proxy)
