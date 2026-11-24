"""Read the four registered white-background sequences from the MLS archive."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import struct
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib

from fetch_mls_light_pilot import ARCHIVE_SIZE, INDEX_SHA, fetch


SCENES = ("02", "03", "05", "22")
ANCHOR = "2HAL_DESK"
LIGHTS = ("2HAL", ANCHOR, *(f"{ANCHOR}_{color}{power}"
                             for color in ("R", "RG", "LED-BG", "LED-B")
                             for power in ("025", "050", "075", "100")))
BUDGET = 18_837_473


def entries(index):
    if sha256(index).hexdigest() != INDEX_SHA:
        raise ValueError("Changed official ZIP directory")
    with ZipFile(BytesIO(index)) as archive:
        selected = {(scene, light): archive.getinfo(f"images_preview/{scene}/{scene}_{light}.jpg")
                    for scene in SCENES for light in LIGHTS}
    total = sum(item.compress_size + 1024 for item in selected.values())
    if total != BUDGET:
        raise ValueError("Changed MLS preview download budget")
    return selected


def fetch_member(entry, directory_size):
    offset = entry.header_offset + ARCHIVE_SIZE - directory_size
    raw = fetch(offset, entry.compress_size + 1024)
    sig, _, flags, method, _, _, _, _, _, name_size, extra = struct.unpack("<4s5H3L2H", raw[:30])
    start = 30 + name_size + extra
    if (sig != b"PK\x03\x04" or start > 1024 or flags & 1 or method != entry.compress_type
            or raw[30:30 + name_size].decode() != entry.filename):
        raise ValueError(f"Invalid ZIP member header: {entry.filename}")
    payload = raw[start:start + entry.compress_size]
    if method == ZIP_DEFLATED:
        return zlib.decompress(payload, -15)
    if method == ZIP_STORED:
        return payload
    raise ValueError(f"Unexpected ZIP compression type: {method}")


def validated(entry, photo):
    if len(photo) != entry.file_size or zlib.crc32(photo) != entry.CRC:
        raise ValueError(f"Unexpected MLS member: {entry.filename}")
    return sha256(photo).hexdigest()


def run(index_file, cache, out):
    index = index_file.read_bytes()
    photos = entries(index)
    out.mkdir(exist_ok=True)

    def receive(item):
        (scene, light), entry = item
        filename = f"{scene}_{light}.jpg"
        dst = out / filename
        previous = cache / filename
        existing = next((path for path in (dst, previous) if path.is_file()), None)
        photo = existing.read_bytes() if existing else fetch_member(entry, len(index))
        fingerprint = validated(entry, photo)
        if existing != dst:
            dst.write_bytes(photo)
        return light, filename, fingerprint

    with ThreadPoolExecutor(max_workers=3) as pool:
        for light, filename, fingerprint in pool.map(receive, photos.items()):
            if light == LIGHTS[-1]:
                print(f"All 18 settings have been verified through {filename} ({fingerprint[:12]})", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("index_file", "cache", "out"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
