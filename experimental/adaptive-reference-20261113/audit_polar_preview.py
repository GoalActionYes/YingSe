"""Read only the ZIP directories of two pinned POLAR preview samples."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from struct import unpack_from
from zipfile import ZipFile

import requests


REPO = "https://huggingface.co/datasets/Rex0191/POLAR"
REVISION = "77ec5422bc9003c202116575ca7f17a25dd85157"
ARCHIVES = {
    "028.zip": (553348947, "298f3deca0d3920f301647a03dd53c9796a80bcac6dc86f10dde46dc08b595ec"),
    "030.zip": (727887907, "624863a37dff05c554120f1334bff5f82fab5ac27e4a24e5b62c8fc508ef4f81"),
}
BUDGET = 4 * 1024**2
TAIL = 65557


def read_range(session, name, start, length):
    size, digest = ARCHIVES[name]
    end = start + length - 1
    with session.get(f"{REPO}/resolve/{REVISION}/{name}",
                     headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                     timeout=(10, 25), stream=True) as response:
        source = (response.history or [response])[0].headers
        if (source.get("X-Repo-Commit"), source.get("X-Linked-Etag", "").strip('"')) != (REVISION, digest):
            raise ValueError(f"Unexpected release identity for {name}")
        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
            raise ValueError(f"Unverified range for {name}: HTTP {response.status_code}")
        data = response.raw.read(length + 1)
    if len(data) != length:
        raise ValueError(f"Unexpected range length for {name}")
    return data


def read_catalog(session, name, budget):
    total, digest = ARCHIVES[name]
    tail = read_range(session, name, total - TAIL, TAIL)
    signature, disk, directory_disk, records_disk, records, size, offset, comment = unpack_from(
        "<4s4H2IH", tail, len(tail) - 22)
    if (signature != b"PK\x05\x06" or disk or directory_disk or records_disk != records or comment
            or offset + size != total - 22 or size + TAIL > budget):
        raise ValueError(f"Unexpected ZIP directory or budget for {name}")
    directory = read_range(session, name, offset, size) if size + 22 > TAIL else tail[-size - 22:-22]
    # A synthetic central-directory-only ZIP gives zipfile the actual records without reading member bodies.
    with ZipFile(BytesIO(directory + tail[-22:])) as archive:
        files = [row for row in archive.infolist() if not row.is_dir()]
        if len(archive.infolist()) != records or len({row.filename for row in files}) != len(files):
            raise ValueError(f"Ambiguous ZIP records for {name}")
        members = [{"path": row.filename, "bytes": row.file_size, "crc32": f"{row.CRC:08x}"}
                   for row in files]
    return {
        "archive_bytes": total, "published_sha256": digest, "members": len(members),
        "suffix_counts": dict(sorted(Counter(PurePosixPath(row["path"]).suffix for row in members).items())),
        "tail_sha256": sha256(tail).hexdigest(), "directory_sha256": sha256(directory).hexdigest(),
        "transferred_bytes": TAIL + (size if size + 22 > TAIL else 0),
    }, members


def run(out, proxy):
    summaries, inventory = {}, {}
    with requests.Session() as session:
        session.trust_env = False
        if proxy:
            session.proxies.update({"https": proxy})
        remaining = BUDGET
        for name in ARCHIVES:
            summaries[name], inventory[name] = read_catalog(session, name, remaining)
            remaining -= summaries[name]["transferred_bytes"]
    out.mkdir()
    for filename, record in (("summary.json", {"revision": REVISION, "archives": summaries}),
                             ("inventory.json", inventory)):
        (out / filename).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--proxy")
    run(**vars(parser.parse_args()))
