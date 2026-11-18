"""Complete the metadata-only PST50 audit for the two omitted containers."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct

from audit_pst50_video_family_metadata import LIST_SHA, REVISION, fetch_one, range_reader

HEADER = bytes.fromhex("060e2b34020501010d0102010102")
PRIMER = bytes.fromhex("060e2b34020501010d01020101050100")
FILL = bytes.fromhex("060e2b34010101020301021001000000")
LOCAL_SET = bytes.fromhex("060e2b34025301010d01010101")
PRIOR_SHA = "3d374931c14aa51e1d186d004d849601b448e2be23e32e4db2698ef9cada05e6"
LIMIT = 512 * 1024


def ber(first, rest):
    if first < 128:
        return first, 1
    count = first & 127
    if not 1 <= count <= 8 or len(rest) < count:
        raise ValueError("Unsupported BER length")
    return int.from_bytes(rest[:count], "big"), 1 + count


def remote_klv(fetch, offset):
    header = fetch(offset, 17)
    count = header[-1] & 127 if header[-1] >= 128 else 0
    if count > 8:
        raise ValueError("Unsupported remote BER length")
    tail = fetch(offset + 17, count) if count else b""
    length, encoded = ber(header[-1], tail)
    return header[:16], offset + 16 + encoded, length


def is_fill(key):
    return key[:7] == FILL[:7] and key[8:] == FILL[8:]


def iter_klv(data):
    pos = 0
    while pos < len(data):
        if pos + 17 > len(data):
            raise ValueError("Truncated KLV header")
        length, field = ber(data[pos + 16], data[pos + 17:pos + 25])
        offset = pos + 16 + field
        end = offset + length
        if end > len(data):
            raise ValueError("KLV extends beyond declared header")
        yield data[pos:pos + 16], data[offset:end]
        pos = end


def local_set(payload):
    tags = {}
    pos = 0
    while pos < len(payload):
        tag, length = struct.unpack_from(">HH", payload, pos)
        pos += 4
        if pos + length > len(payload) or tag in tags:
            raise ValueError("Invalid MXF package local set")
        tags[tag] = payload[pos:pos + length]
        pos += length
    return tags


def inspect_mxf(data):
    packages = []
    count = 0
    for key, payload in iter_klv(data):
        count += 1
        if key != PRIMER and not is_fill(key) and not key.startswith(LOCAL_SET):
            raise ValueError(f"Unexpected key inside header metadata: {key.hex()}")
        if not key.startswith(LOCAL_SET) or key[14] not in (0x36, 0x37):
            continue
        tags = local_set(payload)
        name_bytes = tags.get(0x4402, b"")
        title = name_bytes.decode("utf-16-be").rstrip("\0")
        name = title.replace("\\", "/").rsplit("/", 1)[-1]
        umid = tags.get(0x4401, b"")
        if len(umid) != 32:
            raise ValueError("Missing package UMID")
        packages.append({"type": "source" if key[14] == 0x37 else "material", "name": name,
                         "umid_sha256": sha256(umid).hexdigest()})
    if not packages:
        raise ValueError("MXF header has no identifiable package")
    return packages, count


def fetch_mxf(row, dest, proxy):
    fetch, log = range_reader(row, proxy, LIMIT)
    key, start, length = remote_klv(fetch, 0)
    if not key.startswith(HEADER) or not 88 <= length <= 4096:
        raise ValueError("Expected exact header partition at file start")
    pack = fetch(start, length)
    declared = int.from_bytes(pack[32:40], "big")
    pos = start + length
    key, offset, length = remote_klv(fetch, pos)
    while is_fill(key):
        if offset + length > LIMIT:
            raise ValueError("Padding exceeded the permitted header extent")
        pos = offset + length
        key, offset, length = remote_klv(fetch, pos)
    if key != PRIMER or not offset - pos + length <= declared <= LIMIT - pos:
        raise ValueError("Unexpected MXF header metadata bounds")
    data = fetch(pos, declared)
    packages, count = inspect_mxf(data)
    (dest / "in46-header-metadata.bin").write_bytes(data)
    return {"packages": packages, "klv_entries": count, "ranges": log, "header_metadata_offset": pos,
            "header_metadata_sha256": sha256(data).hexdigest(), "header_metadata_bytes": declared}


def run(inventory, prior, output, proxy):
    raw = inventory.read_bytes()
    if sha256(raw).hexdigest() != LIST_SHA:
        raise ValueError("Fixed dataset identity changed")
    entries = {row["path"]: row for row in json.loads(raw)}
    prior_bytes = prior.read_bytes()
    if sha256(prior_bytes).hexdigest() != PRIOR_SHA:
        raise ValueError("Prior container audit changed")
    previous = json.loads(prior_bytes)
    if previous["revision"] != REVISION or len(previous["video"]) != 39:
        raise ValueError("Expected 39 previously audited container results")
    output.mkdir()
    mov = fetch_one(entries["video/in42.mov"], output, proxy, True, budget_override=LIMIT)
    mxf = fetch_mxf(entries["video/in46.MXF"], output, proxy)
    known = {label: path for path, row in previous["video"].items() for tc in row["timecodes"]
             for label in tc["source_labels"] if label}
    new_labels = [label for tc in mov["timecodes"] for label in tc["source_labels"] if label]
    new_labels.extend(row["name"] for row in mxf["packages"] if row["type"] == "source" and row["name"])
    report = {"revision": REVISION, "prior_sha256": PRIOR_SHA,
              "parsed_container_count": 41, "in42": mov, "in46": mxf,
              "new_exact_matches_to_prior": {label: known[label] for label in new_labels if label in known},
              "downloaded_bytes": sum(item["length"] for row in (mov, mxf) for item in row["ranges"])}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("inventory", "prior", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--proxy")
    run(**vars(parser.parse_args()))
