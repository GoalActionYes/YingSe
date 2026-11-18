"""Inspect only top-level MOV headers and moov metadata for four PST50 clips."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
from urllib.request import ProxyHandler, Request, build_opener

REVISION = "6ed86e34c37c558a0b2401d52f06603d619472e1"
PATHS = ("video/in14.mov", "video/in16.mov", "video/in40.mov", "video/in40b.mov")
LIST_SHA = "6d7b43361672af29da670a5681a19f009b44d9911535adaa629cd319ac139339"
LIMIT_PER_FILE = 512 * 1024
EXPANDED_LIMIT_PER_FILE = 40 * 1024
CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"meta", b"ilst"}


def box_header(data, offset, limit):
    size, tag = struct.unpack_from(">I4s", data, offset)
    header = 8
    if size == 1:
        size = struct.unpack_from(">Q", data, offset + 8)[0]
        header = 16
    if size == 0:
        size = limit - offset
    if not header <= size <= limit - offset:
        raise ValueError(f"Invalid {tag!r} box length {size}")
    return size, tag, header


def scan(data, start, end):
    offset = start
    while offset + 8 <= end:
        length, tag, header = box_header(data, offset, end)
        payload = offset + header
        yield tag, data[payload:offset + length]
        if tag in CONTAINERS:
            if tag == b"meta" and data[payload + 4:payload + 8] not in (b"hdlr", b"keys", b"ilst"):
                payload += 4
            yield from scan(data, payload, offset + length)
        offset += length
    if offset != end:
        raise ValueError("Unexpected trailing bytes in MOV container")


def qt_date(seconds):
    if not seconds:
        return None
    return (datetime(1904, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat()


def inspect(metadata):
    times, handlers, keys = [], [], []
    records = list(scan(metadata, 0, len(metadata)))
    for tag, content in records:
        if tag in (b"mvhd", b"tkhd", b"mdhd"):
            version = content[0]
            pattern = ">II" if version == 0 else ">QQ"
            created, modified = struct.unpack_from(pattern, content, 4)
            times.append({"atom": tag.decode("ascii"), "created": qt_date(created), "modified": qt_date(modified)})
        if tag == b"hdlr":
            handlers.append(content[8:12].decode("ascii", errors="replace"))
        if tag == b"keys":
            count = struct.unpack_from(">I", content, 4)[0]
            offset = 8
            for _ in range(count):
                length = struct.unpack_from(">I", content, offset)[0]
                if not 8 <= length <= len(content) - offset:
                    raise ValueError("Bad QuickTime metadata record")
                keys.append(content[offset + 8:offset + length].decode("utf-8", errors="replace"))
                offset += length
    signature = re.compile(rb"(?i)(quicktime|timecode|orig|reel|camera|shot|creation|davinci|adobe|prores|source|apple|scene|premiere)")
    texts = sorted({block.decode("ascii").strip() for block in re.findall(rb"[ -~]{6,160}", metadata)
                    if signature.search(block)})
    return {"times": times, "handlers": handlers, "metadata_keys": keys, "metadata_strings": texts}


def timecode_track(metadata):
    # A QuickTime track may also contain a data handler. Use the media handler.
    track_records = [list(scan(track, 0, len(track))) for tag, track in scan(metadata, 0, len(metadata))
                     if tag == b"trak"]
    tables = [dict(track) for track in track_records]
    matches = [i for i, track in enumerate(track_records)
               if any(tag == b"hdlr" and data[8:12] == b"tmcd" for tag, data in track)]
    return tables, matches


def timecode_description(description):
    length, kind, _ = box_header(description, 8, len(description))
    if kind != b"tmcd" or 8 + length != len(description) or length < 34:
        raise ValueError("Unexpected sample description")
    flags, timescale, framesize, nominal = struct.unpack_from(">IIIB", description, 28)
    source_labels = []
    # Two Resolve exports have two zero bytes in place of extensions.
    children = () if description[42:] == b"\0\0" else scan(description, 42, len(description))
    for tag, payload in children:
        if tag == b"name":
            text_length = struct.unpack_from(">H", payload)[0]
            if text_length > len(payload) - 4:
                raise ValueError("Invalid source label length")
            source_labels.append(payload[4:4 + text_length].decode("utf-8"))
    return {"flags": flags, "timescale": timescale, "frame_duration": framesize,
            "nominal_fps": nominal, "source_labels": source_labels}


def timecode_sample(tables, index, top_boxes):
    table = tables[index]
    if struct.unpack(">III", table[b"stsz"]) != (0, 4, 1):
        raise ValueError("Timecode track does not describe one four-byte sample")
    if struct.unpack(">IIIII", table[b"stsc"]) != (0, 1, 1, 1, 1):
        raise ValueError("Unexpected timecode sample-to-chunk entry")
    _, count, offset = struct.unpack(">III", table[b"stco"])
    if count != 1:
        raise ValueError("Unexpected timecode offset table")
    mdat = next(box for box in top_boxes if box["tag"] == "mdat")
    if not mdat["offset"] + mdat["header"] <= offset <= mdat["offset"] + mdat["size"] - 4:
        raise ValueError("Timecode offset outside media storage")
    for i, other in enumerate(tables):
        if i != index and b"stco" in other:
            points = tuple(v[0] for v in struct.iter_unpack(">I", other[b"stco"][8:]))
            if any(point < offset + 4 for point in points):
                raise ValueError("Timecode sample overlaps another track's chunk start")

    return offset


def range_reader(row, proxy, budget):
    path, total = row["path"], row["size"]
    name = Path(path).stem
    opener = build_opener(ProxyHandler({"https": proxy} if proxy else {}))
    log = []

    def fetch(start, length):
        if length + sum(item["length"] for item in log) > budget:
            raise ValueError(f"Metadata byte budget exceeded for {name}")
        end = start + length - 1
        url = f"https://huggingface.co/datasets/zrgong/PST50/resolve/{REVISION}/{path}?download=true"
        req = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        with opener.open(req, timeout=18) as response:
            wanted = f"bytes {start}-{end}/{total}"
            if response.status != 206 or response.headers.get("Content-Range") != wanted:
                raise ValueError(f"Range rejected for {name} at {start}: {response.status}")
            data = response.read(length + 1)
            etag = response.headers.get("ETag")
        if len(data) != length:
            raise ValueError(f"Range size mismatch for {name} at {start}")
        if etag is None or etag.strip('"') != row["xetHash"]:
            raise ValueError(f"Unexpected Xet object identity for {name}")
        log.append({"offset": start, "length": length, "etag": etag, "sha256": sha256(data).hexdigest()})
        return data

    return fetch, log


def fetch_one(row, output, proxy, expanded, *, budget_override=None):
    path, total = row["path"], row["size"]
    expected = row["lfs"]["oid"]
    name = Path(path).stem
    budget = EXPANDED_LIMIT_PER_FILE if expanded else LIMIT_PER_FILE
    fetch, log = range_reader(row, proxy, budget if budget_override is None else budget_override)
    position = 0
    boxes = []
    metadata = None
    while position < total:
        if len(boxes) >= 30:
            raise ValueError("Too many top-level boxes")
        top = fetch(position, 8)
        size, tag = struct.unpack(">I4s", top)
        if size == 1:
            top += fetch(position + 8, 8)
        size, tag, header = box_header(top, 0, total - position)
        boxes.append({"tag": tag.decode("ascii", errors="replace"), "offset": position,
                      "size": size, "header": header})
        if tag == b"moov":
            metadata = top + fetch(position + header, size - header)
            (output / f"{name}-moov.bin").write_bytes(metadata)
        position += size
    if metadata is None:
        raise ValueError("No moov box found")
    tables, matches = timecode_track(metadata)
    names = [timecode_description(tables[index][b"stsd"]) for index in matches]
    if not expanded:
        if len(matches) != 1:
            raise ValueError("The timecode track is missing or ambiguous")
        offset = timecode_sample(tables, matches[0], boxes)
        code = fetch(offset, 4)
        fmt = ">I" if names[0]["flags"] & 8 else ">i"
        names[0].update(offset=offset, bytes=4, start_frame=struct.unpack(fmt, code)[0])
    return {"lfs_sha256": expected, "xet_hash": row["xetHash"], "file_size": total, "boxes": boxes,
            "ranges": log, "moov_sha256": sha256(metadata).hexdigest(), "timecodes": names,
            **inspect(metadata)}


def run(inventory, out, proxy, expanded):
    data = inventory.read_bytes()
    if sha256(data).hexdigest() != LIST_SHA:
        raise ValueError("Unexpected published tree receipt")
    entries = {entry["path"]: entry for entry in json.loads(data)}
    out.mkdir()
    paths = [path for path in entries if re.fullmatch(r"video/in\d+b?\.(mov|mp4|mxf)", path, re.I)] if expanded else PATHS
    if expanded and len(paths) != 41:
        raise ValueError("Expected 41 source clips")
    workers = 10 if expanded else 4
    records, errors = {}, {}
    unsupported = [path for path in paths if path.lower().endswith(".mxf")]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {path: pool.submit(fetch_one, entries[path], out, proxy, expanded)
                   for path in paths if path not in unsupported}
        for path, future in futures.items():
            try:
                records[path] = future.result()
            except Exception as error:
                errors[path] = {"type": type(error).__name__, "message": str(error)}
    source_names = defaultdict(list)
    for path, row in records.items():
        for label in sorted({label for tc in row["timecodes"] for label in tc["source_labels"] if label}):
            source_names[label].append(path)
    number = lambda path: int(re.search(r"\d+", path).group())
    cross = {label: paths for label, paths in source_names.items()
             if min(map(number, paths)) <= 15 < max(map(number, paths))}
    result = {"revision": REVISION, "expanded": expanded, "video": records, "errors": errors,
              "unsupported": unsupported,
              "exact_cross_split_source_labels": cross,
              "downloaded_bytes_success": sum(item["length"] for row in records.values() for item in row["ranges"])}
    (out/"results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--proxy")
    parser.add_argument("--all-video", action="store_true")
    args = parser.parse_args()
    run(args.inventory, args.output, args.proxy, args.all_video)
