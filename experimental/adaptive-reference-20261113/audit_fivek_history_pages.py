"""Inspect bounded SQLite metadata pages within the public MIT–Adobe FiveK tar."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from struct import unpack
from tarfile import TarInfo
from urllib.request import Request, urlopen

URL = "https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar"
TAR_SIZE = 50_832_452_608
ETAG = '"bd5d9ac00-4a60220a9ffc0"'
DB_START = 48_979_963_392
DB_SIZE = 1_790_038_016
PAGE_BUDGET = 224 * 1024
FIELDS = {
    "Adobe_libraryImageDevelopHistoryStep": ("image", "dateCreated", "name", "text"),
    "Adobe_images": ("rootFile", "masterImage", "developSettingsIDCache"),
    "Adobe_imageDevelopBeforeSettings": ("developSettings", "beforeText"),
    "Adobe_imageDevelopSettings": ("image", "text", "historySettingsID"),
}
KNOWN_COLOR = ("Temperature", "Tint", "Exposure", "Contrast", "Saturation", "Vibrance",
               "HueAdjustmentRed", "SaturationAdjustmentBlue", "ToneCurve")
KNOWN_LOCAL = ("GradientBasedCorrections", "PaintBasedCorrections", "RetouchAreas", "CircularGradientBasedCorrections")


class RangeUnavailable(ValueError):
    """The requested SQLite page is beyond the allowed reads."""


def integer(data):
    return int.from_bytes(data, "big")


def varint(data, pos):
    value = 0
    for i in range(9):
        part = data[pos + i]
        if i == 8:
            return (value << 8) | part, pos + 9
        value = (value << 7) | (part & 127)
        if part < 128:
            return value, pos + i + 1


def record(data, encoding):
    stop, pos = varint(data, 0)
    tags = []
    while pos < stop:
        tag, pos = varint(data, pos)
        tags.append(tag)
    if pos != stop:
        raise ValueError("SQLite serial types exceed header")
    result = []
    for tag in tags:
        if tag in (0, 8, 9):
            value = None if tag == 0 else tag - 8
        elif 1 <= tag <= 6:
            size = (1, 2, 3, 4, 6, 8)[tag - 1]
            value = int.from_bytes(data[pos:pos + size], "big", signed=True)
            pos += size
        elif tag == 7:
            value, = unpack(">d", data[pos:pos + 8])
            pos += 8
        elif tag >= 12:
            size = (tag - 12) // 2
            value = data[pos:pos + size]
            if tag % 2:
                value = value.decode(encoding)
            pos += size
        else:
            raise ValueError("Unknown SQLite record serial type")
        result.append(value)
    if pos != len(data):
        raise ValueError("SQLite record length mismatch")
    return result


class Catalog:
    def __init__(self, ranges=None, offline=False, new_limit=PAGE_BUDGET):
        self.read_bytes = 0
        self.network_bytes = 0
        self.cache = {}
        self.ranges = ranges
        self.offline = offline
        self.new_limit = new_limit
        head = self.fetch(DB_START - 512, 512)
        item = TarInfo.frombuf(head, "utf-8", "strict")
        if not item.isfile() or Path(item.name).name != "fivek.lrcat" or item.size != DB_SIZE:
            raise ValueError("Archive member differs")
        first = self.fetch(DB_START, 4096)
        self.cache[1] = first
        self.head_sha256 = sha256(head).hexdigest()
        if not first.startswith(b"SQLite format 3\0") or integer(first[16:18]) != 4096:
            raise ValueError("Unexpected SQLite page size")
        self.usable = len(first) - first[20]
        self.encoding = {1: "utf-8", 2: "utf-16-le", 3: "utf-16-be"}[integer(first[56:60])]

    def fetch(self, offset, size):
        if self.read_bytes + size > PAGE_BUDGET:
            raise ValueError("Remote page budget exceeded")
        end = offset + size - 1
        file = self.ranges / f"{offset}-{size}.bin" if self.ranges is not None else None
        if file is not None and file.exists():
            payload = file.read_bytes()
        else:
            if self.offline:
                raise RangeUnavailable("Cached range unavailable in offline mode")
            if self.network_bytes + size > self.new_limit:
                raise RangeUnavailable("New remote range budget exceeded")
            headers = {"Range": f"bytes={offset}-{end}", "If-Range": ETAG, "Accept-Encoding": "identity"}
            with urlopen(Request(URL, headers=headers), timeout=15) as r:
                if (
                    r.url != URL or r.status != 206 or r.headers["ETag"] != ETAG
                    or r.headers["Content-Range"] != f"bytes {offset}-{end}/{TAR_SIZE}"
                ):
                    raise ValueError("Unexpected range response")
                payload = r.read(size + 1)
            self.network_bytes += size
            if file is not None and len(payload) == size:
                file.write_bytes(payload)
        if len(payload) != size:
            raise ValueError("Unexpected range length")
        self.read_bytes += size
        return payload

    def page(self, number):
        if not 1 <= number <= DB_SIZE // 4096:
            raise ValueError("SQLite page outside the tar member")
        if number not in self.cache:
            self.cache[number] = self.fetch(DB_START + (number - 1) * 4096, 4096)
        return self.cache[number]

    def table(self, number, reverse=False, only_rowid=None):
        data = self.page(number)
        offset = 100 if number == 1 else 0
        tag = data[offset]
        if tag not in (5, 13):
            raise ValueError("Expected a SQLite rowid table page")
        count = integer(data[offset + 3:offset + 5])
        cell_base = offset + (12 if tag == 5 else 8)
        cells = [integer(data[i:i + 2]) for i in range(cell_base, cell_base + 2 * count, 2)]
        if tag == 5:
            children = [integer(data[cell:cell + 4]) for cell in cells]
            children.append(integer(data[offset + 8:offset + 12]))
            if only_rowid is not None:
                from bisect import bisect_left

                upper_bounds = [varint(data, cell + 4)[0] for cell in cells]
                children = [children[bisect_left(upper_bounds, only_rowid)]]
            for child in reversed(children) if reverse else children:
                yield from self.table(child, reverse, only_rowid)
        else:
            for cell in reversed(cells) if reverse else cells:
                size, ptr = varint(data, cell)
                rowid, ptr = varint(data, ptr)
                if only_rowid is not None and rowid != only_rowid:
                    continue
                if size > 160 * 1024:
                    raise ValueError("Record exceeds local inspection budget")
                yield rowid, record(self.payload(data, ptr, size), self.encoding)

    def payload(self, data, pos, size):
        minimum = ((self.usable - 12) * 32 // 255) - 23
        local = size
        if size > self.usable - 35:
            candidate = minimum + (size - minimum) % (self.usable - 4)
            local = candidate if candidate <= self.usable - 35 else minimum
        result = bytearray(data[pos:pos + local])
        following = integer(data[pos + local:pos + local + 4]) if size > local else 0
        while len(result) < size:
            part = self.page(following)
            result += part[4:4 + min(self.usable - 4, size - len(result))]
            following = integer(part[:4])
        return result


def field_summary(value):
    if isinstance(value, str):
        return {"type": "text", "length": len(value)}
    if isinstance(value, bytes):
        return {"type": "blob", "length": len(value)}
    return {"type": type(value).__name__}


def snapshot_structure(value):
    if not isinstance(value, str) or not value.strip():
        return {"format": "empty"}
    begin = value.lstrip()
    if re.match(r"(?:s\s*=\s*)?\{", begin):
        kind = "table_text"
    elif begin.startswith("<"):
        kind = "xml_text"
    else:
        kind = "other_text"
    return {
        "format": kind,
        "color_fields_present": [key for key in KNOWN_COLOR if re.search(rf"\b{key}\b", value)],
        "local_fields_present": [key for key in KNOWN_LOCAL if re.search(rf"\b{key}\b", value)],
    }


TOKEN = re.compile(r'\s*(?:"(?:[^"\\]|\\.)*"|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
                   r'|[A-Za-z_]\w*|[{},=])')
IDENT = re.compile(r"[A-Za-z_]\w*\Z")


def snapshot(value):
    """Parse the observed literal-table subset without executing catalog text."""
    from decimal import Decimal

    tokens = []
    data = value.strip()
    pos = 0
    while pos < len(data):
        token = TOKEN.match(data, pos)
        if token is None:
            raise ValueError("Unsupported snapshot syntax")
        tokens.append(token[0].strip())
        pos = token.end()

    def parse(index, depth):
        if depth > 12 or index >= len(tokens):
            raise ValueError("Incomplete or deeply nested snapshot")
        start = tokens[index]
        if start != "{":
            if start.startswith('"'):
                return ("string", json.loads(start)), index + 1
            if start in ("true", "false", "nil"):
                return ("symbol", start), index + 1
            if start[0] not in "0123456789+-.":
                raise ValueError("Unsupported snapshot value")
            return ("number", Decimal(start)), index + 1
        index += 1
        keys, items = {}, []
        while index < len(tokens) and tokens[index] != "}":
            if index + 1 < len(tokens) and IDENT.fullmatch(tokens[index]) and tokens[index + 1] == "=":
                key = tokens[index]
                if key in keys:
                    raise ValueError("Duplicate snapshot key")
                keys[key], index = parse(index + 2, depth + 1)
            else:
                item, index = parse(index, depth + 1)
                items.append(item)
            if index < len(tokens) and tokens[index] == ",":
                index += 1
            elif index >= len(tokens) or tokens[index] != "}":
                raise ValueError("Missing snapshot separator")
        if index >= len(tokens) or (keys and items):
            raise ValueError("Unterminated or mixed snapshot table")
        return (keys if keys else tuple(items)), index + 1

    if tokens[:2] != ["s", "="]:
        raise ValueError("Unexpected snapshot root")
    parsed, end = parse(2, 0)
    if end != len(tokens) or not isinstance(parsed, dict):
        raise ValueError("Snapshot root is not a complete named table")
    return parsed


def snapshot_changes(image_groups):
    from itertools import pairwise

    differences = []
    parsed_rows = 0
    missing = object()
    for group in image_groups.values():
        ordered = sorted(group, key=lambda row: row["dateCreated"])
        parsed = []
        for row in ordered:
            try:
                state = snapshot(row["text"])
            except (ValueError, AttributeError):
                state = None
            parsed.append((row["dateCreated"], state))
            parsed_rows += state is not None
        for (before_time, before), (after_time, after) in pairwise(parsed):
            if before_time == after_time or before is None or after is None:
                continue
            changed = {k for k in before.keys() | after.keys() if before.get(k, missing) != after.get(k, missing)}
            differences.append({
                "changed_top_level_fields": len(changed),
                "known_color_fields_changed": sorted(changed.intersection(KNOWN_COLOR)),
                "local_fields_changed": sorted(changed.intersection(KNOWN_LOCAL)),
                "single_known_color_field": len(changed) == 1 and changed.issubset(KNOWN_COLOR),
            })
    return {"parsed_rows": parsed_rows, "ordered_pairs": differences}


def inspect(output, ranges=None, offline=False, tail_sample=False):
    from itertools import islice
    import sqlite3

    if ranges is not None and not offline:
        ranges.mkdir(exist_ok=True)
    db = Catalog(ranges, offline, 40 * 1024 if tail_sample else PAGE_BUDGET)
    schema = {name: (root, sql) for _, (kind, name, _, root, sql) in db.table(1) if kind == "table"}

    def columns_for(name):
        _, sql = schema[name]
        if not re.match(r"(?is)^CREATE\s+TABLE\s+", sql):
            raise ValueError("Expected an ordinary table definition")
        with sqlite3.connect(":memory:") as local:
            local.execute("PRAGMA trusted_schema = OFF")
            local.execute(sql)
            return [row[1] for row in local.execute(f'PRAGMA table_info("{name}")')]

    summary = {}
    for name in FIELDS:
        root, _ = schema[name]
        columns = columns_for(name)
        samples = [dict(zip(columns, row)) for _, row in islice(db.table(root), 10)]
        summary[name] = {"root_page": root, "columns": columns,
                         "sample_rows": [{col: field_summary(row[col]) for col in FIELDS[name]} for row in samples]}
        text_fields = [col for col in FIELDS[name] if col in ("text", "beforeText")]
        for field in text_fields:
            summary[name][f"{field}_structure"] = [snapshot_structure(row[field]) for row in samples]
        if name == "Adobe_libraryImageDevelopHistoryStep":
            image_groups = {}
            for row in samples:
                image_groups.setdefault(row["image"], []).append(row)
            summary[name]["sample_histories"] = sorted(len(rows) for rows in image_groups.values())
            summary[name]["repeated_same_image_states"] = sum(
                len({row["text"] for row in rows if row["text"]}) >= 2 for rows in image_groups.values()
            )
            summary[name]["snapshot_field_comparison"] = snapshot_changes(image_groups)
            if tail_sample:
                tail = {}
                for _, values in islice(db.table(root, reverse=True), 10):
                    row = dict(zip(columns, values))
                    tail.setdefault(row["image"], []).append(row)
                results = summary[name]
                results["tail_sample_histories"] = sorted(len(rows) for rows in tail.values())
                results["tail_snapshot_field_comparison"] = snapshot_changes(tail)
                photo_root = schema["Adobe_images"][0]
                file_column = columns_for("Adobe_images").index("rootFile")

                def files(groups):
                    keys = set()
                    unresolved = 0
                    for key in groups:
                        try:
                            matches = list(db.table(photo_root, only_rowid=key))
                        except RangeUnavailable:
                            unresolved += 1
                            continue
                        if len(matches) != 1 or matches[0][1][file_column] is None:
                            raise ValueError("History has no unique photo and raw-file reference")
                        keys.add(matches[0][1][file_column])
                    return keys, unresolved

                first_files, first_unresolved = files(image_groups)
                last_files, last_unresolved = files(tail)
                results["file_linkage"] = {
                    "first_files": len(first_files), "last_files": len(last_files),
                    "first_unresolved": first_unresolved, "last_unresolved": last_unresolved,
                    "confirmed_shared_files": len(first_files & last_files),
                }
    report = {
        "source": URL, "tar_bytes": TAR_SIZE, "etag": ETAG, "member_header_sha256": db.head_sha256,
        "catalog_data_offset": DB_START, "catalog_bytes": DB_SIZE, "sqlite_encoding": db.encoding,
        "remote_bytes": db.read_bytes, "table_count": len(schema), "tables": summary,
        "sqlite_pages_sha256": {str(n): sha256(p).hexdigest() for n, p in sorted(db.cache.items())},
    }
    with output.open("xb") as handle:
        handle.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ranges", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tail-sample", action="store_true")
    args = parser.parse_args()
    inspect(args.output, args.ranges, args.offline, args.tail_sample)
