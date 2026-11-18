"""Trace the two fixed FiveK history edits to original and published outputs."""
import argparse
from collections import Counter
from decimal import Decimal
from hashlib import sha256
from itertools import combinations, islice, pairwise
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from tarfile import TarInfo
from urllib.parse import quote
from urllib.request import Request, urlopen

from audit_fivek_history_pages import Catalog, DB_SIZE, DB_START, ETAG, different_fields, snapshot
from audit_fivek_indexed_histories import FILE_INDEX, HISTORY, INDEX, MAX_STEPS, PICTURES, first_keys

FIVEK = "https://data.csail.mit.edu/graphics/fivek/"
SETTINGS = "Adobe_imageDevelopSettings"
FILES = "AgLibraryFile"
MAX_NEW = 496 * 1024


def columns(schema, table):
    sql = schema[table][1]
    if not re.match(r"(?is)^CREATE\s+TABLE\s+", sql):
        raise ValueError("Unexpected table definition")
    with sqlite3.connect(":memory:") as conn:
        conn.execute("PRAGMA trusted_schema = OFF")
        conn.execute(sql)
        return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def lookup(db, root, key, names):
    values = next(db.table(root, only_rowid=key), (None, None))[1]
    if values is None:
        raise ValueError("Catalog link does not resolve")
    return dict(zip(names, values))


def materialize(cached, db, schema, output):
    history_fields, photo_fields = (columns(schema, table) for table in (HISTORY, PICTURES))
    file_index, history_index = (schema[table][0] for table in (FILE_INDEX, INDEX))
    history_root, photo_root = (schema[table][0] for table in (HISTORY, PICTURES))
    sources = first_keys(cached.index(file_index))
    basis = Path(__file__).resolve().parents[2] / "docs/research/2026-11-19-fivek-indexed-histories-evidence.json"
    frozen = json.loads(basis.read_bytes())["raw_file_order"]["sampled_photos"]
    matches = []
    for number, raw in enumerate(sources, 1):
        for copy, (_, image) in enumerate(islice(cached.index(file_index, image=raw), 6), 1):
            if frozen[f"raw-{number}-copy-{copy}"]["status"] != "read":
                continue
            pic = lookup(cached, photo_root, image, photo_fields)
            if pic["rootFile"] != raw:
                raise ValueError("Photo points to an unexpected source")
            indices = list(islice(cached.index(history_index, image=image), MAX_STEPS + 1))
            if len(indices) > MAX_STEPS:
                raise ValueError("Frozen history exceeds the step limit")
            rows = [lookup(cached, history_root, rowid, history_fields) for _, _, rowid in indices]
            if any((state["image"], state["dateCreated"]) != (photo_key, time)
                   for state, (photo_key, time, _) in zip(rows, indices, strict=True)):
                raise ValueError("History row and index do not agree")
            counted = Counter(state["dateCreated"] for state in rows)
            for before, after in pairwise(rows):
                if counted[before["dateCreated"]] != 1 or counted[after["dateCreated"]] != 1:
                    continue
                try:
                    a, b = snapshot(before["text"]), snapshot(after["text"])
                except (ValueError, AttributeError):
                    continue
                left, right = a.get("Contrast"), b.get("Contrast")
                if (left and right and left[0] == right[0] == "number"
                        and right[1] - left[1] == Decimal(13) and different_fields(a, b) == {"Contrast"}):
                    matches.append((raw, pic, before, after, rows[-1], left, right))
    pairs = [(a, b) for a, b in combinations(matches, 2) if a[0] != b[0] and a[-2:] == b[-2:]]
    if len(pairs) != 1:
        raise ValueError("Frozen cross-file operation was not uniquely recovered")

    descriptor = []
    private = []
    file_fields, setting_fields = (columns(schema, name) for name in (FILES, SETTINGS))
    for raw, pic, before, after, terminal, left, right in pairs[0]:
        source = lookup(db, schema[FILES][0], raw, file_fields)
        basename, ext = source["baseName"], source["extension"]
        if not re.fullmatch(r"a\d{4}-[\w.() -]+", basename, re.I) or ext.lower() != "dng":
            raise ValueError("Unexpected published raw file identifier")
        key = pic["developSettingsIDCache"]
        if key != int(key):
            raise ValueError("Invalid Develop settings reference")
        current = lookup(db, schema[SETTINGS][0], int(key), setting_fields)
        if current["image"] != after["image"]:
            raise ValueError("Current settings point to a different copy")
        states = {role: snapshot(value["text"]) for role, value in
                  (("before", before), ("after", after), ("terminal", terminal), ("current", current))}
        legal = [name for name in ("ProcessVersion", "Version") if name in states["before"]]
        descriptor.append({
            "contrast_before": str(left[1]), "contrast_after": str(right[1]),
            "fields_present": legal,
            "process_metadata": {name: str(states["before"][name][1]) for name in legal
                                 if isinstance(states["before"][name], tuple)},
            "current_equal_before": states["current"] == states["before"],
            "current_equal_after": states["current"] == states["after"],
            "terminal_equal_before": states["terminal"] == states["before"],
            "terminal_equal_after": states["terminal"] == states["after"],
            "current_equal_terminal": states["current"] == states["terminal"],
            "current_differing_fields_from_after": len(different_fields(states["current"], states["after"])),
        })
        private.append({
            "basename": basename, "extension": ext, "copy_name": pic["copyName"], "uuid": pic["id_global"],
            "image_id": after["image"],
            "before_digest": before["digest"], "after_digest": after["digest"],
            "current_digest": current["digest"], "before": before["text"], "after": after["text"],
        })
    (output / "local-replay.json").write_bytes((json.dumps(private, indent=2) + "\n").encode())
    return descriptor, private


def check_public_members(private):
    reports = []
    for case in private:
        basename = quote(case["basename"], safe="")
        urls = {"dng": f"{FIVEK}img/dng/{basename}.dng"}
        name = case["copy_name"] or ""
        match = re.fullmatch(r"\s*(?:expert\s*|retouch\s*)?([abcde])\s*", name, re.I)
        urls.update({f"expert_{letter}": f"{FIVEK}img/tiff16_{letter}/{basename}.tif" for letter in "abcde"})
        metadata = {}
        for role, url in urls.items():
            with urlopen(Request(url, method="HEAD"), timeout=15) as response:
                if response.status != 200 or response.url != url:
                    raise ValueError("Unexpected public item header")
                metadata[role] = {"bytes": int(response.headers["Content-Length"]),
                                  "content_type": response.headers.get_content_type()}
        raw_size = metadata["dng"]["bytes"]
        req = Request(urls["dng"], headers={"Range": "bytes=0-15", "Accept-Encoding": "identity"})
        with urlopen(req, timeout=15) as response:
            if (response.status != 206 or response.url != urls["dng"]
                    or response.headers["Content-Range"] != f"bytes 0-15/{raw_size}"):
                raise ValueError("Unexpected original header range")
            first = response.read(17)
        if len(first) != 16 or first[:4] not in (b"II*\x00", b"MM\x00*"):
            raise ValueError("Original does not have a TIFF/DNG container header")
        metadata["dng"]["valid_tiff_header"] = True
        reports.append({"final_expert_designation_matched": bool(match), "public_items": metadata})
    return reports


def inspect_preview(db, output, private):
    offset = DB_START + DB_SIZE
    types = Counter()
    database = None
    for _ in range(18):
        data = db.fetch(offset, 512)
        entry = TarInfo.frombuf(data, "utf-8", "strict")
        suffix = PurePosixPath(entry.name).suffix.lower()
        types["directory" if entry.isdir() else suffix] += 1
        if entry.isfile() and PurePosixPath(entry.name).name == "previews.db":
            if database is not None or entry.size > 480 * 1024:
                raise ValueError("Unexpected preview index member")
            database = (offset + 512, entry.size)
        offset += 512 + 512 * ((entry.size + 511) // 512)
    if database is None:
        raise ValueError("The fixed preview directory window does not include its index")
    path = output / "previews.db"
    payload = db.fetch(*database)
    path.write_bytes(payload)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as conn:
        conn.execute("PRAGMA trusted_schema = OFF")
        if conn.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("Archive preview index fails SQLite validation")

        def scalar(sql, args=()):
            return conn.execute(sql, args).fetchone()[0]

        counts = {
            key: scalar(f'SELECT COUNT(*) FROM "{key}"')
            for key in ("ImageCacheEntry", "Pyramid", "PyramidLevel")
        }
        matches = []
        for state in private:
            image_count = scalar("SELECT count(*) FROM ImageCacheEntry WHERE imageId=? OR lower(uuid)=lower(?)",
                                 (state["image_id"], state["uuid"]))
            rows = {}
            for stage in ("before", "after", "current"):
                digest = state[f"{stage}_digest"]
                rows[stage] = {
                    "pyramids_matching_digest": scalar("SELECT count(*) FROM Pyramid WHERE lower(digest)=lower(?)",
                                                       (digest,)),
                    "pyramids_matching_uuid_and_digest": scalar(
                        "SELECT count(*) FROM Pyramid WHERE lower(uuid)=lower(?) AND lower(digest)=lower(?)",
                        (state["uuid"], digest)),
                }
            matches.append({"image_cache_rows_by_id_or_uuid": image_count, "states": rows})
    return {"headers_read": sum(types.values()), "member_types": dict(sorted(types.items())),
            "index_bytes": len(payload), "index_sha256": sha256(payload).hexdigest(),
            "rows": counts, "selected_case_matches": matches}


def write_json(path, data):
    path.write_bytes((json.dumps(data, indent=2, allow_nan=False) + "\n").encode())


def run(source_cache, output):
    output.mkdir()
    db = Catalog(source_cache, True, read_limit=512 * 1024)
    schema = {name: (root, sql) for _, (_, name, _, root, sql) in db.table(1)}
    # Freeze candidate selection against pages already consumed in the indexed study.
    db_from_source = Catalog(source_cache, False, new_limit=MAX_NEW, read_limit=640 * 1024)
    facts, private = materialize(db, db_from_source, schema, output)
    release = check_public_members(private)
    previews = inspect_preview(db_from_source, output, private)
    write_json(output / "results.json", {
        "etag": ETAG, "version": "Contrast +13", "states": facts,
        "released_files": release, "previews": previews,
        "local_replay_sha256": sha256((output / "local-replay.json").read_bytes()).hexdigest(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.cache, args.output)
