"""Read fixed photo histories via their numeric SQLite index."""
import argparse
from collections import Counter
from datetime import date
from hashlib import sha256
from itertools import combinations, islice
import json
from pathlib import Path
import sqlite3

from audit_fivek_history_pages import (
    Catalog, DB_START, ETAG, KNOWN_COLOR, KNOWN_LOCAL, RangeUnavailable,
    adjacent_states, different_fields,
)

HISTORY = "Adobe_libraryImageDevelopHistoryStep"
PICTURES = "Adobe_images"
INDEX = "index_Adobe_libraryImageDevelopHistoryStep_imageDateCreated"
FILE_INDEX = "index_Adobe_images_rootFile"
MAX_STEPS = 64


def first_keys(rows):
    keys = {}
    for image, *_ in islice(rows, 4 * MAX_STEPS + 1):
        keys[image] = None
        if len(keys) == 4:
            return list(keys)
    raise ValueError("Not enough distinct history keys in the fixed index window")


def read_case(db, image, index, history, history_fields, photo, file_col, capture_col):
    keys = list(islice(db.index(index, image=image), MAX_STEPS + 1))
    if len(keys) > MAX_STEPS:
        return None, {"status": "step_limit", "histories_at_least": len(keys)}, []
    _, pic = next(db.table(photo, only_rowid=image), (None, None))
    if pic is None or not isinstance(pic[file_col], int):
        raise ValueError("Missing indexed history or raw-file reference")
    time = pic[capture_col]
    try:
        day = date.fromisoformat(time[:10]) if isinstance(time, str) else None
    except ValueError:
        day = None
    origin = (pic[file_col], day)
    if not keys:
        return origin, {"status": "no_history", "history_rows": 0}, []
    rows = []
    for key, timestamp, rowid in keys:
        match = next(db.table(history, only_rowid=rowid), (None, None))[1]
        if match is None:
            raise ValueError("Index points at a missing history")
        values = dict(zip(history_fields, match))
        if (values["image"], values["dateCreated"]) != (key, timestamp):
            raise ValueError("Index key disagrees with its history row")
        rows.append(values)
    parsed, pairs = adjacent_states(rows)
    colors, local_only, other = Counter(), 0, 0
    operations = []
    for before, after in pairs:
        changed = different_fields(before, after)
        if len(changed) == 1 and changed.issubset(KNOWN_COLOR):
            field = next(iter(changed))
            colors[field] += 1
            start, stop = before.get(field), after.get(field)
            if start is not None and stop is not None and start[0] == stop[0] == "number":
                operations.append((field, stop[1] - start[1], start[1], stop[1],
                                   {k: v for k, v in before.items() if k != field}))
        elif changed and changed.issubset(KNOWN_LOCAL):
            local_only += 1
        else:
            other += 1
    return origin, {"status": "read", "history_rows": len(rows), "parsed_rows": parsed,
                      "comparable_pairs": len(pairs), "only_color": dict(sorted(colors.items())),
                      "only_known_local_pairs": local_only, "other_pairs": other}, operations


def run(output, ranges, offline, by_file=False):
    if not offline:
        ranges.mkdir(exist_ok=True)
    db = Catalog(ranges, offline, new_limit=(224 if by_file else 384) * 1024, read_limit=512 * 1024)
    schema = {name: (root, sql) for _, (_, name, _, root, sql) in db.table(1)}
    index, declaration = schema[INDEX]
    if "( image, dateCreated )" not in declaration:
        raise ValueError("Unexpected history index declaration")
    history, photo = schema[HISTORY][0], schema[PICTURES][0]
    with sqlite3.connect(":memory:") as local:
        local.execute("PRAGMA trusted_schema = OFF")
        for name in (HISTORY, PICTURES):
            local.execute(schema[name][1])
        fields = [row[1] for row in local.execute(f'PRAGMA table_info("{HISTORY}")')]
        photo_fields = [row[1] for row in local.execute(f'PRAGMA table_info("{PICTURES}")')]
        file_col, capture_col = (photo_fields.index(field) for field in ("rootFile", "captureTime"))

    if by_file:
        file_index, file_decl = schema[FILE_INDEX]
        if "( rootFile )" not in file_decl:
            raise ValueError("Unexpected raw-file index declaration")
        sample = [(f"raw-{group_number}-copy-{copy_number}", image, raw_file)
                  for group_number, raw_file in enumerate(first_keys(db.index(file_index)), 1)
                  for copy_number, (key, image) in enumerate(islice(db.index(file_index, image=raw_file), 6), 1)
                  if key == raw_file]
    else:
        photo_keys = [*first_keys(db.index(index)), *first_keys(db.index(index, reverse=True))]
        if len(set(photo_keys)) != 8:
            raise ValueError("Index boundary samples overlap")
        sample = [(f"{'first' if n <= 4 else 'last'}-{(n - 1) % 4 + 1}", image, None)
                  for n, image in enumerate(photo_keys, 1)]
    rows, per_file, transitions = {}, {}, []
    for label, image, expected_file in sample:
        try:
            origin, report, operations = read_case(db, image, index, history, fields, photo, file_col, capture_col)
        except RangeUnavailable:
            rows[label] = {"status": "page_unavailable"}
            continue
        if origin is not None:
            raw_file, capture_day = origin
            if expected_file is not None and raw_file != expected_file:
                raise ValueError("Raw-file index disagrees with its photo record")
            family = per_file.setdefault(raw_file, f"source-{len(per_file) + 1}")
            report["anonymous_source"] = family
            transitions.extend((family, capture_day, *operation) for operation in operations)
        rows[label] = report

    matched = Counter()
    for a, b in combinations(transitions, 2):
        family_a, day_a, field_a, delta_a, start_a, stop_a, rest_a = a
        family_b, day_b, field_b, delta_b, start_b, stop_b, rest_b = b
        if family_a != family_b and field_a == field_b and delta_a == delta_b:
            different = different_fields(rest_a, rest_b)
            same_day = "unknown" if day_a is None or day_b is None else "same" if day_a == day_b else "different"
            key = (field_a, format(delta_a.normalize(), "f"),
                   start_a == start_b and stop_a == stop_b, same_day, len(different),
                   tuple(sorted(different.intersection(KNOWN_COLOR))))
            matched[key] += 1
    matches = [{"field": field, "signed_delta": delta, "same_endpoints": endpoints,
                "capture_day_relation": day_relation, "same_remaining_state": other == 0,
                "other_fields_differing": other, "known_other_color_fields_differing": colors,
                "transition_pairs": count}
               for (field, delta, endpoints, day_relation, other, colors), count in sorted(matched.items())]
    report = {"etag": ETAG, "catalog_offset": DB_START, "index_page": index,
              "sampled_photos": rows, "distinct_linked_raw_files": len(per_file), "matching_changes": matches,
              "range_bytes": 512 + 4096 * len(db.cache),
              "sqlite_pages_sha256": {str(n): sha256(p).hexdigest() for n, p in sorted(db.cache.items())}}
    with output.open("xb") as handle:
        handle.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("ranges", type=Path)
    cli.add_argument("--offline", action="store_true")
    cli.add_argument("--by-file", action="store_true")
    args = cli.parse_args()
    run(args.output, args.ranges, args.offline, args.by_file)
