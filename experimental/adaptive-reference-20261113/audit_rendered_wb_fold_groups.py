"""Audit named scene groups in the author's six MAT v5 fold files."""

import argparse
import itertools
from pathlib import Path
import json
import re
import struct
import zlib


def chunks(payload, compressed=False):
    offset = 0
    while offset < len(payload):
        tag, length = struct.unpack_from("<II", payload, offset)
        if tag >> 16:
            length, tag = tag >> 16, tag & 0xffff
            assert length <= 4
            yield tag, payload[offset + 4:offset + 4 + length]
            offset += 8
        else:
            end = offset + 8 + length
            assert end <= len(payload)
            yield tag, payload[offset + 8:end]
            offset = end if tag == 15 and compressed else (end + 7) & ~7
    assert offset == len(payload)


def parse(path):
    raw = path.read_bytes()
    assert raw[:19] == b"MATLAB 5.0 MAT-file" and raw[126:128] == b"IM"
    groups = {}

    def unpack_matrix(data, field=None):
        records = list(chunks(data))
        array_class = records[0][1][0]
        if array_class == 2:
            assert records[2][1] == b"data"
            width = struct.unpack("<i", records[3][1])[0]
            names = records[4][1]
            for i, (tag, inner) in enumerate(records[5:]):
                assert tag == 14
                label = names[i * width:(i + 1) * width].rstrip(b"\0").decode("ascii")
                unpack_matrix(inner, label)
            return
        assert array_class == 1
        rows, cols = struct.unpack("<ii", records[1][1])
        variable = field or records[2][1].decode("ascii")
        assert len(records[3:]) == rows * cols and variable not in groups
        values = []
        for tag, cell in records[3:]:
            assert tag == 14
            flags, _, _, value = chunks(cell)
            assert flags[1][0] == 4 and value[0] in (4, 16)
            encoding = "utf-16le" if value[0] == 4 else "utf-8"
            values.append(value[1].decode(encoding))
        groups[variable] = values

    for tag,data in chunks(raw[128:], compressed=True):
        assert tag == 15
        [(tag, data)] = chunks(zlib.decompress(data))
        assert tag == 14
        unpack_matrix(data)
    return groups


def normalize(label):
    return label.rstrip("*").rstrip("_")


def physical(key):
    return re.sub(r"((?:Indoor|Outdoor)_Scene_\d+)_\d{4}$", r"\1", key, flags=re.I)


def audit(folder):
    lookup = {}
    per_file = {}
    for fold, suffix in itertools.product(range(1, 4), ("", "_")):
        filename = f"fold{fold}{suffix}.mat"
        rows = parse(folder / filename)
        if set(rows) != {"training", "validation"}:
            raise ValueError(f"Unexpected split fields: {filename}")
        groups = {group: set(map(normalize, labels)) for group, labels in rows.items()}
        training, validation = groups["training"], groups["validation"]
        overlaps = set(map(physical, training)) & set(map(physical, validation))
        per_file[filename] = {
            "rows": {name: len(values) for name, values in rows.items()},
            "unique_labels": {name: len(values) for name, values in groups.items()},
            "exact_in_file_overlap": len(training & validation),
            "named_scene_overlap": len(overlaps),
            "validation_after_excluding_overlapping_named_scenes": sum(
                physical(label) not in overlaps for label in validation),
        }
        if overlaps:
            example = min(overlaps)
            per_file[filename]["example"] = {
                "family": example,
                "training": sorted(label for label in training if physical(label) == example),
                "validation": sorted(label for label in validation if physical(label) == example),
            }
        lookup[filename] = groups
    result = {"per_file": per_file}
    for name, suffix in (("plain", ""), ("underscore", "_")):
        files = [f"fold{fold}{suffix}.mat" for fold in range(1, 4)]
        folds = [lookup[file]["validation"] for file in files]
        universe = set.union(*folds)
        result[name] = {
            "validation_union": len(universe),
            "exact_validation_overlap": [len(a & b) for a, b in itertools.combinations(folds, 2)],
            "named_scene_validation_overlap": [len(set(map(physical, a)) & set(map(physical, b)))
                                               for a, b in itertools.combinations(folds, 2)],
            "sample_8D5U5524": [file for file, group in zip(files, folds) if "8D5U5524" in group],
            "explicit_variant_ids": sum(bool(re.search(r"_[CDFSTG]_(AS|CS)$", label)) for label in universe),
        }
    result["struct_and_cell_groups_match"] = all(
        lookup[f"fold{fold}.mat"] == lookup[f"fold{fold}_.mat"] for fold in range(1, 4))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    values = audit(args.input)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(values, indent=2) + "\n")
