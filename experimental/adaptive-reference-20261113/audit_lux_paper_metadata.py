"""Audit whether public white-paper captures identify controlled lamp-color swaps."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path


ANNOTATION_SHA = '1fe82940c4eeed2c64a169269e8184277cf8cf97701b64cd3cd3d7037d04d8e9'
DICTIONARY_SHA = '316b8f68e30ad89eba215ec75eeaa460fcf72b66103a8ec1feda273562aa78e4'
LUX = ('lux_C', 'lux_UL', 'lux_UR', 'lux_LR', 'lux_LL')


def read(path, expected):
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Unexpected archive member: {path.name}')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def audit(annotation, dictionary, out):
    records = read(annotation, ANNOTATION_SHA)
    keys = set(records[0])
    schema = {row['column'] for row in read(dictionary, DICTIONARY_SHA)}
    if keys != schema or len(records) != 5814:
        raise ValueError('Incorrect release table or schema')
    white = [row for row in records if row['surface_group_norm'] == 'white_paper']
    groups = defaultdict(list)
    for row in white:
        if row['label_kind'] != '5point' or not row['point'] or not row['physical_point_id']:
            continue
        if all(row[field] for field in LUX):
            groups[row['table'], row['physical_point_id']].append(row)
    feasible = Counter()
    for group in groups.values():
        for a, b in combinations(group, 2):
            if a['session'] != b['session'] and all(
                abs(float(a[field]) - float(b[field])) <= .10 * max(float(a[field]), float(b[field]))
                for field in LUX
            ):
                feasible[a['family_group']] += 1
    assignment = sorted(key for key in keys if any(token in key.lower() for token in ('lamp', 'cct', 'temperature', 'illuminant')))
    result = dict(archive_sha=[ANNOTATION_SHA, DICTIONARY_SHA], rows=len(records), fields=sorted(keys),
                  white_count=len(white), white_kind_counts=Counter(row['label_kind'] for row in white),
                  white_family_counts=Counter(row['family_group'] for row in white),
                  five_point_rows_with_point=sum(map(len, groups.values())), available_points=len(groups),
                  five_point_within_ten_percent_by_family=feasible,
                  five_point_provisional_pair_count=sum(feasible.values()),
                  lamp_assignment_columns=assignment, certified_pairs=0 if not assignment else None)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('annotation', 'dictionary', 'out'):
        parser.add_argument(name, type=Path)
    audit(**vars(parser.parse_args()))
