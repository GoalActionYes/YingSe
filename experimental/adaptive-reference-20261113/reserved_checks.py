"""Fail closed before allowing held-out ground truth retrieval."""
from pathlib import Path
from independent_coco import digest


def verify_frozen_predictions(directory,cases):
    if len(cases)!=32 or len({r['index'] for r in cases})!=32:raise ValueError('Incomplete reserved predictions')
    directory=Path(directory).resolve()
    for row in cases:
        if 'failure' in row or set(row.get('outputs',{}))!={'input','candidate','current','before'}:raise ValueError('All four outputs must be frozen before GT')
        for entry in row['outputs'].values():
            path=(directory/entry['path']).resolve()
            if not path.is_relative_to(directory) or not path.is_file() or digest(path)!=entry['sha256']:raise ValueError('Reserved prediction missing or changed')
