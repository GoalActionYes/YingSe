"""Validate fixed PSBench metadata and the single registered local color example."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from independent_coco import digest
from probe_tst_reference_spatial_signal import measure, structure_score
from reference_color_app.media import decode_image

REVISION = '929d0710eb7c86a988f8a69a74619052b60ac293'
IDS = [f'{i:03d}' for i in (*range(35, 56), *range(201, 241))]
LOCAL_COLOR = ['205', '227', '230', '233', '234', '240']


def audit(folder, out):
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    assert [row[0] for row in metadata] == IDS
    bytes_read = 0
    for item, size, sha in metadata:
        path = folder / f'{item}.json'
        assert path.stat().st_size == size <= 4096 and digest(path) == sha
        task = json.loads(path.read_text(encoding='utf-8'))
        assert task['id'] == item and task['instruction']
        bytes_read += size
    assert bytes_read <= 262144
    tree_path = folder / '230-tree.json'
    names = ['230.json', '230_rubric.json', '230_start.jpg', '230_end.jpg']
    assets = {Path(row['path']).name: row for row in json.loads(tree_path.read_text(encoding='utf-8'))}
    checks = {}
    for name in names:
        path, entry = folder / name, assets[name]
        data = path.read_bytes()
        assert len(data) == entry['size']
        if 'lfs' in entry and entry['lfs']:
            assert digest(path) == entry['lfs']['oid']
        else:
            assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == entry['oid']
        checks[name] = dict(bytes=len(data), sha256=digest(path))
    original, edited = (decode_image(folder / name) for name in names[-2:])
    small = [np.array(Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR))
             for image in (original, edited)]
    geometry = structure_score(*small)
    spatial = None
    if original.shape == edited.shape:
        spatial = measure(original, original, edited, seed=20261118)
    report = dict(revision=REVISION, tasks_checked=len(metadata), task_bytes=bytes_read,
                  task_manifest_sha256=digest(folder / 'metadata.json'), candidates_from_instruction=LOCAL_COLOR,
                  source_files=checks, input_dimensions=list(original.shape[:2]),
                  edited_dimensions=list(edited.shape[:2]), geometry=geometry, neutral_rgb_map=spatial,
                  protocol_sha256=digest(Path(__file__).with_name('psbench-local-color-source-protocol.md')),
                  script_sha256=digest(__file__), verified_cross_scene_same_edit_groups=0)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    audit(args.folder, args.out)
