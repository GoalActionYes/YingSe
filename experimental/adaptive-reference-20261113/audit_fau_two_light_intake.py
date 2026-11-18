"""Verify a bounded laboratory sample of the official FAU MIRF archive."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import tarfile

import numpy as np
from PIL import Image

from reference_color_app.media import ffmpeg_path


ARCHIVE_SHA = '8a808c59c1896d68fe9b7a52a3e7cbb6434fa2bbf6323a8090ba7ffcf44ef98c'
ROLES = dict(img=(16, 2), groundtruth=(8, 2), masks=(8, 0), srgb8bit=(8, 2))
LAB = 'benchmark_mirf/lab'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def native_rgb(data, height, width):
    result = subprocess.run(
        [str(ffmpeg_path()), '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', 'pipe:0',
         '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', 'pipe:1'],
        input=data, capture_output=True, check=True, timeout=20,
    )
    assert len(result.stdout) == height * width * 6
    return np.frombuffer(result.stdout, dtype='<u2').reshape(height, width, 3)


def read_case(archive, files, scene, folder):
    arrays, info = {}, {}
    for role, encoding in ROLES.items():
        source = f'{LAB}/{role}/{scene}.png'
        member = files[source]
        assert 0 < member.size <= 2*1024**2
        data = archive.extractfile(member).read()
        assert len(data) == member.size and data[:16] == b'\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR'
        width, height, depth, kind = struct.unpack('>IIBB', data[16:26])
        assert (width, height) == (452, 260) and (depth, kind) == encoding
        if role == 'img':
            image = native_rgb(data, height, width)
        else:
            with Image.open(io.BytesIO(data)) as decoded:
                image = np.array(decoded)
        arrays[role] = image
        info[role] = dict(archive_path=source, bytes=len(data), sha256=sha(data), depth=depth,
                          min=int(image.min()), max=int(image.max()))
        (folder / f'{role}.png').write_bytes(data)
    mask = arrays['masks']
    assert set(np.unique(mask)) == {0, 255}
    gt = arrays['groundtruth'].astype(float)
    valid = (mask == 255) & (np.linalg.norm(gt, axis=-1) > 0)
    assert valid.any()
    values = gt[valid]
    normalized = values / np.linalg.norm(values, axis=1, keepdims=True)
    direction = normalized.mean(axis=0)
    axis = direction / np.linalg.norm(direction)
    angular = np.degrees(np.arccos(np.clip(normalized @ axis, -1, 1)))
    variation = float(np.quantile(angular, .95))
    assert variation > 0
    return dict(files=info, shape=[260,452], valid_pixels=int(valid.sum()), valid_fraction=float(valid.mean()),
                angular_q95_from_mean=variation, unique_valid_illuminants=len(np.unique(values, axis=0)))


def run(path, out):
    assert path.stat().st_size == 42311680 and sha(path.read_bytes()) == ARCHIVE_SHA
    out.mkdir()
    with tarfile.open(path, 'r:gz') as archive:
        members = [entry for entry in archive if entry.isfile()]
        assert len(members) == len({entry.name for entry in members})
        assert all(not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in members)
        files = {entry.name: entry for entry in members}
        counts = Counter(str(PurePosixPath(name).parent) for name in files)
        roles = [{PurePosixPath(name).stem for name in files if name.startswith(f'{LAB}/{role}/')}
                 for role in ROLES]
        candidates = sorted(set.intersection(*roles))
        assert len(candidates) == 58
        chosen = {}
        for key in candidates:
            match = re.fullmatch(r'(.+)_([wrb])_([wrb])', key)
            assert match and match[2] != match[3]
            chosen.setdefault(match[1], key)
        cases = {}
        for key in list(chosen.values())[:2]:
            directory = out / key
            directory.mkdir()
            cases[key] = read_case(archive, files, key, directory)
        assert len(cases) == 2
        overview = archive.extractfile('benchmark_mirf/README_overview.txt').read()
    report = dict(archive_sha256=ARCHIVE_SHA, archive_bytes=path.stat().st_size, member_counts=counts,
                  complete_lab_combinations=len(candidates), lab_name_prefixes=len(chosen),
                  overview_sha256=sha(overview), cases=cases, separate_influence_map=False,
                  source_sha256=sha(Path(__file__).read_bytes()))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: [row['valid_fraction'], row['angular_q95_from_mean']] for key,row in cases.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
