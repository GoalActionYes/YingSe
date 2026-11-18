"""Screen boundary-connected homogeneous patches as a white surface cue."""
import argparse
from collections import deque
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_mls_material_identity import checked
from independent_coco import ROOT, digest
from probe_multilum_chromaticity import linear_rgb
from reference_color_app.media import decode_image


GRID = 16
SIZE = (256, 208)


def tiles(values):
    return values.reshape(GRID, 13, GRID, 16, *values.shape[2:]).swapaxes(1, 2).reshape(
        GRID, GRID, 208, *values.shape[2:])


def propose(rgb):
    sample = np.asarray(Image.fromarray(rgb).resize(SIZE, Image.Resampling.BILINEAR))
    raw = tiles(sample.astype(float))
    luma = raw @ np.array([.299, .587, .114])
    valid = ((raw >= 16) & (raw <= 239)).all(axis=-1)
    logs = np.log(np.maximum(linear_rgb(sample), 1/(255*12.92)))
    colors = tiles(logs[..., [0,2]]-logs[..., 1:2])
    valid_counts = valid.sum(axis=-1)
    denominator = np.maximum(valid_counts,1)[...,None]
    chroma = np.where(valid[...,None], colors, 0).sum(axis=2)/denominator
    spread = np.sqrt(np.where(valid[...,None], (colors-chroma[:,:,None,:])**2, 0).sum(axis=2)/denominator)
    supported = (valid_counts>104) & (luma.std(axis=-1)<=12) & (spread<=.12).all(axis=-1)
    edge = {(r,c) for r,c in zip(*np.nonzero(supported)) if r==0 or c in (0,GRID-1)}
    queue = deque(sorted(edge))
    seen = set(edge)
    while queue:
        r,c = queue.popleft()
        for i,j in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if not (0<=i<GRID and 0<=j<GRID and supported[i,j]) or (i,j) in seen:
                continue
            if np.linalg.norm(chroma[r,c]-chroma[i,j])<=.12:
                seen.add((i,j))
                queue.append((i,j))
    grid = np.zeros((GRID,GRID), dtype=bool)
    if len(seen)>=32:
        for cell in seen:
            grid[cell] = True
    return np.repeat(np.repeat(grid,13,axis=0),16,axis=1), dict(
        valid_tiles=int((valid_counts>104).sum()), smooth_tiles=int(supported.sum()), connected_tiles=len(seen))


def run(base, out):
    docs = ROOT/'docs/research'
    lp = docs/'2026-11-18-mls-linear-pilot-evidence.json'
    jp = docs/'2026-11-18-mls-light-response-evidence.json'
    linear, preview = [json.loads(path.read_bytes()) for path in (lp,jp)]
    cases, inferred = {}, {}
    out.mkdir()
    input_sets = {
        'linear': {name:sha for scene in ('02','03')
                   for name,sha in linear['scenes'][scene]['inputs_sha256'].items()},
        'preview': {name:record['sha256'] for name,record in preview['input_members'].items()
                    if name[:2] in ('13','22')},
    }
    assert all(len(values)==4 for values in input_sets.values())
    for encoding, entries in input_sets.items():
        folder = base/('mls-linear-pilot' if encoding=='linear' else 'mls-light-pilot-inputs')
        for filename, sha in entries.items():
            path = folder/filename
            assert digest(path)==sha
            scene = filename[:2]
            label = encoding+'-'+path.stem
            predicted, stats = propose(decode_image(path))
            file = out/(label+'.png')
            Image.fromarray(predicted.astype('uint8')*255).save(file)
            inferred[label] = predicted
            cases[label] = dict(scene=scene, encoding=encoding, input_sha256=sha, mask_sha256=digest(file), **stats)

    materials = base/'mls-material-identity'
    with checked(materials, 'surfaces.zip') as zip_spectra:
        table = list(csv.DictReader(io.StringIO(zip_spectra.read('surfaces/mask_to_surface.csv').decode())))
    with checked(materials, 'masks_preview.zip') as masks:
        for label, record in cases.items():
            scene = record['scene']
            names = [row for row in table if int(row['scene'])==int(scene)
                     and row['surface'] in ('background_white','background_red')]
            assert len(names)==1
            physical = names[0]['surface']
            color = np.array(names[0]['mask_colour_rgb'].split(), dtype=int)
            with Image.open(io.BytesIO(masks.read(f'masks_preview/{scene}.png'))) as image:
                authored = (np.asarray(image.resize(SIZE,Image.Resampling.NEAREST))==color).all(-1)
            candidate = inferred[label]
            intersection = int((candidate&authored).sum())
            total = int(candidate.sum())
            precision = intersection/total if total else None
            recall = intersection/int(authored.sum())
            passed = not total if physical=='background_red' else bool(total and precision>=.9 and recall>=.2)
            record.update(material=physical, proposed=int(total), true_pixels=int(authored.sum()), intersection=intersection,
                          precision=precision, coverage=recall, passed=passed)
    necessary = [row for row in cases.values() if row['encoding']=='linear' or row['material']=='background_red']
    data = dict(cases=cases, proceeded=all(row['passed'] for row in necessary), linear_sha256=digest(lp),
                preview_sha256=digest(jp), source_sha256=digest(Path(__file__)))
    (out/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({name:(row['material'],row['connected_tiles'],row['passed']) for name,row in cases.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.base,args.out)
