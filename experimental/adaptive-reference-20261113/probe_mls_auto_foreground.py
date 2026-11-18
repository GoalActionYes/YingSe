"""Replace known MLS material masks with independent tile-region proposals."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_mls_material_identity import checked
from background_intervention import target_regions
from fetch_mls_linear_pilot import LIGHTS, SCENES
from independent_coco import ROOT, digest
from probe_mls_auto_anchor_replay import automatic
from probe_mls_border_surface import GRID, SIZE, propose, tiles
from probe_mls_foreground_corroboration import supported
from probe_mls_light_response import GROUPS, evaluate
from probe_mls_linear_pilot import valid_pixels
from probe_mls_neutral_surface import SURFACE
from probe_multilum_chromaticity import linear_rgb
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


def local_regions(rgb):
    sample = np.asarray(Image.fromarray(rgb).resize(SIZE, Image.Resampling.BILINEAR))
    mask, _ = propose(rgb)
    if not mask.any():
        return []
    logs = np.log(np.maximum(linear_rgb(sample),1/(255*12.92)))
    chroma = logs[..., [0,2]]-logs[..., 1:2]
    valid = valid_pixels(sample,sample)
    background = chroma[mask&valid].mean(axis=0)
    available = {}
    blocks, valid_tiles = tiles(chroma), tiles(valid)
    for i,j in zip(*np.nonzero(~mask[::13,::16])):
        part = blocks[i,j][valid_tiles[i,j]]
        if len(part)>104 and (part.std(axis=0)<=.12).all():
            mean = part.mean(axis=0)
            if np.linalg.norm(mean-background)>=.20:
                available[i,j] = mean
    visited = set()
    regions = []
    for first in sorted(available):
        if first in visited:
            continue
        cells = []
        stack = [first]
        visited.add(first)
        while stack:
            i,j = current = stack.pop()
            cells.append(current)
            for other in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                if other in available and other not in visited and np.linalg.norm(
                    available[other]-available[current])<=.12:
                    visited.add(other)
                    stack.append(other)
        if len(cells)>=2:
            mean = np.mean([available[cell] for cell in cells],axis=0)
            regions.append(dict(cells=[[int(i),int(j)] for i,j in sorted(cells)], chroma=mean.tolist()))
    return regions


def correspond(first, second, background):
    length = np.linalg.norm(background)
    if not first or not second or not length:
        return []
    orthogonal = np.array([-background[1],background[0]])/length

    def features(regions):
        return np.array([np.array([np.asarray(row['chroma'])@orthogonal,.1*np.log(len(row['cells']))])
                         for row in regions])
    distances = np.abs(features(first)[:,None,:]-features(second)[None,:,:]).sum(axis=-1)
    matches = []
    nearest = distances.argmin(axis=0)
    for i,j in enumerate(distances.argmin(axis=1)):
        if nearest[j]!=i or distances[i,j]>.25:
            continue
        delta = np.array(second[j]['chroma'])-first[i]['chroma']
        matches.append(dict(target=i, reference=int(j), distance=float(distances[i,j]),
                            **supported(background,delta)))
    return matches


def material_at(region, masks):
    selected = np.zeros((GRID,GRID),bool)
    for i,j in region['cells']:
        selected[i,j] = True
    area = np.repeat(np.repeat(selected,13,axis=0),16,axis=1)
    for name,truth in masks.items():
        if (truth&area).sum()>.5*area.sum():
            return name
    return None


def run(base, out):
    docs = ROOT/'docs/research'
    old_path = docs/'2026-11-18-mls-linear-pilot-evidence.json'
    auto_path = docs/'2026-11-18-mls-auto-anchor-replay-evidence.json'
    guide_path = docs/'2026-11-18-mls-foreground-corroboration-evidence.json'
    old, auto, guide = (json.loads(path.read_bytes()) for path in (old_path,auto_path,guide_path))
    assert guide['proceeded'] and guide['auto_evidence_sha256']==digest(auto_path)
    assert guide['baseline_sha256']==digest(old_path)
    cache = base/'mls-linear-pilot'
    photos = {}
    for scene in SCENES:
        for light in LIGHTS:
            path = cache/f'{scene}_{light}.png'
            assert digest(path)==old['scenes'][scene]['inputs_sha256'][path.name]
            photos[scene,light] = decode_image(path)
    out.mkdir()
    original, changed = LIGHTS
    report = {}
    for scene in SCENES:
        donor, _ = GROUPS[scene]
        target = photos[scene,original]
        first = local_regions(target)
        warm = base/'mls-auto-anchor-replay'/f'{scene}-tinted-reference.png'
        blue = base/'mls-foreground-corroboration'/f'{scene}-blue-reference.png'
        assert digest(warm)==auto['scenes'][scene]['tinted_reference_sha256']
        assert digest(blue)==guide['scenes'][scene]['blue_reference_sha256']
        refs = dict(same_setting=photos[donor,original], other_swap=photos[donor,changed],
                    surface_tint=decode_image(warm), blue_tint=decode_image(blue))
        arms = {}
        for arm,reference in refs.items():
            proposal, details = automatic(target,reference)
            second = local_regions(reference)
            pairs = correspond(first, second, np.array(details['offset'])) if details['anchor'] else []
            accepted = sum(pair['supports'] for pair in pairs)>=2
            path = out/f'{scene}-{arm}.png'
            Image.fromarray(proposal if accepted else target).save(path)
            arms[arm] = dict(accepted=accepted, reference=second, pairs=pairs, sha256=digest(path))
        report[scene] = dict(target=first, arms=arms)

    materials = base/'mls-material-identity'
    with checked(materials,'surfaces.zip') as spectra:
        rows = list(csv.DictReader(io.StringIO(spectra.read('surfaces/mask_to_surface.csv').decode())))
    truths = {}
    with checked(materials,'masks_preview.zip') as masks:
        for scene in SCENES:
            with Image.open(io.BytesIO(masks.read(f'masks_preview/{scene}.png'))) as image:
                native = np.asarray(image.resize((1157,950),Image.Resampling.NEAREST))
                sized = np.asarray(image.resize(SIZE,Image.Resampling.NEAREST))
            colors = {row['surface']:np.array(row['mask_colour_rgb'].split(),dtype=int) for row in rows
                      if int(row['scene'])==int(scene)}
            truths[scene] = ((native==colors[SURFACE]).all(-1), {
                name:(sized==code).all(-1) for name,code in colors.items() if name!=SURFACE})
    for scene,item in report.items():
        donor, _ = GROUPS[scene]
        target, change = (photos[scene,light] for light in LIGHTS)
        white, foreground = truths[scene]
        neutral = target_regions(target,white)['neutral_proxy'] & ~white
        valid = valid_pixels(target,change)
        expected = finish_spatial(target,change,.8)
        for arm,entry in item['arms'].items():
            output = decode_image(out/f'{scene}-{arm}.png')
            for pair in entry['pairs']:
                pair['target_material'] = material_at(item['target'][pair['target']],foreground)
                pair['reference_material'] = material_at(entry['reference'][pair['reference']],truths[donor][1])
            entry['score'] = evaluate(target,output,expected,neutral,valid,True)
        real = item['arms']['other_swap']
        correct = {pair['target_material'] for pair in real['pairs'] if pair['supports'] and pair['target_material']
                   and pair['target_material']==pair['reference_material']}
        gain = 1-real['score']['error']/auto['scenes'][scene]['arms']['other_swap']['baseline']['error']
        item['correct_materials'] = sorted(correct)
        item['swap_gain'] = gain
        item['passed'] = bool(real['score']['lamp_pass'] and gain>=.15 and len(correct)>=2 and all(
            arm['score']['same_setting_pass'] for name,arm in item['arms'].items() if name!='other_swap'))
    result = dict(scenes=report,proceeded=all(row['passed'] for row in report.values()),
                  source_sha256=digest(Path(__file__)), control_evidence_sha256=digest(guide_path))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({scene:{name:[row['accepted'],len(row['pairs'])] for name,row in item['arms'].items()}
                      for scene,item in report.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    run(args.base,args.out)
