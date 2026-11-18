"""Replay all reference histogram channel subsets on the BG-REAL style cross."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from background_intervention import target_regions
from probe_bg_real_style_cross import compare, cool, gates, write
from probe_gray_edge_correction import PARENT_SHA
from probe_local_edit_toggle import mask_for
from probe_reference_chroma_spread import BASE, PARENT, ROOT, digest, read
from reference_color_app.media import decode_image
from spatial_render import PreparedSpatial, SpatialEngine, finish_spatial


LABELS = {1:'Y', 2:'Cb', 3:'Y-Cb', 4:'Cr', 5:'Y-Cr', 6:'Cb-Cr', 7:'full'}
ROLES = ('C1', 'F1', 'C1-cool', 'F1-cool')


def infer(engine, target, low, target_hist, reference_hist):
    grid = engine.api['lowpass'](engine.model.apply_grid(low, torch.cat((target_hist, reference_hist), 1)))
    return finish_spatial(target, PreparedSpatial(engine, grid, {}).apply(target), .8)


def histogram(engine, photo):
    tensor = engine.analysis(photo)
    return engine.api['batch_histogram_1D'](engine.api['srgb_to_ycbcr'](tensor))


def interaction_metrics(predicted, masks):
    interaction = {
        k: pictures['F1-cool'].astype(np.int16) - pictures['C1-cool'] - pictures['F1'] + pictures['C1']
        for k,pictures in predicted.items()
    }
    unique = {}
    for k in LABELS:
        unique[k] = interaction[k] - sum(part for j,part in unique.items() if (j & k) == j)
    np.testing.assert_array_equal(sum(unique.values()), interaction[7])
    scores = {}
    for key,mask in masks.items():
        if not mask.any():
            scores[key] = None
            continue
        full = interaction[7][mask]
        base = float(np.abs(full).mean())
        scores[key] = dict(
            full_mae=base,
            subset_residual={label:float(np.abs(interaction[k][mask] - full).mean())
                             for k,label in LABELS.items()},
            nonadditive_mae={label:float(np.abs(unique[k][mask]).mean()) for k,label in LABELS.items()},
        )
    return scores


def panel(predicted, out):
    tile = 220
    sheet = Image.new('RGB', (4*(tile+8)+8, 3*(tile+34)+8), '#191c20')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for row,k in enumerate((7, 6, 1)):
        for col,role in enumerate(ROLES):
            x,y = 8+col*(tile+8),8+row*(tile+34)
            draw.text((x,y), f'{LABELS[k]} / {role}',font=font,fill='#eef1f5')
            im = Image.fromarray(predicted[k][role])
            im.thumbnail((tile,tile),Image.Resampling.LANCZOS)
            sheet.paste(im,(x,y+24))
    sheet.save(out,quality=92,subsampling=0)


@torch.inference_mode()
def run(out):
    start = time.monotonic()
    torch.set_num_threads(4)
    docs = ROOT/'docs/research'
    evidence = docs/'2026-11-18-bg-real-style-cross-evidence.json'
    before = docs/'2026-11-18-bg-real-pixel-pilot-evidence.json'
    source = json.loads(evidence.read_bytes())
    if digest(before) != source['source_evidence_sha256'] or digest(PARENT/'results.json') != PARENT_SHA:
        raise ValueError('Frozen parent changed')
    cohort = {row['pair']:row for row in json.loads((PARENT/'results.json').read_bytes())['cases']}
    inputs = json.loads(before.read_bytes())['cases']
    cross = BASE/'bg-real-style-cross'
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != source['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    out.mkdir()
    records = {}
    for pair, parent in source['cases'].items():
        row = cohort[pair]
        saved = inputs[parent['source']]['input']
        folder = out/pair
        folder.mkdir()
        target = read(row['input_file'])
        c = engine.analysis(target)
        low = engine.api['TF'].resize(c,(64,64),antialias=True)
        hc = engine.api['batch_histogram_1D'](engine.api['srgb_to_ycbcr'](c))
        hist = {}
        for role in ROLES:
            if role in ('C1','F1'):
                path = BASE/'bg-real-pixel-inputs'/saved[role]['name']
                sha = saved[role]['sha256']
            else:
                path = cross/f'{pair}-ref-{role}.png'
                sha = parent['cool_reference_sha256'][role]
            if digest(path) != sha:
                raise ValueError(f'Unexpected reference: {pair}/{role}')
            hist[role] = histogram(engine,decode_image(path))
        anchor = infer(engine,target,low,hc,hist['C1'])
        hashes = {'C1.png':write(folder/'C1.png',anchor)}
        if hashes['C1.png'] != parent['output_sha256']['C1']:
            raise ValueError('Anchor render changed')
        predicted = {}
        for k,label in LABELS.items():
            active = [bool(k & (1 << i)) for i in range(3)]
            selection = torch.tensor(active, device=engine.device).view(1,3,1)
            pictures = {'C1':anchor}
            for role in ROLES[1:]:
                hybrid = torch.where(selection,hist[role],hist['C1'])
                rgb = infer(engine,target,low,hc,hybrid)
                name = f'{role}-{label}.png'
                hashes[name] = write(folder/name,rgb)
                if k == 7 and hashes[name] != parent['output_sha256'][role]:
                    raise ValueError(f'Full replay changed: {pair}/{role}')
                pictures[role] = rgb
            predicted[k] = pictures
        person = mask_for(row,pair.split('-')[0])
        masks = target_regions(target,person)
        masks['neutral_person'] = person & masks['neutral_proxy']
        expected = finish_spatial(target,cool(target),.8).astype(np.int16)-target
        metrics = {label:{name:compare(predicted[k],expected,mask) for name,mask in masks.items()}
                   for k,label in LABELS.items()}
        checks = {label:gates(row) for label,row in metrics.items()}
        if checks['full'] != parent['checks']:
            raise ValueError('Full replay gates changed')
        records[pair] = dict(artifacts=hashes,metrics=metrics,checks=checks,
                             interaction=interaction_metrics(predicted,masks))
        if pair == '872-1268':
            panel(predicted,out/'white-jersey.jpg')
    if time.monotonic()-start > 120:
        raise RuntimeError('Run exceeded compute budget')
    score = {label:sum(all(v for v in row['checks'][label].values() if v is not None)
                       for row in records.values()) for label in LABELS.values()}
    script = Path(__file__)
    report = dict(cases=records,eligible_groups=score,previous_sha256=digest(evidence),
                  checkpoint_sha256=engine.checkpoint_sha,source_sha256=digest(script),
                  comparison_sha256=digest(out/'white-jersey.jpg'),
                  protocol_sha256=digest(script.with_name('bg-real-channel-replay-protocol.md')))
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(score)
    for pair,row in records.items():
        print(pair,row['interaction']['person'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    run(parser.parse_args().out)
