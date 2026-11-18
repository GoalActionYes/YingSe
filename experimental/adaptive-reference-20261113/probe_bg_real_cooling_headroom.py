"""Measure crossed cooling with unclipped RGB and exactly fixed reference luma."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from background_intervention import target_regions
from probe_bg_real_channel_replay import ROLES, infer
from probe_bg_real_style_cross import compare, cool, gates, write
from probe_gray_edge_correction import PARENT_SHA
from probe_local_edit_toggle import COOL, mask_for
from probe_reference_chroma_spread import BASE, PARENT, ROOT, digest, read
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


ARMS = ('per_background_y', 'common_background_y')


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cold_shift(engine):
    transform = engine.api['srgb_to_ycbcr']
    basis = torch.eye(3, device=engine.device).view(3,3,1,1)
    y_weights = transform(basis)[:,0,0,0]
    shift = torch.as_tensor(COOL/255, dtype=basis.dtype, device=engine.device)
    # The model's exact Y coefficients differ slightly from the diagnostic weights.
    shift[1] -= (y_weights @ shift) / y_weights[1]
    return shift.view(1,3,1,1), y_weights.detach().cpu().tolist()


def reference_histograms(engine, photo, shift):
    transform = engine.api['srgb_to_ycbcr']
    histogram = engine.api['batch_histogram_1D']
    rgb = engine.analysis(photo) * (228 / 255) + (9 / 255)
    cooled = rgb + shift
    if not bool(((cooled>=0) & (cooled<=1)).all()):
        raise ValueError('Headroom did not prevent RGB clipping')
    base, grade = transform(rgb), transform(cooled)
    y_error = float((grade[:, :1]-base[:, :1]).abs().max())
    if y_error > 2e-7:
        raise ValueError('Unexpected luma difference before freezing Y')
    grade = torch.cat((base[:, :1], grade[:, 1:]), dim=1)
    hist, cold_hist = (histogram(tensor) for tensor in (base, grade))
    if not torch.equal(hist[:, :1], cold_hist[:, :1]):
        raise ValueError('Reference cold grade changed the Y histogram')
    stats = dict(cold_rgb_range=[float(cooled.min())*255,float(cooled.max())*255],
                 raw_luma_max_difference=y_error, base_y_sha256=tensor_digest(hist[:, :1]),
                 cold_y_sha256=tensor_digest(cold_hist[:, :1]),
                 chroma_hist_l1=(cold_hist[:, 1:]-hist[:, 1:]).abs().sum(-1).flatten().tolist())
    return hist, cold_hist, stats


def panel(old, arms, out):
    rows = {'Original base':old, 'No clip':arms[ARMS[0]], 'Common reference Y':arms[ARMS[1]]}
    tile = 220
    sheet = Image.new('RGB', (4*(tile+8)+8, len(rows)*(tile+34)+8), '#191c20')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for row,(name, pictures) in enumerate(rows.items()):
        for col,role in enumerate(ROLES):
            x,y = 8+col*(tile+8),8+row*(tile+34)
            draw.text((x,y), f'{name} / {role}',font=font,fill='#eef1f5')
            thumb = Image.fromarray(pictures[role])
            thumb.thumbnail((tile,tile),Image.Resampling.LANCZOS)
            sheet.paste(thumb,(x,y+24))
    sheet.save(out,quality=92,subsampling=0)


@torch.inference_mode()
def run(out):
    start = time.monotonic()
    torch.set_num_threads(4)
    docs = ROOT/'docs/research'
    cross = docs/'2026-11-18-bg-real-style-cross-evidence.json'
    pilot = docs/'2026-11-18-bg-real-pixel-pilot-evidence.json'
    previous = json.loads(cross.read_bytes())
    raw = json.loads(pilot.read_bytes())
    if digest(pilot) != previous['source_evidence_sha256'] or digest(PARENT/'results.json') != PARENT_SHA:
        raise ValueError('Frozen controls changed')
    cohort = {r['pair']:r for r in json.loads((PARENT/'results.json').read_bytes())['cases']}
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != previous['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    shift, y_weights = cold_shift(engine)
    out.mkdir()
    results = {}
    for pair, old in previous['cases'].items():
        row = cohort[pair]
        target = read(row['input_file'])
        content = engine.analysis(target)
        low = engine.api['TF'].resize(content,(64,64),antialias=True)
        hc = engine.api['batch_histogram_1D'](engine.api['srgb_to_ycbcr'](content))
        hist, stats = {}, {}
        for role in ROLES[:2]:
            entry = raw['cases'][old['source']]['input'][role]
            path = BASE/'bg-real-pixel-inputs'/entry['name']
            if digest(path) != entry['sha256']:
                raise ValueError(f'Reference changed: {pair}/{role}')
            hist[role],hist[f'{role}-cool'],stats[role] = reference_histograms(engine,decode_image(path),shift)
        prepared, output_hashes, hist_hashes = {}, {}, {}
        for arm in ARMS:
            panels, files, identities = {}, {}, {}
            for role in ROLES:
                condition = hist[role]
                if arm == 'common_background_y':
                    condition = torch.cat((hist['C1'][:, :1], condition[:,1:]),dim=1)
                panels[role] = infer(engine,target,low,hc,condition)
                filename = f'{pair}-{arm}-{role}.png'
                files[role] = write(out/filename,panels[role])
                identities[role] = tensor_digest(condition[:, :1])
            prepared[arm], output_hashes[arm], hist_hashes[arm] = panels, files, identities
        person = mask_for(row,pair.split('-')[0])
        scopes = target_regions(target,person)
        scopes['neutral_person'] = person & scopes['neutral_proxy']
        expected = finish_spatial(target,cool(target),.8).astype(np.int16)-target
        scores = {arm:{k:compare(data,expected,mask) for k,mask in scopes.items()} for arm,data in prepared.items()}
        checks = {arm:gates(records) for arm,records in scores.items()}
        results[pair] = dict(reference_stats=stats,y_histogram_sha256=hist_hashes,
                             output_sha256=output_hashes,scores=scores,checks=checks,
                             old_person_interaction=old['metrics']['person']['interaction_mae'])
        if pair == '872-1268':
            old_outputs = {}
            for role in ROLES:
                path = BASE/'bg-real-style-cross'/f'{pair}-{role}.png'
                if digest(path) != old['output_sha256'][role]:
                    raise ValueError('Saved old panel image changed')
                old_outputs[role] = decode_image(path)
            panel(old_outputs,prepared,out/'white-jersey.jpg')
    if time.monotonic()-start > 120:
        raise RuntimeError('Run exceeded compute budget')
    source = Path(__file__)
    report = dict(cases=results,comparison_sha256=digest(out/'white-jersey.jpg'),
                  model_y_weights=y_weights, reference_cool_rgb8=(shift.flatten()*255).detach().cpu().tolist(),
                  cross_sha256=digest(cross),checkpoint_sha256=engine.checkpoint_sha,source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('bg-real-cooling-headroom-protocol.md')))
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for pair,row in results.items():
        print(pair, row['old_person_interaction'],
              {arm:(scopes['person']['interaction_mae'], scopes['person']['response']) for arm,scopes in row['scores'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
