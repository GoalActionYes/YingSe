"""Cross one intentional global grade with the frozen BG-REAL background swap."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from background_intervention import LUMA, target_regions
from probe_bg_real_pixel_pilot import interior, measure
from probe_gray_edge_correction import PARENT_SHA
from probe_local_edit_toggle import COOL, mask_for
from probe_reference_chroma_spread import BASE, PARENT, ROOT, digest, projection, read
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


ROLES = ('C1', 'F1')


def cool(rgb):
    return np.rint((rgb.astype(np.float64) + COOL).clip(0, 255)).astype(np.uint8)


def write(path, rgb):
    Image.fromarray(rgb).save(path)
    return digest(path)


def compare(outputs, expected, mask):
    count = int(mask.sum())
    if count == 0:
        return dict(pixels=0)
    c, f, c_cool, f_cool = [outputs[k].astype(np.int16) for k in ('C1', 'F1', 'C1-cool', 'F1-cool')]
    adjustments = dict(C1=c_cool-c, F1=f_cool-f)
    baseline_swap = float(np.abs(f[mask]-c[mask]).mean())
    cool_swap = float(np.abs(f_cool[mask]-c_cool[mask]).mean())
    return dict(pixels=count, swap_mae=[baseline_swap, cool_swap],
                control_mae=float(np.abs(expected[mask]).mean()),
                interaction_mae=float(np.abs((adjustments['F1']-adjustments['C1'])[mask]).mean()),
                response={k:dict(projection=projection(a, expected, mask),
                                 error=float(np.abs((a-expected)[mask]).mean())) for k,a in adjustments.items()})


def panel(out, refs, outputs):
    tile = 256
    sheet = Image.new('RGB', (4*(tile+8)+8, 2*(tile+34)+8), '#191c20')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for row, arrays in enumerate((refs, outputs)):
        for col, (name, rgb) in enumerate(arrays.items()):
            x, y = 8+col*(tile+8), 8+row*(tile+34)
            draw.text((x,y), ('Reference' if row == 0 else 'YingSe 13')+' / '+name, font=font, fill='#eef1f5')
            thumb = Image.fromarray(rgb)
            thumb.thumbnail((tile,tile), Image.Resampling.LANCZOS)
            sheet.paste(thumb,(x,y+24))
    sheet.save(out, quality=92, subsampling=0)


def gates(scopes):
    passed = {}
    for key in ('person', 'whole'):
        passed[f'{key}_cold'] = all(row['projection'] is not None and .75 <= row['projection'] <= 1.25
                                    and row['error'] <= 3 for row in scopes[key]['response'].values())
    for key in ('person', 'neutral_person'):
        value = scopes[key]
        passed[f'{key}_swap'] = max(value['swap_mae']) <= 1.5 if value['pixels'] >= 256 else None
    passed['interaction'] = scopes['person']['interaction_mae'] <= 1.5
    return passed


def run(out):
    started = time.monotonic()
    torch.set_num_threads(4)
    np.testing.assert_allclose(COOL @ LUMA, 0, atol=1e-12)
    evidence = ROOT / 'docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json'
    parent = PARENT / 'results.json'
    frozen = json.loads(evidence.read_bytes())
    cohort = json.loads(parent.read_bytes())
    if digest(parent) != PARENT_SHA or len(frozen['cases']) != 4 or cohort['strength'] != .8:
        raise ValueError('Unexpected frozen controls')
    inputs = BASE / 'bg-real-pixel-inputs'
    targets = {row['pair']:row for row in cohort['cases']}
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != frozen['checkpoint_sha256'] or engine.checkpoint_sha != cohort['model_sha256']:
        raise ValueError('Frozen 13 checkpoint changed')
    out.mkdir()
    records = {}
    for key, info in frozen['cases'].items():
        if not info['qualified']:
            raise ValueError(f'Unexpected ineligible reference pair: {key}')
        pair = info['target_pair']
        row = targets[pair]
        target = read(row['input_file'])
        files = {k:inputs / v['name'] for k,v in info['input'].items()}
        if any(digest(path) != info['input'][name]['sha256'] for name,path in files.items()):
            raise ValueError(f'Reference identity changed: {key}')
        refs = {role:decode_image(files[role]) for role in ROLES}
        refs.update({f'{role}-cool':cool(refs[role]) for role in ROLES})
        outputs, hashes, ref_hashes = {}, {}, {}
        for role,rgb in refs.items():
            if role.endswith('-cool'):
                ref_hashes[role] = write(out / f'{pair}-ref-{role}.png', rgb)
            output = finish_spatial(target, engine.prepare(target, rgb).apply(target), .8)
            outputs[role] = output
            hashes[role] = write(out / f'{pair}-{role}.png', output)
            if role in ROLES and hashes[role] != info['outputs'][role]:
                raise ValueError(f'Original baseline did not reproduce: {pair} {role}')

        expected = finish_spatial(target, cool(target), .8).astype(np.int16)-target
        human = mask_for(row, pair.split('-')[0])
        masks = target_regions(target, human)
        masks['neutral_person'] = masks['neutral_proxy'] & human
        scopes = {name:compare(outputs,expected,masks[name])
                  for name in ('whole','person','background','neutral_person','neutral_proxy')}
        with Image.open(files['mask']) as im:
            core = interior(np.array(im.convert('L'))>=128)
        subject = measure(refs['C1-cool'], refs['F1-cool'], core)
        clip = {}
        for role in ROLES:
            unbounded = refs[role].astype(np.float64)+COOL
            clip[role] = float(((unbounded<0)|(unbounded>255)).mean())
        checks = gates(scopes)
        records[pair] = dict(source=key, clipped=clip, cold_reference_subject_core=subject,
                             metrics=scopes, checks=checks, passed=all(v for v in checks.values() if v is not None),
                             output_sha256=hashes, cool_reference_sha256=ref_hashes)
        if pair == '872-1268':
            panel(out / 'white-jersey.jpg', refs, outputs)
    if time.monotonic()-started > 120:
        raise RuntimeError('Run exceeded compute budget')
    script = Path(__file__)
    report = dict(cases=records, parent_sha256=PARENT_SHA, source_evidence_sha256=digest(evidence),
                  checkpoint_sha256=engine.checkpoint_sha, source_sha256=digest(script),
                  comparison_sha256=digest(out/'white-jersey.jpg'),
                  protocol_sha256=digest(script.with_name('bg-real-style-cross-protocol.md')))
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for pair,row in records.items():
        print(pair, row['checks'], row['metrics']['person'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
