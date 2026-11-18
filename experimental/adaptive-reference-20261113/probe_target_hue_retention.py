"""Compare fixed-luma original chroma with target-hue saturation transfer."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from probe_gray_edge_correction import PARENT_SHA, authored, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_neutral_relation_balance import PSBENCH_SHA, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, ROOT, decision, digest
from probe_roundtrip_grade_choice import checked_json
from probe_sparks_shared_chroma import LUMA, chroma, render


class ChromaProbe:
    def __init__(self, retain_saturation):
        self.retain_saturation = retain_saturation

    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(same_image=True, compressed=0., max_luma_error=0.)
        target_y = target @ LUMA
        grade_y = baseline @ LUMA
        source = target - target_y[..., None]
        color = chroma(target, target_y)
        if self.retain_saturation:
            grade = baseline - grade_y[..., None]
            magnitude = np.linalg.norm(source, axis=-1)
            desired = np.linalg.norm(grade, axis=-1)
            blend = np.clip((np.ptp(target, axis=-1).astype(float) - 12) / 12, 0, 1)
            relative = np.divide(desired - magnitude, magnitude, out=np.zeros_like(magnitude), where=magnitude > 0)
            color *= (1 + relative * blend)[..., None]
        result, compressed = render(color, grade_y)
        # Compare the four-decimal Rec.709 coefficients exactly at the half-code gate.
        luma_steps = (result.astype(np.int32) - baseline) @ np.rint(10000 * LUMA).astype(np.int32)
        return result, dict(same_image=False, compressed=compressed,
                            max_luma_error=float(np.abs(luma_steps).max() / 10000))


def summarize(background, local, real):
    counts = dict(local=0, whole=0, no_person=0, authored=sum(row['gate'] for row in real.values()))
    for row in local.values():
        checks = row['gates']['checks']
        if row['gates']['positive']:
            counts['local'] += all(value for name, value in checks.items()
                                   if name.startswith('local_') and value is not None)
        counts['whole'] += checks['whole_response']
        counts['no_person'] += checks['no_person'] is True
    authored_safe = all(row['gate'] or all(
        part['methods']['candidate']['mae'] <= part['methods']['current13']['mae'] + 2
        for part in row['metrics'].values()) for row in real.values())
    luma = max([variant['max_luma_error'] for row in background for variant in row['variants'].values()]
               + [variant['max_luma_error'] for row in local.values() for variant in row['output'].values()]
               + [row['max_luma_error'] for row in real.values()])
    bg = decision(background)
    passed = (bg['proceed'] and counts['local'] >= 3 and counts['whole'] >= 4 and counts['no_person'] == 2
              and counts['authored'] >= 1 and authored_safe and luma <= .5)
    return dict(background=bg, **counts, authored_safe=authored_safe, max_luma_error=luma, passed=passed)


def panel(out):
    pair = '872-1268'
    images = []
    toggle = BASE / 'local-edit-toggle-v1' / pair
    bg = PARENT / 'results.json'
    frozen = next(row for row in json.loads(bg.read_text(encoding='utf-8'))['cases'] if row['pair'] == pair)
    variant = next(row for row in frozen['variants'] if row['name'] == 'background_magenta_100')
    images.append((PARENT / variant['methods']['adaptive']['file']['path'],
                   out / 'original_chroma' / 'background' / pair / 'background_magenta_100.png',
                   out / 'retain_saturation' / 'background' / pair / 'background_magenta_100.png'))
    for role in ('local', 'whole'):
        images.append((toggle / f'{role}-current13.png',
                       out / 'original_chroma' / 'local' / f'{pair}-{role}.png',
                       out / 'retain_saturation' / 'local' / f'{pair}-{role}.png',
                       toggle / f'{role}-expected.png'))
    tile = 256
    sheet = Image.new('RGB', (4 * (tile + 8) + 8, 3 * (tile + 34) + 8), '#191c20')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for row, files in enumerate(images):
        for col, path in enumerate(files):
            x, y = 8 + col * (tile + 8), 8 + row * (tile + 34)
            label = ('BG swap', 'Local cool', 'Whole cool')[row] + ' / ' + (
                '13', 'Original chroma', 'Target hue', 'Control')[col]
            draw.text((x, y), label, font=font, fill='#edf1f5')
            with Image.open(path) as image:
                sheet.paste(image.convert('RGB'), (x, y + 24))
    path = out / 'white-jersey.jpg'
    sheet.save(path, quality=92, subsampling=0)
    return digest(path)


def run(out):
    start = time.monotonic()
    docs = ROOT / 'docs/research'
    frozen = checked_json(PARENT / 'results.json', PARENT_SHA)
    toggle = checked_json(docs / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    pair = docs / '2026-11-18-psbench-action-pair-evidence.json'
    if digest(pair) != PSBENCH_SHA or toggle['parent_sha256'] != PARENT_SHA or frozen['strength'] != .8:
        raise ValueError('Unexpected control identity')
    out.mkdir()
    summary, records = {}, {}
    for name, preserve in (('original_chroma', False), ('retain_saturation', True)):
        probe = ChromaProbe(preserve)
        sample = np.array([[[128, 127, 126], [91, 117, 152]]], dtype=np.uint8)
        same, _ = probe.measure(sample, sample.copy(), 255 - sample)
        np.testing.assert_array_equal(same, sample)
        target = out / name
        for subfolder in ('background', 'local', 'authored'):
            (target / subfolder).mkdir(parents=True)
        bg = background_controls(target / 'background', frozen, probe.measure)
        local = local_controls(probe, target / 'local', toggle, {row['pair']: row for row in frozen['cases']})
        real = authored(target / 'authored', BASE / 'psbench-shared-local-v1', pair, probe.measure)
        records[name] = dict(background=bg, local=local, authored=real)
        summary[name] = summarize(bg, local, real)
    panel_hash = panel(out)
    if time.monotonic() - start > 120:
        raise RuntimeError('Exceeded compute budget')
    path = Path(__file__)
    report = dict(summary=summary, records=records, comparison_sha256=panel_hash,
                  background_sha256=PARENT_SHA, local_sha256=PARENT_REPORT, authored_sha256=PSBENCH_SHA,
                  source_sha256=digest(path), protocol_sha256=digest(path.with_name('target-hue-retention-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
