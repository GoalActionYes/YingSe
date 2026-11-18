"""Measure the minimum error with target hue and authored luma held fixed."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from background_intervention import target_regions
from probe_facd_two_image_baseline import STYLES, digest, read_checked
from probe_sparks_shared_chroma import LUMA
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


def fit_pixels(original, author):
    pole = (author @ LUMA).clip(0, 255)[..., None]
    ray = original - (original @ LUMA)[..., None]
    ray[np.ptp(original, axis=-1) == 0] = 0.
    lower = np.divide(-pole, ray, out=np.full_like(ray, np.inf), where=ray < 0)
    upper = np.divide(255 - pole, ray, out=np.full_like(ray, np.inf), where=ray > 0)
    limit = np.minimum(lower, upper).min(axis=-1)
    limit = np.where(np.isfinite(limit), limit, 0.)
    desired = author - pole
    bends = np.divide(desired, ray, out=np.zeros_like(ray), where=ray != 0).clip(0., limit[..., None])
    factors = np.concatenate((bends, np.zeros_like(pole), limit[..., None]), axis=-1)
    # Convex piecewise-linear RGB error reaches its minimum at a bend or endpoint.
    candidates = pole[..., None, :] + factors[..., None] * ray[..., None, :]
    error = np.abs(candidates - author[..., None, :]).mean(-1)
    best = np.take_along_axis(candidates, error.argmin(-1)[..., None, None], axis=-2)[..., 0, :]
    anchor = pole + np.minimum(1., limit)[..., None] * ray
    return best, anchor


def report_regions(target, author, optimal, outputs):
    masks = target_regions(target, np.zeros(target.shape[:2], dtype=bool))
    result = {}
    for name in ('whole', 'neutral_proxy'):
        mask = masks[name]
        count = int(mask.sum())
        if not count:
            result[name] = dict(pixels=count)
            continue
        continuous = float(np.abs(optimal[mask] - author[mask]).mean())
        result[name] = dict(pixels=count, continuous_min=continuous,
                            rounding_lower_bound=max(continuous - .5, 0.),
                            rgb_mae={role: float(np.abs(rgb[mask].astype(np.int16) - author[mask]).mean())
                                     for role, rgb in outputs.items()})
    return result


def comparison(rows, out):
    tile = 220
    sheet = Image.new('RGB', (6 * (tile + 8) + 8, 3 * (tile + 34) + 8), '#191c20')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    labels = ('Original', 'Authored luma only', 'Best fixed hue', 'YingSe 13 ref A', 'YingSe 13 ref B', 'Author')
    for row, style in enumerate(STYLES):
        for col, (rgb, label) in enumerate(zip(rows[style], labels, strict=True)):
            x, y = 8 + col * (tile + 8), 8 + row * (tile + 34)
            draw.text((x, y), f'{style} / {label}', font=font, fill='#edf1f5')
            thumb = Image.fromarray(rgb)
            thumb.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x, y + 24))
    sheet.save(out, quality=92, subsampling=0)


def run(aligned, aligned_evidence, baseline_dir, baseline_evidence, out):
    start = time.monotonic()
    verified = json.loads(aligned_evidence.read_bytes())
    frozen = json.loads(baseline_evidence.read_bytes())
    if digest(aligned_evidence) != frozen['data_sha256'] or len(frozen['references']) != 7:
        raise ValueError('Unexpected FACD control groups')
    if digest(baseline_dir / 'results.json') != frozen['repeat_results_sha256']:
        raise ValueError('Frozen baseline result changed')
    result = {}
    out.mkdir()
    for family, refs in frozen['references'].items():
        target = read_checked(aligned, family, 'Origin', verified)
        folder = out / family
        folder.mkdir()
        records, panels = {}, {}
        for style in STYLES:
            edit = read_checked(aligned, family, style, verified)
            author = finish_spatial(target, edit, .8)
            optimal, anchor = fit_pixels(target, author)
            if np.min(optimal) < -1e-9 or np.max(optimal) > 255 + 1e-9:
                raise ValueError('Fixed hue projection escaped the RGB cube')
            np.testing.assert_allclose(optimal @ LUMA, author @ LUMA, atol=1e-10, rtol=0)
            output = dict(original=target)
            hashes = {}
            for name, array in (('best', optimal), ('anchor', anchor)):
                rgb = np.rint(array.clip(0, 255)).astype(np.uint8)
                path = folder / f'{style}-{name}.png'
                Image.fromarray(rgb).save(path)
                hashes[name] = digest(path)
                output[name] = rgb
            for ref in refs:
                name = f'{family}/{ref}-{style}.png'
                path = baseline_dir / name
                if digest(path) != frozen['artifacts'][name]:
                    raise ValueError(f'Frozen baseline changed: {name}')
                output[ref] = decode_image(path)
            scores = report_regions(target, author, optimal, output)
            neutral = scores['neutral_proxy']
            records[style] = dict(regions=scores, output_sha256=hashes,
                                  exceeds=neutral['pixels'] >= 256 and neutral['continuous_min'] > 2)
            if family == '64573':
                panels[style] = (target, output['anchor'], output['best'],
                                 *(output[ref] for ref in refs), author)
        if family == '64573':
            comparison(panels, out / 'wedding.jpg')
        result[family] = records
    counts = {style: sum(row[style]['exceeds'] for row in result.values()) for style in STYLES}
    if time.monotonic() - start > 120:
        raise RuntimeError('Run exceeded the compute budget')
    source = Path(__file__)
    report = dict(cases=result, exceeds_neutral_count=counts,
                  color_change_required=[style for style in STYLES if counts[style] >= 4],
                  aligned_sha256=digest(aligned_evidence), baseline_sha256=digest(baseline_evidence),
                  source_sha256=digest(source), comparison_sha256=digest(out / 'wedding.jpg'),
                  protocol_sha256=digest(source.with_name('facd-fixed-hue-limit-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(counts, result['64573'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('aligned', 'aligned_evidence', 'baseline_dir', 'baseline_evidence', 'out'):
        parser.add_argument(arg, type=Path)
    run(**vars(parser.parse_args()))
