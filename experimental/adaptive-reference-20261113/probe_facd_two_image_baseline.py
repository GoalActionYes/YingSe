"""Render fixed same-filter reference swaps on distinct FACD photo groups."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import target_regions
from independent_coco import digest
from probe_facd_range import FILTERS
from probe_psbench_action_rgb_transfer import score
from probe_reference_chroma_spread import projection
from reference_color_app.media import decode_image
from spatial_render import SpatialEngine, finish_spatial


STYLES = FILTERS[1:]
CHECKPOINT = 'd8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'


def read_checked(folder, family, role, verified):
    name = 'Origin.png' if role == 'Origin' else f'{role}-overlay.png'
    path = folder / family / name
    if digest(path) != verified['cases'][family]['artifacts'][name]:
        raise ValueError(f'Frozen aligned input changed: {family}/{role}')
    return decode_image(path)


def run(folder, evidence, out):
    verified = json.loads(evidence.read_text(encoding='utf-8'))
    groups = [name for name, case in verified['cases'].items()
              if all(row['overlay']['passed'] for row in case['checks'].values())]
    exclusions = [name for name in verified['cases'] if name not in groups]
    if len(groups) != 7 or exclusions != ['488102']:
        raise ValueError('Unexpected FACD development group assignment')
    refs = {target: [groups[(i - 1) % len(groups)], groups[(i + 1) % len(groups)]]
            for i, target in enumerate(groups)}
    engine = SpatialEngine('multiref-v1')
    if engine.checkpoint_sha != CHECKPOINT:
        raise ValueError('Unexpected 13 model')
    out.mkdir(exist_ok=False)
    (out / 'assignments.json').write_text(json.dumps(refs, indent=2), encoding='utf-8')
    artifacts = {}
    for group, references in refs.items():
        target = read_checked(folder, group, 'Origin', verified)
        (out / group).mkdir()
        for reference in references:
            for style in STYLES:
                look = read_checked(folder, reference, style, verified)
                prepared = engine.prepare(target, look)
                rgb = finish_spatial(target, prepared.apply(target), .8)
                path = out / group / f'{reference}-{style}.png'
                Image.fromarray(rgb).save(path)
                artifacts[path.relative_to(out).as_posix()] = digest(path)
    (out / 'predictions.json').write_text(json.dumps(artifacts, indent=2), encoding='utf-8')

    report = {}
    for group, references in refs.items():
        origin = read_checked(folder, group, 'Origin', verified)
        masks = target_regions(origin, np.zeros(origin.shape[:2], dtype=bool))
        masks = {name: masks[name] for name in ('whole', 'neutral_proxy')}
        variants = {}
        authored = {}
        for style in STYLES:
            edit = read_checked(folder, group, style, verified)
            authored[style] = finish_spatial(origin, edit, .8).astype(np.int16)
            variants[style] = {ref: decode_image(out / group / f'{ref}-{style}.png').astype(np.int16)
                               for ref in references}
        same_style, distinguish, errors, separations = {}, {}, {}, {}
        for style, renders in variants.items():
            a, b = renders.values()
            same_style[style] = {name: float(np.abs(a[mask] - b[mask]).mean()) if mask.any() else None
                                 for name, mask in masks.items()}
            errors[style] = score(origin, authored[style], {'original': origin, **renders}, masks['whole'])
        for a, b in combinations(STYLES, 2):
            key = f'{a}-{b}'
            delta = authored[b] - authored[a]
            separations[key] = float(np.abs(delta).mean())
            distinguish[key] = {ref: {name: projection(variants[b][ref] - variants[a][ref], delta, mask)
                                      for name, mask in masks.items()} for ref in references}
        swap_limit = max(2., .5 * min(separations.values()))
        stable = all(row['whole'] <= swap_limit for row in same_style.values())
        responds = all((metric['whole'] or 0.) >= .5 for row in distinguish.values() for metric in row.values())
        whole_errors = [regions['whole']['methods'] for regions in errors.values()]
        fidelity = all(methods[ref]['mae'] <= methods['original']['mae'] - 1
                       for methods in whole_errors for ref in references)
        report[group] = dict(same_filter_swaps=same_style, author_style_differences=separations,
                             style_projections=distinguish, fidelity=errors, swap_limit=swap_limit,
                             gates=dict(stable=stable, responds=responds, fidelity=fidelity))
    count = sum(all(item['gates'].values()) for item in report.values())
    result = dict(count=count, eligible=count >= 6, references=refs, exclusions=exclusions, cases=report,
                  artifacts=artifacts, checkpoint_sha256=CHECKPOINT, data_sha256=digest(evidence),
                  script_sha256=digest(__file__), protocol_sha256=digest(Path(__file__).with_name(
                      'facd-two-image-baseline-protocol.md')))
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(qualified=count, gates={name: item['gates'] for name, item in report.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('folder', 'evidence', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
