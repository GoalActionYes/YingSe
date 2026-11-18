"""Check whether reference color support accounts for paired transfer errors."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from probe_s24_donor_delta import decode_image, digest, finish_spatial, image, region_scores, target_regions


REGIONS = ('whole', 'neutral_proxy')


def color_cells(rgb):
    r, g, b = np.moveaxis(rgb.astype(np.uint16) // 16, -1, 0)
    return r * 256 + g * 16 + b


def covered(source, donor):
    counts = np.bincount(color_cells(donor[::8, ::8]).ravel(), minlength=4096)
    return counts[color_cells(source)] >= 8


def blend_on_coverage(paired, current, coverage):
    return np.where(coverage[..., None], paired, current)


def subset_scores(frames, desired, mask):
    count = int(mask.sum())
    expected = desired[mask].astype(np.int16)
    return dict(pixels=count, errors={
        name: float(np.abs(frame[mask].astype(np.int16) - expected).mean()) if count else None
        for name, frame in frames.items()
    })


def run(assets, evidence, previous, attribution, within, out):
    prior = json.loads(evidence.read_bytes())
    reconstruction = json.loads(attribution.read_bytes())
    assert digest(evidence) == reconstruction['previous_evidence_sha256']
    assert digest(assets / 'assets.json') == prior['assets_sha256']
    manifest = json.loads((assets / 'assets.json').read_bytes())
    out.mkdir()

    def load(scene, grade):
        return decode_image(image(assets, manifest, scene, f'srgb_images_style_{grade}'))

    cases, outputs = {}, {}
    for scene, record in prior['cases'].items():
        source = load(scene, 1)
        identity = []
        for slot, (donor, _) in enumerate(record['references']):
            coverage = covered(source, load(donor, 1))
            frames = {}
            for name in ('paired', 'current'):
                path = previous / f'{scene}-{name}-{slot}.png'
                assert digest(path) == prior['predictions'][scene][path.name]
                frames[name] = decode_image(path)
            guarded = blend_on_coverage(frames['paired'], frames['current'], coverage)
            path = out / f'{scene}-guarded-{slot}.png'
            Image.fromarray(guarded).save(path)
            outputs[path.name] = digest(path)
            identity.append(bool(np.array_equal(blend_on_coverage(source, source, coverage), source)))
        cases[scene] = dict(identity=all(identity), slots=[])

    (out / 'predictions.json').write_text(json.dumps(outputs, indent=2) + '\n', encoding='utf-8')
    for scene, record in prior['cases'].items():
        source = load(scene, 1)
        regions = target_regions(source, np.zeros(source.shape[:2], bool))
        desired = [finish_spatial(source, load(scene, grade), .8) for grade in (2, 3)]
        guarded = []
        for slot, (donor, grade) in enumerate(record['references']):
            coverage = covered(source, load(donor, 1))
            same = within / f'{scene}-self-{grade}.png'
            assert digest(same) == reconstruction['self_sha256'][same.name]
            frame = decode_image(out / f'{scene}-guarded-{slot}.png')
            guarded.append(frame)
            frames = {name: decode_image(previous / f'{scene}-{name}-{slot}.png')
                      for name in ('paired', 'current')}
            frames.update(guarded=frame, target_self=decode_image(same))
            held_out = np.ones(source.shape[:2], bool)
            held_out[::8, ::8] = False
            target = desired[grade - 2]
            scopes = {
                key: {
                    'covered': subset_scores(frames, target, regions[key] & held_out & coverage),
                    'uncovered': subset_scores(frames, target, regions[key] & held_out & ~coverage),
                } for key in REGIONS
            }
            cases[scene]['slots'].append(dict(donor=donor, grade=grade, scopes=scopes))
        for key in REGIONS:
            cases[scene][key] = region_scores(guarded, desired, regions[key])
        whole = cases[scene]['whole']
        cases[scene]['joint'] = (whole['expected_delta'] >= 1 and whole['same_swap'] < min(whole['different_swap'])
                                 and whole['projection'] is not None and whole['projection'] >= .5)

    aggregate = {}
    for key in REGIONS:
        scores = np.array([cases[scene][key]['errors'] for scene in cases])
        current = np.array([prior['cases'][scene][key]['current']['errors'] for scene in cases])
        unchanged = np.array([prior['cases'][scene][key]['unchanged']['errors'] for scene in cases])
        aggregate[key] = dict(guarded=float(scores.mean()), current=float(current.mean()),
                              unchanged=float(unchanged.mean()), worst_regression=float((scores-current).max()))
    joint = sum(row['joint'] for row in cases.values())
    passed = joint >= 3 and all(row['identity'] for row in cases.values()) and all(
        row['guarded'] <= .9 * row['current'] and row['guarded'] < row['unchanged']
        and row['worst_regression'] <= 2 for row in aggregate.values()
    )
    report = dict(aggregate=aggregate, joint=joint, passed=passed, cases=cases, predictions=outputs,
                  previous_sha256=digest(evidence), attribution_sha256=digest(attribution),
                  source_sha256=digest(Path(__file__)), excluded=prior['excluded'])
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(joint=joint, passed=passed, aggregate=aggregate)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('assets', 'evidence', 'previous', 'attribution', 'within', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
