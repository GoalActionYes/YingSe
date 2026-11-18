"""Check identical RGB8 inputs in the two scenes from the PSBench 105 tutorial."""
import argparse
import json
from pathlib import Path

import numpy as np

from probe_psbench_teal_portrait import verify_source
from probe_s24_exact_rgb import compare, decode_image, digest, finish_spatial, per_half


def fixed_transform(source):
    # A global edit must render every occurrence of an RGB code identically.
    return np.clip(source.astype(np.int16) + np.array([12, -5, -18]), 0, 255).astype(np.uint8)


def run(folder, evidence, out):
    prior = json.loads(evidence.read_bytes())
    files = verify_source(folder, '105')
    assert files == prior['files']
    assert all(row['passed'] for row in prior['geometry'].values())
    source = decode_image(folder / '105_start.jpg')
    grade = decode_image(folder / '105_end.jpg')
    assert source.shape == grade.shape == (1080, 1440, 3)
    summaries, comparisons, totals = {}, {}, {}
    for name, (left, right) in prior['crop_columns'].items():
        frame = source[:, left:right]
        summaries[name], totals[name] = per_half(frame, {
            'authored': finish_spatial(frame, grade[:, left:right], .8),
            'fixed': finish_spatial(frame, fixed_transform(frame), .8),
        })
    for mode in ('authored', 'fixed'):
        comparisons[mode] = {
            target: compare(summaries[donor], summaries[target], mode, totals[target])
            for donor, target in (('portrait', 'tower'), ('tower', 'portrait'))
        }
    status = {}
    for region in ('whole', 'neutral_proxy'):
        authored = [row[region] for row in comparisons['authored'].values()]
        fixed = [row[region] for row in comparisons['fixed'].values()]
        assert all(row.get('cross', 0.) == 0. and row.get('cross_excess', 0.) == 0. for row in fixed)
        adequate = all(row['colors'] >= 128 and row['target_fraction'] >= .10 for row in authored)
        status[region] = (
            'insufficient_coverage' if not adequate else
            'cross_excess' if authored[0]['cross_excess'] >= 2 else 'below_threshold'
        )
    report = dict(files=files, prior_sha256=digest(evidence), source_sha256=digest(Path(__file__)),
                  comparisons=comparisons, status=status)
    out.mkdir()
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(comparisons=comparisons['authored'], status=status)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('folder', 'evidence', 'out'):
        parser.add_argument(argument, type=Path)
    run(**vars(parser.parse_args()))
