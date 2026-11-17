"""Compare same-result reference swaps and different-result swaps on prepared training assets."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from audit_tst_reference_equivalence import COHORT, audit
from background_intervention import target_regions
from independent_coco import digest
from probe_reference_chroma_spread import projection
from spatial_render import SpatialEngine, finish_spatial


@torch.inference_mode()
def run(out, split_dir=None, engine=None):
    metadata, cases, ready = audit()
    assert metadata['proceed'] and len(cases) == 8
    if split_dir is not None:
        split = json.loads((split_dir / 'results.json').read_text(encoding='utf-8'))
        path = split_dir / 'private-assignment.json'
        assert split['readiness'] and digest(path) == split['assignment_sha256']
        cases = json.loads(path.read_text(encoding='utf-8'))['development']
        assert len(cases) == split['development_cases']
        metadata = split
    out.mkdir(exist_ok=False)
    (out / 'private-selection.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')

    def load(path):
        entry = ready[path]
        prepared = COHORT.parent / entry['prepared_path']
        assert digest(prepared) == entry['sha256']
        with Image.open(prepared) as im:
            return np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)).copy()

    if engine is None:
        engine = SpatialEngine('multiref-v1')
    torch.set_num_threads(4)
    start = time.monotonic()
    predictions = {}
    for index, row in enumerate(cases):
        target = load(row['content'])
        for slot, ref_path in enumerate(row['refs']):
            prepared = engine.prepare(target, load(ref_path))
            graded = finish_spatial(target, prepared.apply(target), .8)
            path = out / f'{index}-{slot}.png'
            Image.fromarray(graded).save(path)
            predictions[path.name] = digest(path)
            assert time.monotonic() - start <= 90
    prediction_path = out / 'predictions.json'
    prediction_path.write_text(json.dumps(predictions, indent=2), encoding='utf-8')

    def image(path):
        with Image.open(path) as value:
            return np.asarray(value, dtype=np.int16)

    scored = []
    for index, row in enumerate(cases):
        target = load(row['content'])
        truth = [finish_spatial(target, load(row[k]), .8).astype(np.int16) for k in ('same_truth', 'different_truth')]
        delta = truth[1] - truth[0]
        a, b, c = (image(out / f'{index}-{slot}.png') for slot in range(3))
        output_errors = (a - truth[0], b - truth[0], c - truth[1])
        changes = (c - a, c - b)
        masks = target_regions(target, np.zeros(target.shape[:2], dtype=bool))
        scores = {}
        for region in ('whole', 'neutral_proxy'):
            mask = masks[region]
            assert mask.any()
            scores[region] = dict(expected_delta=float(np.abs(delta[mask]).mean()),
                                  same_swap=float(np.abs((a - b)[mask]).mean()),
                                  different_swap=[float(np.abs(d[mask]).mean()) for d in changes],
                                  projection=projection((changes[0] + changes[1]) / 2, delta, mask),
                                  errors=[float(np.abs(error[mask]).mean()) for error in output_errors])
        whole = scores['whole']
        passed = whole['expected_delta'] >= 1 and whole['same_swap'] < min(whole['different_swap']) and whole['projection'] >= .50
        scored.append(dict(index=index, passed=passed, regions=scores))
    count = sum(row['passed'] for row in scored)
    protocol = 'tst-reference-equivalence-pilot-protocol.md' if split_dir is None else 'tst-reference-equivalence-split-protocol.md'
    report = dict(metadata=metadata, cases=scored, passed_cases=count, passed=count * 4 >= len(cases) * 3,
                  checkpoint_sha256=engine.checkpoint_sha, predictions=predictions,
                  prediction_manifest_sha256=digest(prediction_path),
                  script_sha256=digest(__file__), protocol_sha256=digest(Path(__file__).with_name(protocol)),
                  seconds=time.monotonic() - start)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(count, report['seconds'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--split-dir', type=Path)
    args = parser.parse_args()
    run(args.out, args.split_dir)
