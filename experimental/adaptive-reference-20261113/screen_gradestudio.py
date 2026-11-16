"""Pinned author analytic transfer screening; local RGB inputs, no fitting or app changes."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dense_anchor_model import DenseAnchorModel
from independent_coco import ROOT, digest
from spatial_render import SpatialEngine


def pixels(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = ROOT / '.tools/GradeStudio-source'
    provenance = json.loads((source / 'source-audit.json').read_text())
    assert provenance['revision'] == '5c10ccb685be7b58cdd88ec2f5e78a8975d4139e'
    for e in provenance['files']:
        assert digest(source / e['path']) == e['sha256']
    sys.path.insert(0, str(source / 'src'))
    import grademl.semantic_match as author
    from grademl.lut import apply_lut, identity_lut
    from grademl.transfer import transfer_lut

    base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
    registry_path = base / 'regional-correction-inputs-v1/results.json'
    registry = json.loads(registry_path.read_text())
    rows = [x for x in registry['rows'] if x['split'] == 'validation']
    assert registry['finished'] and len(rows) == 64
    inputs = {x['index']: x for x in registry['inputs']}
    data = base / 'cohort-multiref-v1'
    assert digest(data / 'cohort.json') == registry['cohort_sha256']
    cohort = json.loads((data / 'cohort.json').read_text())
    manifest_path = data / 'srgb-v1/manifest.json'
    assert digest(manifest_path) == registry['prepared_manifest_sha256']
    lookup = {e['original_path']: e for e in json.loads(manifest_path.read_text())['files']}
    out = base / 'gradestudio-screen-v1'
    out.mkdir(exist_ok=False)
    names = ['screen_gradestudio.py', 'gradestudio-screen-protocol.md']
    for n in names:
        (out / n).write_bytes(Path(__file__).with_name(n).read_bytes())
    descriptor = DenseAnchorModel()
    engine = SpatialEngine('multiref-v1')
    # Author preprocessing and formula are unchanged; only prevent model downloading.
    author._MODEL[engine.device] = descriptor.model
    report = dict(scope='Consumed development64, full strength; not independent acceptance',
                  author_source=provenance, descriptor=descriptor.provenance,
                  registry_sha256=digest(registry_path), baseline_sha256=engine.checkpoint_sha,
                  source_sha256={n: digest(out / n) for n in names},
                  rows=[], finished=False, product_integrated=False, training=False)
    (out / 'frozen-config.json').write_text(json.dumps(report, indent=2))

    def read(path, expected):
        assert digest(path) == expected
        with Image.open(path) as im:
            return np.array(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))

    def quant(t):
        assert t.shape == (3, 256, 256) and torch.isfinite(t).all()
        return np.rint(t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    for i, row in enumerate(rows):
        arrays = []
        for role in ['content', 'reference']:
            entry = inputs[row['roles'][role]['feature_index']]
            a = read(ROOT / entry['path'], entry['parent_sha256'])
            assert pixels(a) == entry['pixel_sha256']
            arrays.append(a)
        x, y = [torch.from_numpy(a).to(engine.device).permute(2, 0, 1).float() / 255 for a in arrays]
        assert torch.allclose(apply_lut(x[None], identity_lut(device=engine.device))[0], x, atol=1e-6)
        outputs = dict(baseline=engine.prepare(*arrays).byte_output(arrays[0]))
        seconds = {}
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs['global'] = quant(apply_lut(x[None], transfer_lut(x[None], y[None]))[0])
            torch.cuda.synchronize()
            seconds['global'] = time.perf_counter() - start
            start = time.perf_counter()
            match = author.semantic_match(x, y)
            outputs['semantic'] = quant(match.render(x))
            torch.cuda.synchronize()
            seconds['semantic'] = time.perf_counter() - start
            assert match.weights.shape == (6, 256, 256)
            assert torch.isfinite(match.weights).all() and (match.weights >= 0).all()
            assert torch.allclose(match.weights.sum(0), torch.ones_like(x[0]), atol=1e-5)
            assert torch.isfinite(match.luts).all() and torch.isfinite(match.confidence).all()
            if i == 0:
                repeated = author.semantic_match(x, y)
                assert np.array_equal(quant(repeated.render(x)), outputs['semantic'])
                report['first_exact_repeat'] = True
        entry = dict(index=row['index'], seconds=seconds,
                     confidence=match.confidence.cpu().tolist(), coverage=match.coverage.cpu().tolist(), outputs={})
        for arm, a in outputs.items():
            folder = out / arm
            folder.mkdir(exist_ok=True)
            p = folder / f"{row['index']}.png"
            Image.fromarray(a).save(p)
            entry['outputs'][arm] = dict(path=str(p.relative_to(out)), sha256=digest(p), pixel_sha256=pixels(a))
        report['rows'].append(entry)
        (out / 'progress.json').write_text(json.dumps(report, indent=2))
        if (i + 1) % 8 == 0:
            print(json.dumps(dict(predictions=i + 1)), flush=True)
    # GT is accessed only after every candidate image has been persisted.
    truths = {r['index']: r for r in cohort['groups']['validation']}
    for entry in report['rows']:
        gt_entry = lookup[truths[entry['index']]['gt']]
        gt = read(data / gt_entry['prepared_path'], gt_entry['sha256'])
        original_row = next(r for r in rows if r['index'] == entry['index'])
        source_entry = inputs[original_row['roles']['content']['feature_index']]
        original = read(ROOT / source_entry['path'], source_entry['parent_sha256'])
        entry['original_error'] = float(np.abs(original.astype(float) - gt).mean())
        for arm, artifact in entry['outputs'].items():
            assert digest(out / artifact['path']) == artifact['sha256']
            with Image.open(out / artifact['path']) as im:
                a = np.array(im)
            assert pixels(a) == artifact['pixel_sha256']
            artifact['error'] = float(np.abs(a.astype(float) - gt).mean())
    b = np.array([r['outputs']['baseline']['error'] for r in report['rows']])
    summary = {}
    for arm in ['global', 'semantic']:
        e = np.array([r['outputs'][arm]['error'] for r in report['rows']])
        severe = (e - b > 5) & (e > b * 1.25)
        summary[arm] = dict(mean=float(e.mean()), baseline_mean=float(b.mean()),
                            relative_improvement=float(1 - e.mean() / b.mean()),
                            improved_over_point1=int((b - e > .1).sum()), severe=int(severe.sum()))
        s = summary[arm]
        s['first_gate_passed'] = bool(s['relative_improvement'] >= .05 and s['improved_over_point1'] >= 40 and s['severe'] <= 2)
    for n, sha in report['source_sha256'].items():
        assert digest(Path(__file__).with_name(n)) == sha
    report.update(summary=summary, finished=True)
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
