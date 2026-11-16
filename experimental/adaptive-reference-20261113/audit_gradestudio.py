"""Re-read every PNG and original GT; compare baseline to earlier run."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from independent_coco import ROOT, digest


def main():
    base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
    out = base / 'gradestudio-screen-v1'
    r = json.loads((out / 'results.json').read_text())
    assert r['finished'] and len(r['rows']) == 64 and r['first_exact_repeat']
    for e in r['author_source']['files']:
        assert digest(ROOT / '.tools/GradeStudio-source' / e['path']) == e['sha256']
    for name, sha in r['source_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == sha
    data = base / 'cohort-multiref-v1'
    cohort = json.loads((data / 'cohort.json').read_text())
    lookup = {e['original_path']: e for e in json.loads((data / 'srgb-v1/manifest.json').read_text())['files']}
    rows = {e['index']: e for e in cohort['groups']['validation']}
    old = base / 'semantic-baseline-context-v1/source_control/epoch-0'
    errors = {a: [] for a in ['baseline', 'global', 'semantic']}
    cases = []
    for row in r['rows']:
        e = lookup[rows[row['index']]['gt']]
        p = data / e['prepared_path']
        assert digest(p) == e['sha256']
        with Image.open(p) as im:
            gt = np.array(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)).astype(float)
        result = dict(index=row['index'], outputs={})
        for arm in errors:
            artifact = row['outputs'][arm]
            p = out / artifact['path']
            assert digest(p) == artifact['sha256']
            with Image.open(p) as im:
                a = np.array(im)
            assert a.shape == (256, 256, 3) and a.dtype == np.uint8
            err = float(np.abs(a.astype(float) - gt).mean())
            assert err == artifact['error']
            if arm == 'baseline':
                with Image.open(old / f"{row['index']}.png") as im:
                    assert np.array_equal(a, np.array(im))
            errors[arm].append(err)
            result['outputs'][arm] = dict(error=err, png_sha256=artifact['sha256'])
        cases.append(result)
    errors = {a: np.array(v) for a, v in errors.items()}
    b = errors['baseline']
    for arm in ['global', 'semantic']:
        e = errors[arm]
        assert r['summary'][arm]['mean'] == float(e.mean())
        assert r['summary'][arm]['improved_over_point1'] == int((b - e > .1).sum())
        assert r['summary'][arm]['severe'] == int(((e - b > 5) & (e > b * 1.25)).sum())
        assert r['summary'][arm]['relative_improvement'] == float(1 - e.mean() / b.mean())
    evidence = dict(scope=r['scope'], results_sha256=digest(out / 'results.json'),
                    source_sha256=r['source_sha256'], revision=r['author_source']['revision'],
                    author_file_hashes=r['author_source']['files'],
                    descriptor=r['descriptor'], baseline_sha256=r['baseline_sha256'],
                    summary=r['summary'], checked_png=192, prior_baseline_exact=64,
                    first_exact_repeat=True, original_mean=float(np.mean([e['original_error'] for e in r['rows']])),
                    semantic_versus_global_improvement=float(1 - errors['semantic'].mean() / errors['global'].mean()),
                    semantic_median_seconds=float(np.median([e['seconds']['semantic'] for e in r['rows']])),
                    cases=cases, product_integrated=False)
    p = ROOT / 'docs/research/2026-11-17-gradestudio-evidence.json'
    p.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in evidence.items() if k not in ['cases', 'author_file_hashes', 'descriptor']}, indent=2))


if __name__ == '__main__':
    main()
