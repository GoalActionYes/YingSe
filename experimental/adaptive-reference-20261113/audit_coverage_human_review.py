"""Check every displayed RGB tile against its original input or saved prediction."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from independent_coco import ROOT, digest


def main():
    root = ROOT / 'outputs/adaptive-reference-20261113/human-diagnostic-20261117-v2'
    base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
    data = base / 'cohort-training-coverage-v2'
    record = json.loads((root / 'results.json').read_text())
    cohort = json.loads((data / 'cohort.json').read_text())
    rows = {x['index']: x for x in cohort['groups']['validation']}
    training = json.loads((base / 'training-coverage-expanded-v2/results.json').read_text())
    baseline = json.loads((base / 'appearance-pairwise-replica-v1/baseline-current/scores.json').read_text())
    assert baseline == training['baseline'] and len(baseline['cases']) == 64
    assert training['best_epoch'] == 4 and training['selected_checkpoint_sha256'] == record['selected_checkpoint_sha256']
    lookup = {x['original_path']: x for x in json.loads((data / 'srgb-v1/manifest.json').read_text())['files']}
    for name, sha in record['source_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == sha
    seen, tiles = [], 0
    for part in (1, 2):
        folder = root / f'part-{part}'
        keys = json.loads((folder / 'private-key.json').read_text())
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert digest(folder / 'manifest.json') == record['pages'][part - 1]['material_sha256']
        assert digest(folder / '匿名评价.html') == record['pages'][part - 1]['html_sha256']
        for sheet in range(8):
            path = folder / f'sheet-{sheet + 1}.png'
            assert digest(path) == manifest['sheets'][path.name]
            with Image.open(path) as im:
                panel = np.array(im)
            for offset in range(4):
                key = keys[4 * sheet + offset]
                row = rows[key['index']]
                seen.append(row['index'])
                sources = [data / lookup[row[v]]['prepared_path'] for v in ['content', 'reference']]
                for label in ['A', 'B']:
                    assert key[label] in ['baseline', 'candidate']
                    parent = base / ('appearance-pairwise-replica-v1/baseline-current' if key[label] == 'baseline' else 'training-coverage-expanded-v2/epoch-4')
                    sources.append(parent / f"{key['index']}.png")
                assert key['A'] != key['B']
                for col, path in enumerate(sources):
                    assert digest(path) == key['input_sha256'][col]
                    with Image.open(path) as im:
                        image = im.convert('RGB')
                        if col < 2:
                            image = image.resize((256, 256), Image.Resampling.BILINEAR)
                        expected = np.array(image)
                    actual = panel[offset * 292 + 32:offset * 292 + 288, col * 256:col * 256 + 256]
                    assert np.array_equal(actual, expected), (part, sheet, offset, col)
                    tiles += 1
    assert len(seen) == len(set(seen)) == 64 and set(seen) == set(rows)
    browser = json.loads((root / 'browser-checks.json').read_text())
    assert browser['passed'] and browser['fresh_context_empty']
    evidence = {k: v for k, v in record.items() if k != 'pages'}
    evidence['pages'] = [{k: v for k, v in p.items() if k not in ['indices', 'key_sha256']} for p in record['pages']]
    evidence.update(tiles_pixel_exact=tiles, all64_present_once=True, browser=browser,
                    browser_report_sha256=digest(root / 'browser-checks.json'),
                    verification_source_sha256={n: digest(Path(__file__).with_name(n)) for n in ['audit_coverage_human_review.py', 'verify_coverage_human_review.cjs']},
                    initial_v1_not_delivered='Source/reference display aspect differed from saved model working geometry; preserved locally, replaced by v2 using exact bilinear256 input pixels')
    (ROOT / 'docs/research/2026-11-17-human-diagnostic-evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(dict(pixel_exact_tiles=tiles, unique_cases=64, browser_passed=True, human_reviews=0)))


if __name__ == '__main__':
    main()
