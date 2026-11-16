"""All consumed development pairs, anonymous diagnostics only, no new inference."""
import json
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from independent_coco import ROOT, digest


def main():
    base = ROOT / 'outputs/adaptive-reference-20261113/tst100k'
    data = base / 'cohort-training-coverage-v2'
    cohort_path = data / 'cohort.json'
    assert digest(cohort_path) == 'ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077'
    cohort = json.loads(cohort_path.read_text())
    prior = base / 'appearance-pairwise-replica-v1/baseline-current'
    run = base / 'training-coverage-expanded-v2'
    record = json.loads((run / 'results.json').read_text())
    assert record['finished'] and record['best_epoch'] == 4
    checkpoint = run / 'epoch-4.pth'
    assert digest(checkpoint) == record['selected_checkpoint_sha256'] == 'cde15ae2d5295de709b14c8a4fe6e72e3cd7343fa5766d6ac30a9c8858db3ee1'
    selected = run / 'epoch-4'
    scorepaths = {'baseline': prior / 'scores.json', 'candidate': selected / 'scores.json'}
    scores = {k: {r['index']: r for r in json.loads(p.read_text())['cases']} for k, p in scorepaths.items()}
    manifest_path = data / 'srgb-v1/manifest.json'
    assert digest(manifest_path) == record['prepared_manifest_sha256']
    lookup = {e['original_path']: e for e in json.loads(manifest_path.read_text())['files']}
    out = ROOT / 'outputs/adaptive-reference-20261113/human-diagnostic-20261117-v2'
    out.mkdir(exist_ok=False)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    rng = random.Random(20261130)
    report = dict(scope='Consumed development diagnostic; failed promotion unchanged; not independent acceptance',
                  cohort_sha256=digest(cohort_path), selected_checkpoint_sha256=digest(checkpoint),
                  scores_sha256={k: digest(p) for k, p in scorepaths.items()},
                  source_sha256={n: digest(Path(__file__).with_name(n)) for n in ['build_coverage_human_review.py', 'build_human_review.py', 'coverage-human-diagnostic-protocol.md']},
                  pages=[], human_reviewers=0, human_ratings=0, product_integrated=False)
    seen = set()
    for style in (0, 1):
        folder = out / f'part-{style + 1}'
        folder.mkdir()
        rows = [r for r in cohort['groups']['validation'] if r['style_index'] == style]
        assert len(rows) == 32 and len({r['group_id'] for r in rows}) == 32
        rng.shuffle(rows)
        key = []
        for batch in range(8):
            canvas = Image.new('RGB', (1024, 1168), '#22252a')
            draw = ImageDraw.Draw(canvas)
            for position, row in enumerate(rows[4 * batch:4 * batch + 4]):
                assert row['index'] not in seen
                seen.add(row['index'])
                alias = f'C{batch * 4 + position + 1:02d}'
                methods = ['baseline', 'candidate']
                rng.shuffle(methods)
                assets = []
                for role in ['content', 'reference']:
                    e = lookup[row[role]]
                    assets.append((data / e['prepared_path'], e['sha256']))
                for method in methods:
                    parent = prior if method == 'baseline' else selected
                    assets.append((parent / f"{row['index']}.png", scores[method][row['index']]['sha256']))
                for column, ((path, expected), label) in enumerate(zip(assets, ['Source', 'Reference', 'A', 'B'])):
                    assert digest(path) == expected
                    with Image.open(path) as im:
                        rgb = im.convert('RGB')
                        # Show exactly the working geometry used by these saved predictions.
                        if column < 2:
                            rgb = rgb.resize((256, 256), Image.Resampling.BILINEAR)
                        tile = ImageOps.contain(rgb, (256, 256), Image.Resampling.LANCZOS)
                        if column >= 2:
                            assert im.size == (256, 256)
                        canvas.paste(tile, (column * 256 + (256 - tile.width) // 2, position * 292 + 32 + (256 - tile.height) // 2))
                    draw.text((column * 256 + 8, position * 292 + 8), alias + ' ' + label, fill='white', font=font)
                key.append(dict(alias=alias, index=row['index'], A=methods[0], B=methods[1], input_sha256=[sha for _, sha in assets]))
            canvas.save(folder / f'sheet-{batch + 1}.png')
        (folder / 'private-key.json').write_text(json.dumps(key, indent=2))
        manifest = dict(scope=report['scope'], part=style + 1, cases=32,
                        method_roles=dict(baseline='current13_multiref_epoch6', candidate='failed_expanded4096_epoch4'),
                        source_sha256=report['source_sha256'], scores_sha256=report['scores_sha256'],
                        sheets={p.name: digest(p) for p in folder.glob('sheet-*.png')})
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        subprocess.run([sys.executable, str(Path(__file__).with_name('build_human_review.py')), str(folder)], check=True)
        page = folder / '匿名评价.html'
        html = page.read_text(encoding='utf-8')
        assert all(s not in html for s in ['current13_multiref_epoch6', 'failed_expanded4096_epoch4', 'private-key', 'checkpoint'])
        report['pages'].append(dict(part=style + 1, cases=32, material_sha256=digest(folder / 'manifest.json'),
                                   html_sha256=digest(page), key_sha256=digest(folder / 'private-key.json'),
                                   indices=[r['index'] for r in rows]))
    assert seen == {r['index'] for r in cohort['groups']['validation']} and len(seen) == 64
    report['finished'] = True
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(cases=len(seen), pages=2, human_ratings=0)))


if __name__ == '__main__':
    main()
