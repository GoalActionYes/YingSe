"""Reuse verified default-80 global/adaptive outputs; no inference or ratings."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from independent_coco import ROOT, digest


def main():
    base = ROOT / 'outputs/adaptive-reference-20261113'
    natural = base / 'tst100k/reserved-multiref-v1'
    sweep = base / 'natural-review/strength-sweep-20261115-v1'
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    source = load(sweep / 'results.json'); data = load(natural / 'results.json')
    assert source['finished'] and len(source['cases']) == len(data['cases']) == 32
    assert source['dataset_sha256'] == digest(natural / 'results.json')
    assert source['model_sha256'] == 'd8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'
    assert source['global_fingerprint'] == 'c1d65198cfa49ea4655d4eceb03421f6cb6c688131576cee4ce9c36f77aa2ee1'
    for name, sha in source['source_sha256'].items(): assert digest(sweep / 'frozen-source' / name) == sha
    frozen = (sweep / 'frozen-source/probe_default_strength.py').read_text(encoding='utf-8')
    assert 'baseline=analyze(target,ref)' in frozen and 'apply_mapping(source,baseline,strength)' in frozen
    indexed = {c['index']: c for c in data['cases']}; cases = []
    for c in source['cases']:
        row = indexed[c['index']]
        item = next(x for x in c['rows'] if x['strength'] == .8)
        target = natural / row['outputs']['input']['path']
        reference = natural / 'srgb' / (row['triplet']['reference'] + '.png')
        assert digest(target) == c['source_sha256'] == row['outputs']['input']['sha256']
        assert digest(reference) == c['reference_sha256']
        paths = {'input': target, 'reference': reference}
        for method in ['adaptive', 'global']:
            e = item['files'][method]; paths[method] = sweep / e['path']
            assert digest(paths[method]) == e['sha256']
        cases.append((c['index'], paths))
    out = base / 'natural-review/default80-global-vs-adaptive-20261115-v1'
    out.mkdir(exist_ok=False)
    rng = random.Random(20261130); rng.shuffle(cases)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16); key = []
    for page in range(8):
        canvas = Image.new('RGB', (1024, 1168), '#22252a'); draw = ImageDraw.Draw(canvas)
        for pos, (index, paths) in enumerate(cases[page * 4:page * 4 + 4]):
            methods = ['adaptive', 'global']; rng.shuffle(methods); alias = f'C{page * 4 + pos + 1:02d}'
            for col, method in enumerate(['input', 'reference'] + methods):
                draw.text((col * 256 + 8, pos * 292 + 8), alias + ' ' + ['Source', 'Reference', 'A', 'B'][col], font=font, fill='white')
                with Image.open(paths[method]) as im: tile = ImageOps.contain(im.convert('RGB'), (256, 256), Image.Resampling.LANCZOS)
                canvas.paste(tile, (col * 256 + (256 - tile.width) // 2, pos * 292 + 32 + (256 - tile.height) // 2))
            key.append(dict(alias=alias, index=index, A=methods[0], B=methods[1], files={m: dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for m, p in paths.items()}))
        canvas.save(out / f'sheet-{page + 1}.png')
    (out / 'private-key.json').write_text(json.dumps(key, indent=2), encoding='utf-8')
    report = dict(scope='Consumed natural32, actual default80 global/adaptive outputs; not independent acceptance or ratings',
                  strength=.8, baseline_role='legacy_global_pipeline', candidate_role='multiref_epoch6',
                  sweep_sha256=digest(sweep / 'results.json'), dataset_sha256=digest(natural / 'results.json'),
                  generator_sha256=digest(Path(__file__)), model_sha256=source['model_sha256'],
                  global_fingerprint=source['global_fingerprint'], cases=32, checked_input_output_files=128,
                  human_ratings=0, ai_ratings=0, sheets={p.name: digest(p) for p in out.glob('sheet-*.png')})
    (out / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
