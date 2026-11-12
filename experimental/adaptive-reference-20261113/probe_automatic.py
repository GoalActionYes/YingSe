"""Automatic local integration/visual probe. No subjective quality score is fabricated."""
import json
import argparse
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reference_color_app.pipeline import analyze
from reference_color_app.lut import apply_mapping
from regional import fit, apply
from segment import Segmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--historical-dataset', type=Path)
    args = parser.parse_args()
    out = ROOT / 'outputs/adaptive-reference-20261113' / ('historical-development' if args.historical_dataset else 'automatic')
    out.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / 'experimental/skin-protection-20261019/tests/reference_color/fixtures/skin-official.jpg',
             ROOT / 'reference_color_app/demo/source.png',
             ROOT / 'reference_color_app/demo/reference.png']
    sources = []
    if args.historical_dataset:
        manifest = json.loads((args.historical_dataset / 'sources.json').read_text(encoding='utf-8'))
        # Fixed before this run: first ten non-obama entries, preserving manifest
        # order. Historical content is development only, not unseen holdout.
        sources = [s for s in manifest if s['file'] != 'obama.jpg'][:10]
        if len(sources) != 10:
            raise ValueError('Expected ten historical development sources')
        paths = [args.historical_dataset / 'assets' / s['file'] for s in sources]
        for path, entry in zip(paths, sources):
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Historical asset checksum mismatch')
    arrays = []
    for path in paths:
        image = Image.open(path).convert('RGB')
        image.thumbnail((640, 640), Image.Resampling.BILINEAR)
        arrays.append(np.array(image))
    init = time.perf_counter()
    segmenter = Segmenter()
    initialization = time.perf_counter() - init
    try:
        masks, diagnostics = zip(*(segmenter.predict(x) for x in arrays))
        rows = []
        pairs = ([(i, (i+step) % 10) for i in range(10) for step in (0, 1, 3, 5)]
                 if args.historical_dataset else [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)])
        for i, j in pairs:
            target, reference = arrays[i], arrays[j]
            targetfile, referencefile = out / f'target-{i}.png', out / f'reference-{j}.png'
            Image.fromarray(target).save(targetfile)
            Image.fromarray(reference).save(referencefile)
            baseline = apply_mapping(target, analyze(targetfile, referencefile), 1.)
            start = time.perf_counter()
            model = fit(baseline, reference, masks[i], masks[j])
            result = apply(baseline, masks[i], model)
            seconds = time.perf_counter()-start
            if i == j: assert np.array_equal(result, target)
            panel = Image.new('RGB', (4*320, 352), '#202020')
            draw = ImageDraw.Draw(panel)
            for k, (label, rgb) in enumerate(zip(['target', 'reference', 'global', 'candidate'],
                                                 [target, reference, baseline, result])):
                image = Image.fromarray(rgb)
                image.thumbnail((320, 320), Image.Resampling.BILINEAR)
                panel.paste(image, (k*320+(320-image.width)//2, 32))
                draw.text((k*320+10, 8), label, fill='white')
            panel.save(out / f'{i}-{j}-comparison.png')
            rows.append(dict(target=paths[i].name, reference=paths[j].name,
                             fit_apply_seconds=seconds, model=model,
                             mean_absolute_extra_edit=float(np.abs(result.astype(float)-baseline).mean()),
                             changed_fraction=float(np.any(result != baseline, axis=2).mean()),
                             identity_checked=i==j))
        report = dict(scope='Automatic integration probe only; existing fixtures/demo not an independent evaluation set',
                      no_quality_claim=True, initialization_seconds=initialization,
                      historical_sources=sources, segmentation=list(diagnostics), cases=rows)
        (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(dict(initialization_seconds=initialization,
                              segmentation_seconds=[x['seconds'] for x in diagnostics],
                              cases=[{k:v for k,v in r.items() if k!='model'} for r in rows]), indent=2))
    finally:
        segmenter.close()


if __name__ == '__main__': main()
