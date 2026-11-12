"""Summarize every development pairing without selecting successful outputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
directory = ROOT / 'outputs/adaptive-reference-20261113/historical-development'
report = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
rows = report['cases']
natural = [r for r in rows if not r['identity_checked']]
identity = [r for r in rows if r['identity_checked']]
summary = dict(total=len(rows), historical_development_photos=len(report['historical_sources']),
               identity_cases=len(identity), identity_unchanged=sum(r['mean_absolute_extra_edit']==0 for r in identity),
               cross_content_pairs=len(natural),
               active_cross_content_pairs=sum(r['mean_absolute_extra_edit']>0 for r in natural),
               extra_edit_mae_median=float(np.median([r['mean_absolute_extra_edit'] for r in natural])),
               segmentation_seconds_p50_p95=np.percentile([d['seconds'] for d in report['segmentation']], [50,95]).tolist(),
               residual_seconds_p50_p95=np.percentile([r['fit_apply_seconds'] for r in rows], [50,95]).tolist(),
               quality_verdict='Unproven. Edit magnitude is not improvement. No human preference scores collected.',
               performance_scope='Images <=640px; excludes global analysis, display, decode and encoding; not video fps')
(directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
sheet = Image.new('RGB', (1920, 10*194), '#202020')
draw = ImageDraw.Draw(sheet)
for row in range(10):
    for column, step in enumerate((1,3,5)):
        reference = (row+step) % 10
        image = Image.open(directory / f'{row}-{reference}-comparison.png').resize((640,176))
        sheet.paste(image, (column*640, row*194+18))
        draw.text((column*640+5, row*194), f'{row} -> {reference}', fill='white')
sheet.save(directory / 'all-cross-content.jpg')
print(json.dumps(summary, indent=2))
