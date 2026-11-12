import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
directory=ROOT/'outputs/adaptive-reference-20261113'
report=json.loads((directory/'canonical-model/results.json').read_text(encoding='utf-8'))
rows=report['cases']
timings=[r['forward_seconds'] for r in rows]
summary=dict(summary=report['summary'],by_style=report['by_style'],
             forward_seconds_p50_p95=np.percentile(timings,[50,95]).tolist(),
             load_seconds=report['load_seconds'],
             first_process_cold_load_seconds=json.loads((directory/'canonical-model/smoke-results.json').read_text(encoding='utf-8'))['load_seconds'],
             scope=report['scope'])
(directory/'canonical-model/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
sheet=Image.new('RGB',(1280,4*176),'#202020')
for i in range(8):
    image=Image.open(directory/f'author-samples/{i:02d}-comparison.png').resize((640,176))
    sheet.paste(image,((i%2)*640,(i//2)*176))
sheet.save(directory/'author-samples/all-eight.jpg')
print(json.dumps(summary,indent=2))
