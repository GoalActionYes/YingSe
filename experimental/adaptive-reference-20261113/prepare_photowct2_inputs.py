"""Prepare exact historic RGB256 pixels in the original Pillow environment."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest

base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
path=base/'regional-correction-inputs-v1/results.json';registry=json.loads(path.read_text())
inputs={e['index']:e for e in registry['inputs']};out=base/'photowct2-inputs-v1';out.mkdir(exist_ok=False)
entries={}
for row in registry['rows']:
    if row['split']!='validation':continue
    for role in ['content','reference']:
        index=row['roles'][role]['feature_index']
        if str(index) in entries:continue
        e=inputs[index];p=ROOT/e['path'];assert digest(p)==e['parent_sha256']
        with Image.open(p) as im:rgb=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==e['pixel_sha256']
        p=out/f'{index}.png';Image.fromarray(rgb).save(p)
        entries[str(index)]=dict(path=p.name,sha256=digest(p),pixel_sha256=e['pixel_sha256'])
r=dict(registry_sha256=digest(path),inputs=entries,source_sha256=digest(Path(__file__)))
(out/'results.json').write_text(json.dumps(r,indent=2));print('Prepared exact inputs:',len(entries))
