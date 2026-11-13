"""Explain new-vs-old Canon outputs before interpreting color-probe comparisons."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from canonical_model import ROOT,CanonicalModel
from independent_coco import digest


def rgb(path):
    with Image.open(path) as image: return np.array(image.convert('RGB'))


def resize(image):
    return np.array(Image.fromarray(image).resize((640,360),Image.Resampling.BILINEAR))


def main():
    data=ROOT/'outputs/adaptive-reference-20261113/pst50'
    out=data/'diagnostic-canonical-resolution-v1'; out.mkdir(exist_ok=False)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4); model=CanonicalModel(); rows=[]
    report=dict(scope='Already-used development pairs only; same checkpoint, source and runtime, input resolution changes',
        model=model.provenance,script_sha256=digest(Path(__file__)),cases=rows)
    for index in range(1,6):
        target,reference=rgb(data/f'content_709/in{index}.png'),rgb(data/f'paired_style/tar{index}.png')
        golden=resize(rgb(data/f'development-salut/{index}-Canon_E2E.png'))
        full_own,full_meta=model.predict(target,reference)
        full=resize(full_own)
        small,small_meta=model.predict(resize(target),resize(reference))
        row=dict(index=index,native_target_shape=list(target.shape),native_reference_shape=list(reference.shape),
            full_vs_recorded_mae=float(np.abs(full.astype(np.float32)-golden.astype(np.float32)).mean()),
            full_vs_640_mae=float(np.abs(full.astype(np.float32)-small.astype(np.float32)).mean()),
            full_seconds=full_meta,small_seconds=small_meta)
        Image.fromarray(full).save(out/f'{index}-native.png'); Image.fromarray(small).save(out/f'{index}-640.png')
        rows.append(row)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)


if __name__=='__main__': main()
