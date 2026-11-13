"""Compare all port blocks to independently executed original TensorFlow code."""
import json
from pathlib import Path
import numpy as np
import torch
from independent_coco import ROOT,digest
from pca_kd import PCAKD,zca


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assets=ROOT/'.tools/PCA-KD-converted'
    manifest=json.loads((assets/'manifest.json').read_text(encoding='utf-8'))
    if digest(assets/'oracle.npz')!=manifest['oracle_sha256']:raise ValueError('Oracle changed')
    out=ROOT/'outputs/adaptive-reference-20261113/pca-kd/conversion-v1'
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    with np.load(assets/'oracle.npz',allow_pickle=False) as data:
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            model=PCAKD(device=device)
            for name in ['noise','ramp','natural']:
                def compare(key,value):
                    expected=data[name+'_'+key]
                    actual=value.detach().cpu().numpy()
                    error=np.abs(actual-expected)
                    rows.append(dict(device=device,case=name,stage=key,shape=list(actual.shape),
                        mae=float(error.mean()),max_abs=float(error.max()),finite=bool(np.isfinite(actual).all())))
                c=torch.from_numpy(data[name+'_content'].copy()).to(device)
                s=torch.from_numpy(data[name+'_style'].copy()).to(device)
                with torch.inference_mode():
                    cf,cs=model.encode(c);sf,_=model.encode(s)
                    for b in range(4):
                        compare(f'enc{b}',cf[b]);compare(f'skip{b}',cs[b]);compare(f'style{b}',sf[b])
                    x=cf[-1]
                    for b in reversed(range(4)):
                        x=zca(x,sf[b]);compare(f'zca{b}',x)
                        x=model.decode_block(b,x,cs[b]);compare(f'dec{b}',x)
                    compare('output',x.clamp(0,1))
    final=[r for r in rows if r['stage']=='output']
    passed=all(r['finite'] for r in rows) and all(r['max_abs']<2e-4 for r in final)
    report=dict(source_hashes={p:digest(Path(__file__).parent/p) for p in ['pca_kd.py','convert_pca_kd.py','check_pca_conversion.py']},
        converted_manifest_sha256=digest(assets/'manifest.json'),rows=rows,final_outputs=final,
        output_max_abs_gate=2e-4,passed=passed,scope='Numerical conversion only; not photographic quality or video validation')
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=passed,final_outputs=final)),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
