"""One frozen offline property test; no product inference or ground truth."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'.tools/DeepWB-audit'
SOURCE=ROOT/'outputs/adaptive-reference-20261113/background-intervention-20261116-v2'
OUT=ROOT/'outputs/adaptive-reference-20261113/deepwb-prior-v2'
Y=np.array([.2126,.7152,.0722])

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def chroma(a):
    y=a@Y
    return np.stack((a[...,0]-y,a[...,2]-y),-1)

def movement(a,b,mask):
    delta=(a-b)*255
    c=chroma(delta)[mask].mean(0)
    return dict(chroma_vector=c.tolist(),chroma_norm=float(np.linalg.norm(c)),
                rgb_mae=float(np.abs(delta[mask]).mean()),y_mean=float((delta@Y)[mask].mean()))

def properties(rows):
    by={r['name']:r for r in rows};checks=[]
    for color in ['green','magenta']:
        whole=by['whole_'+color];background=by['background_'+color+'_100']
        compensation=np.array(whole['common_clipped']['chroma_vector'])
        known=np.array(whole['input_delta']['chroma_vector'])
        n=float(np.linalg.norm(compensation));k=float(np.linalg.norm(known))
        cosine=float(-compensation@known/(n*k)) if n*k else None
        magnitude=n/k if k else None
        ratio=background['common_clipped']['chroma_norm']/n if n else None
        checks.append(dict(color=color,opposing_cosine=cosine,compensation_fraction=magnitude,
                           background_to_whole_ratio=ratio,
                           response_pass=bool(cosine is not None and cosine>=.5 and magnitude>=.25),
                           invariance_pass=bool(ratio is not None and ratio<=.5)))
    return checks

def main():
    OUT.mkdir(exist_ok=False)
    for p in [ASSETS/'vendor',ROOT/'.tools/ModFlows-audit/vendor',ASSETS/'source/PyTorch']:
        sys.path.insert(0,str(p))
    import torch
    from arch.deep_wb_single_task import deepWBnet
    from utilities.deepWB import deep_wb
    import utilities.utils as utils
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    assets=json.loads((ASSETS/'assets.json').read_text())
    for r in assets['files']:
        assert digest(ASSETS/'source'/r['path'])==r['sha256']
    net=deepWBnet()
    state=torch.load(ASSETS/'source/PyTorch/models/net_awb.pth',map_location='cpu',weights_only=True)
    assert all(torch.isfinite(v).all() for v in state.values())
    net.load_state_dict(state,strict=True);net.to('cuda').eval()
    captured=[];ranges=[]
    original=utils.get_mapping_func
    def fit(a,b):
        m=original(a,b);captured.append(m);return m
    utils.get_mapping_func=fit
    def hook(module,args,output):
        ranges.append(dict(min=float(output.min()),max=float(output.max()),
                           outside_fraction=float(((output<0)|(output>1)).float().mean())))
    net.register_forward_hook(hook)
    parent=json.loads((SOURCE/'results.json').read_text())
    protocol=Path(__file__).with_name('deepwb-prior-protocol.md')
    report=dict(scope='consumed COCO controlled property diagnosis; not quality or product acceptance',
                assets=assets,source_sha256=digest(SOURCE/'results.json'),
                protocol_sha256=digest(protocol),driver_sha256=digest(__file__),
                primary='clipped common-original RGB, reference person mask; whole image for no-person descriptive controls',
                cases=[],finished=False,app_promoted=False)
    (OUT/'protocol.md').write_bytes(protocol.read_bytes())
    (OUT/'driver.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    start=time.perf_counter()
    with threadpool_limits(limits=4):
      for case in parent['cases']:
        folder=OUT/case['pair'];folder.mkdir()
        mask=np.array(Image.open(SOURCE/case['pair']/'reference-mask.png').convert('L'))>0
        descriptive=not mask.any()
        if descriptive:mask=np.ones(mask.shape,bool)
        identity=None;base_raw=None;base_clip=None;rows=[]
        for row in case['variants']:
            if time.perf_counter()-start>600:raise TimeoutError('Frozen execution budget exceeded')
            p=SOURCE/row['reference']['path'];assert digest(p)==row['reference']['sha256']
            im=Image.open(p);assert im.mode=='RGB' and im.size==(256,256)
            rgb=np.array(im)
            captured.clear();ranges.clear();t=time.perf_counter()
            output=deep_wb(im,task='awb',net_awb=net,device='cuda',s=656)
            torch.cuda.synchronize();seconds=time.perf_counter()-t
            assert len(captured)==len(ranges)==1 and np.isfinite(output).all()
            m=captured[0]
            recreated=utils.apply_mapping_func(rgb,m).clip(0,1)
            assert np.array_equal(recreated,output)
            if identity is None:
                identity=rgb.copy()
                base_raw=utils.apply_mapping_func(identity,m)
                base_clip=base_raw.clip(0,1)
            common_raw=utils.apply_mapping_func(identity,m);common_clip=common_raw.clip(0,1)
            png=np.rint(output*255).astype(np.uint8)
            q=folder/(row['name']+'.png');Image.fromarray(png).save(q)
            r=dict(name=row['name'],input_path=str(p.relative_to(ROOT)),input_sha256=digest(p),
                   output_path=str(q.relative_to(OUT)),output_sha256=digest(q),seconds=seconds,
                   coef=m.coef_.tolist(),intercept=m.intercept_.tolist(),rank=int(m.rank_),singular=m.singular_.tolist(),
                   network=ranges[0],output_float_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                   common_raw=movement(common_raw,base_raw,mask),common_clipped=movement(common_clip,base_clip,mask),
                   input_delta=movement(rgb/255,identity/255,mask))
            if row['name']=='identity_repeat':
                first=rows[0]
                assert all(r[k]==first[k] for k in ['coef','intercept','output_float_sha256'])
            rows.append(r)
            print(json.dumps(dict(pair=case['pair'],name=r['name'],seconds=round(seconds,3))),flush=True)
        report['cases'].append(dict(pair=case['pair'],descriptive_only=descriptive,
                                   mask_path=str((SOURCE/case['pair']/'reference-mask.png').relative_to(ROOT)),
                                   mask_sha256=digest(SOURCE/case['pair']/'reference-mask.png'),
                                   rows=rows,checks=properties(rows)))
        (OUT/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    report.update(finished=True,seconds=time.perf_counter()-start,
                  necessary_conditions_pass=all(x['response_pass'] and x['invariance_pass'] for c in report['cases'] if not c['descriptive_only'] for x in c['checks']),
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    assert digest(__file__)==report['driver_sha256'] and digest(protocol)==report['protocol_sha256']
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['finished','seconds','necessary_conditions_pass']}),flush=True)

if __name__=='__main__':main()
