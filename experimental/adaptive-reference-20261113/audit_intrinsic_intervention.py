"""Independent saved-layer/input measurement and fixed necessary-condition audit."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/adaptive-reference-20261113/background-intervention-20261116-v2'
FIRST=ROOT/'outputs/adaptive-reference-20261113/intrinsic-prior-first-v1'
OUT=ROOT/'outputs/adaptive-reference-20261113/intrinsic-intervention-v1'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def linear(rgb):
    a=rgb.astype(np.float64)/255;out=np.empty_like(a);low=a<=.04045
    out[low]=a[low]/12.92;out[~low]=np.power((a[~low]+.055)/1.055,2.4)
    return out

def color(a):
    logs=np.log(np.clip(a.astype(np.float64),1e-4,None))
    return logs[...,[0,2]]-logs[...,[1]]

def metrics(a,b,mask):
    if not mask.any():return None
    v=np.mean((color(a)-color(b))[mask],axis=0)
    return dict(vector=v.tolist(),norm=float(np.sqrt(np.sum(v*v))),floor_fraction=float(np.mean(a[mask]<1e-4)),pixels=int(mask.sum()))

def close(actual,expected):
    if actual is None:assert expected is None;return
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for k in actual:close(actual[k],expected[k])
    else:assert np.allclose(actual,expected,atol=1e-10,rtol=0),(actual,expected)

def main():
    report=json.loads((OUT/'results.json').read_text());assert report['finished'] and not report['product_integrated']
    assert digest(SOURCE/'results.json')==report['source_results_sha256'] and digest(FIRST/'results.json')==report['first_results_sha256']
    for n,h in report['source_sha256'].items():assert digest(OUT/n)==h
    source=json.loads((SOURCE/'results.json').read_text());first=json.loads((FIRST/'results.json').read_text())
    assert report['assets']==first['assets'] and report['config']==first['config']
    for f in report['assets']['files']:assert digest(ROOT/'.tools/Marigold-IID-audit/models'/f['path'])==f['sha256']
    bridge=OUT/report['bridge']['file'];assert digest(bridge)==report['bridge']['sha256']
    assert np.array_equal(np.load(bridge,allow_pickle=False),np.load(FIRST/first['runs'][0]['file'],allow_pickle=False))
    assert len(report['cases'])==len(source['cases'])==6
    passed=[];predictions=1;panel=Image.new('RGB',(1280,6*282),'white');draw=ImageDraw.Draw(panel)
    for i,(case,old) in enumerate(zip(report['cases'],source['cases'],strict=True)):
        assert case['pair']==old['pair'];folder=SOURCE/case['pair'];p=folder/'reference-mask.png'
        assert digest(p)==case['mask_sha256'];person=np.array(Image.open(p).convert('L'))>0
        identity=np.array(Image.open(SOURCE/old['variants'][0]['reference']['path']))
        selected=person&(identity.min(axis=-1)>=32)&(identity.max(axis=-1)<=223)
        assert int(selected.sum())==case['primary_pixels'] and int(person.sum())==case['person_pixels']
        assert case['descriptive_only']==(not person.any())
        masks=dict(primary=selected,person=person,all=np.full(person.shape,True))
        base=None;measured={};layers={}
        assert len(case['rows'])==len(old['variants'])==12
        for row,prior in zip(case['rows'],old['variants'],strict=True):
            assert row['name']==prior['name'];p=SOURCE/prior['reference']['path']
            assert digest(p)==row['input_sha256']==prior['reference']['sha256'];rgb=np.array(Image.open(p))
            p=OUT/row['prediction']['file'];assert digest(p)==row['prediction']['sha256']
            a=np.load(p,allow_pickle=False);assert a.shape==(3,256,256,3) and np.isfinite(a).all();predictions+=1
            if base is None:base=a.copy()
            if row['name']=='identity_repeat':assert np.array_equal(base,a)
            stats={name:None if not mask.any() else dict(shading=metrics(a[1],base[1],mask),
                albedo=metrics(a[0],base[0],mask),input=metrics(linear(rgb),linear(identity),mask)) for name,mask in masks.items()}
            close(stats,row['regions']);measured[row['name']]=stats
            if row['name'] in ['identity','background_green_100','whole_green']:layers[row['name']]=a[1]
        for check in case['checks']:
            supported=int(selected.sum())>=32;assert check['supported']==supported
            if not supported:
                assert not check['passed']
                if not case['descriptive_only']:passed.append(False)
                continue
            whole=measured['whole_'+check['color']]['primary'];bg=measured['background_'+check['color']+'_100']['primary']
            v=np.array(whole['shading']['vector']);k=np.array(whole['input']['vector']);n=np.linalg.norm(v);kn=np.linalg.norm(k)
            cosine=float(np.dot(v,k)/(n*kn)) if n*kn else None
            fraction=float(n/kn) if kn else None;ratio=float(bg['shading']['norm']/n) if n else None
            for key,value in [('cosine',cosine),('response_fraction',fraction),('background_to_whole_ratio',ratio)]:close(value,check[key])
            response=cosine is not None and cosine>=.5 and fraction>=.25
            invariant=ratio is not None and ratio<=.5
            reliable=max(measured['identity']['primary']['shading']['floor_fraction'],whole['shading']['floor_fraction'],bg['shading']['floor_fraction'])<=.01
            assert response==check['response_pass'] and invariant==check['invariance_pass'] and reliable==check['floor_reliable']
            assert bool(response and invariant and reliable)==check['passed']
            if not case['descriptive_only']:passed.append(check['passed'])
        draw.text((4,i*282+4),case['pair']+' | original / S identity / bg green input / S bg / S whole green',fill='black')
        bginput=np.array(Image.open(folder/'background_green_100-reference.png'))
        tiles=[identity,layers['identity'],bginput,layers['background_green_100'],layers['whole_green']]
        for j,a in enumerate(tiles):
            if j in [1,3,4]:
                x=a.astype(float).clip(0,1);a=np.rint(np.where(x<=.0031308,x*12.92,1.055*x**(1/2.4)-.055)*255).astype(np.uint8)
            panel.paste(Image.fromarray(a),(j*256,i*282+26))
    assert len(passed)==8 and predictions==73 and all(passed)==report['necessary_conditions_pass']
    panel.save(OUT/'diagnostic-panel.png')
    audit=dict(finished=True,predictions_checked=predictions,directions=8,passed=sum(passed),necessary_conditions_pass=all(passed),
        results_sha256=digest(OUT/'results.json'),scope='offline input/layer/log-metric audit, not independent GPU or natural evaluation')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit))

if __name__=='__main__':main()
