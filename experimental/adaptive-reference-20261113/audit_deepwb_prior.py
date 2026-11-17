"""Recompute fixed property diagnostics from saved maps and pixels, without GPU."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/adaptive-reference-20261113/deepwb-prior-v2'
Y=np.array([.2126,.7152,.0722])

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def mapped(rgb,row):
    # Preserve author's uint8 intermediate products, including wraparound.
    r,g,b=rgb.reshape(-1,3).T
    x=np.stack((r,g,b,r*g,r*b,g*b,r*r,g*g,b*b,r*g*b,np.ones(len(r),np.int64)),1)
    return (x@np.array(row['coef']).T+np.array(row['intercept'])).reshape(rgb.shape)

def metric(a,b,mask):
    delta=(a-b)*255;y=delta@Y
    c=np.stack((delta[...,0]-y,delta[...,2]-y),-1)[mask].mean(0)
    return dict(chroma_vector=c.tolist(),chroma_norm=float(np.linalg.norm(c)),
                rgb_mae=float(np.abs(delta[mask]).mean()),y_mean=float(y[mask].mean()))

def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:close(a[k],b[k])
    else:assert np.allclose(a,b,rtol=0,atol=1e-9),(a,b)

def main():
    report=json.loads((OUT/'results.json').read_text())
    assert report['finished'] and not report['app_promoted']
    assert len(report['cases'])==6
    for name,field in [('driver.py','driver_sha256'),('protocol.md','protocol_sha256')]:
        assert digest(OUT/name)==report[field]
    assets=ROOT/'.tools/DeepWB-audit'
    for row in report['assets']['files']:
        p=assets/'source'/row['path'];data=p.read_bytes()
        assert digest(p)==row['sha256'] and len(data)==row['bytes']
        assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==row['git_blob']
    source=ROOT/'outputs/adaptive-reference-20261113/background-intervention-20261116-v2'
    assert digest(source/'results.json')==report['source_sha256']
    parent=json.loads((source/'results.json').read_text())
    required=['identity','identity_repeat','background_green_25','background_magenta_25',
              'background_green_50','background_magenta_50','background_green_100',
              'background_magenta_100','whole_green','whole_magenta','whole_contrast','whole_desaturate']
    tests=[];images=0;panel=Image.new('RGB',(256*6,6*282),'white');draw=ImageDraw.Draw(panel)
    for ci,(case,old) in enumerate(zip(report['cases'],parent['cases'],strict=True)):
        assert case['pair']==old['pair'] and [r['name'] for r in case['rows']]==required
        p=ROOT/case['mask_path'];assert digest(p)==case['mask_sha256']
        mask=np.array(Image.open(p).convert('L'))>0
        assert case['descriptive_only']==(not mask.any())
        if not mask.any():mask=np.ones(mask.shape,bool)
        baseline=case['rows'][0];original=np.array(Image.open(ROOT/baseline['input_path']))
        base=mapped(original,baseline);by={r['name']:r for r in case['rows']}
        for row,prior in zip(case['rows'],old['variants'],strict=True):
            p=ROOT/row['input_path'];assert digest(p)==row['input_sha256']==prior['reference']['sha256']
            rgb=np.array(Image.open(p));q=OUT/row['output_path'];assert digest(q)==row['output_sha256']
            recreated=mapped(rgb,row).clip(0,1)
            assert np.array_equal(np.rint(recreated*255).astype(np.uint8),np.array(Image.open(q)))
            common=mapped(original,row)
            close(metric(common,base,mask),row['common_raw'])
            close(metric(common.clip(0,1),base.clip(0,1),mask),row['common_clipped'])
            close(metric(rgb/255,original/255,mask),row['input_delta'])
            if row['name']=='identity_repeat':
                assert all(row[k]==baseline[k] for k in ['coef','intercept','output_float_sha256'])
                close(row['common_clipped'],dict(chroma_vector=[0,0],chroma_norm=0,rgb_mae=0,y_mean=0))
            images+=1
        for check in case['checks']:
            c=check['color'];whole=by['whole_'+c];bg=by['background_'+c+'_100']
            v=np.array(whole['common_clipped']['chroma_vector']);k=np.array(whole['input_delta']['chroma_vector'])
            n=np.linalg.norm(v);kn=np.linalg.norm(k)
            cosine=float(-v@k/(n*kn)) if n*kn else None
            fraction=float(n/kn) if kn else None
            ratio=float(bg['common_clipped']['chroma_norm']/n) if n else None
            for a,b in [(cosine,check['opposing_cosine']),(fraction,check['compensation_fraction']),(ratio,check['background_to_whole_ratio'])]:
                assert (a is None and b is None) or np.isclose(a,b,rtol=0,atol=1e-12)
            response=cosine is not None and cosine>=.5 and fraction>=.25
            invariant=ratio is not None and ratio<=.5
            assert response==check['response_pass'] and invariant==check['invariance_pass']
            if not case['descriptive_only']:tests.append(response and invariant)
        # Fixed six columns: original, identity output, background green input/output, whole green input/output.
        tiles=[Image.fromarray(original),Image.open(OUT/baseline['output_path'])]
        for n in ['background_green_100','whole_green']:
            tiles.extend([Image.open(ROOT/by[n]['input_path']),Image.open(OUT/by[n]['output_path'])])
        for col,im in enumerate(tiles):panel.paste(im,(col*256,ci*282+26))
        draw.text((4,ci*282+4),case['pair']+' | original / AWB / bg input / AWB / whole input / AWB',fill='black')
    assert len(tests)==8 and images==72
    assert all(tests)==report['necessary_conditions_pass']
    panel.save(OUT/'diagnostic-panel.png')
    result=dict(finished=True,images_checked=images,person_direction_checks=8,
                necessary_conditions_pass=all(tests),results_sha256=digest(OUT/'results.json'),
                note='offline file/map/metric audit; no independent GPU rerun or human quality evaluation')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
