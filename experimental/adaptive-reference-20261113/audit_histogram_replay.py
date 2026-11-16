"""Independent saved-condition and pixel-statistic audit, without replay helper."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from independent_coco import ROOT,digest


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/histogram-replay-20261116-v1'
    base=ROOT/'outputs/adaptive-reference-20261113/background-intervention-20261116-v2'
    r=json.loads((out/'results.json').read_text());parent=json.loads((base/'results.json').read_text())
    assert r['finished'] and len(r['cases'])==6 and digest(base/'results.json')==r['input_results_sha256']
    for n,h in r['source_sha256'].items():assert digest(out/'frozen-source'/n)==h and digest(Path(__file__).with_name(n))==h
    parents={x['pair']:x for x in parent['cases']};rows=[];count=0;region_count=0
    def read(root,entry):
        nonlocal count
        p=root/entry['path'];assert digest(p)==entry['sha256'];count+=1
        with Image.open(p) as im:return np.array(im.convert('RGB'))
    for c in r['cases']:
        old=parents[c['pair']];original=read(out,c['baseline']);source=read(base,old['input_file'])
        assert np.array_equal(original,read(base,old['variants'][0]['methods']['adaptive']['file']))
        with Image.open(base/c['pair']/'target-mask.png') as im:mask=np.array(im)>0
        x=source.astype(float);y=x@np.array([.299,.587,.114])
        regions=dict(person=mask,background=~mask,neutral_proxy=(np.ptp(x,axis=2)<=20)&(y>=40)&(y<=220),whole=np.ones(mask.shape,bool))
        images={}
        assert len(c['variants'])==6
        for v in c['variants']:
            entry=v['conditions_file'];assert digest(out/entry['path'])==entry['sha256']
            with np.load(out/entry['path'],allow_pickle=False) as a:
                for n in a.files:assert a[n].shape==(1,3,128) and np.isfinite(a[n]).all() and (a[n]>=0).all() and np.allclose(a[n].sum(-1),1,atol=1e-6)
                np.testing.assert_array_equal(a['full'],a['h1'])
                np.testing.assert_array_equal(a['y_only'][:,:1],a['h1'][:,:1]);np.testing.assert_array_equal(a['y_only'][:,1:],a['h0'][:,1:])
                np.testing.assert_array_equal(a['chroma_only'][:,:1],a['h0'][:,:1]);np.testing.assert_array_equal(a['chroma_only'][:,1:],a['h1'][:,1:])
                np.testing.assert_array_equal(np.abs(a['h1']-a['h0']).sum((0,2)),v['histogram_l1_by_channel'])
            saved=next(z for z in old['variants'] if z['name']==v['name'])
            row=dict(pair=c['pair'],variant=v['name'],histogram_l1_by_channel=v['histogram_l1_by_channel'],methods={})
            for method,entry in v['methods'].items():
                actual=read(out,entry['file']);images[(v['name'],method)]=actual
                if method=='full':assert np.array_equal(actual,read(base,saved['methods']['adaptive']['file']))
                for n,m in regions.items():
                    z=entry['regions'][n]
                    if not m.any():assert z is None;continue
                    d=actual[m].astype(float)-original[m].astype(float)
                    assert z['pixels']==int(m.sum())
                    assert abs(float(np.abs(d).mean())-z['rgb_mae'])<1e-10
                    np.testing.assert_allclose(d.mean(0),z['rgb_mean_delta'],atol=1e-10)
                    assert abs(float((d@np.array([.299,.587,.114])).mean())-z['luminance_mean_delta'])<1e-10
                    a,b=actual[m].astype(float),original[m].astype(float)
                    assert abs(float((np.ptp(a,axis=1)-np.ptp(b,axis=1)).mean())-z['saturation_range_mean_delta'])<1e-10
                    assert z['sensitive_over_one_code']==(z['rgb_mae']>1)
                    region_count+=1
                row['methods'][method]=dict(regions=entry['regions'],producer_grid_mae=entry['grid_mae'],producer_projection_mae=entry['projection_mae'])
            rows.append(row)
        canvas=Image.new('RGB',(1024,864),'#22252a');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
        for rr,method in enumerate(['full','y_only','chroma_only']):
            for col,label in enumerate(['source','baseline','green','magenta']):
                img=source if col==0 else original if col==1 else images[(f'background_{label}_100',method)]
                canvas.paste(Image.fromarray(img),(col*256,rr*288+30));draw.text((col*256+3,rr*288+6),method+' '+label,font=font,fill='white')
        canvas.save(out/c['pair']/'review.png')
    evidence=dict(scope=r['scope'],results_sha256=digest(out/'results.json'),model_sha256=r['model_sha256'],source_sha256=r['source_sha256'],audit_script_sha256=digest(Path(__file__)),
        pairs=6,conditions=36,outputs=114,exact_product_replays=42,checked_png_reads=count,recomputed_region_records=region_count,
        grid_and_projection_metrics='Producer values, not independently replayed in this saved-pixel audit',rows=rows,app_promoted=False,human_evaluations=0)
    (ROOT/'docs/research/2026-11-16-histogram-replay-evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in evidence.items() if k not in ['rows','source_sha256']}))
    for coverage in [25,50,100]:
        for method in ['full','y_only','chroma_only']:
            data=[z['methods'][method]['regions']['person']['rgb_mae'] for z in rows if z['variant'].endswith('_'+str(coverage)) and z['methods'][method]['regions']['person']]
            print(coverage,method,'person_mean',np.mean(data),'range',min(data),max(data),'over1',sum(x>1 for x in data))


if __name__=='__main__':main()
