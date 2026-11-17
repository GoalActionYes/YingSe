"""Independent NumPy pooling, cosine correspondence, and categorical statistics."""
import hashlib,json,math,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/adaptive-reference-20261113/tst100k'
OUT=BASE/'semantic-category-assay-v1'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pooled(mask):
    image=np.asarray(Image.fromarray(mask).resize((252,252),Image.Resampling.BILINEAR),dtype=np.float64)
    patches=image.reshape(18,14,18,14).mean((1,3))
    out=np.empty((8,8))
    for y in range(8):
        for x in range(8):out[y,x]=patches[math.floor(y*18/8):math.ceil((y+1)*18/8),math.floor(x*18/8):math.ceil((x+1)*18/8)].mean()
    return out.ravel()

def statistics(w,a,b):
    matched=w@b;other=1-a
    person=float(np.sum(a*matched)/np.sum(a)) if np.sum(a)>0 else None
    background=float(np.sum(other*(1-matched))/np.sum(other)) if np.sum(other)>0 else None
    balanced=(person+background)/2 if person is not None and background is not None else None
    return dict(person=person,background=background,balanced=balanced,lift=balanced-.5 if balanced is not None else None)

def main():
    report=json.loads((OUT/'results.json').read_text());assert report['finished'] and not report['trained'] and not report['product_integrated']
    for n,h in report['source_sha256'].items():assert digest(OUT/n)==h
    region=BASE/'regional-correction-inputs-v1';token=BASE/'semantic-token-inputs-v1'
    assert digest(region/'results.json')==report['region_report_sha256']
    assert digest(token/'results.json')==report['token_report_sha256']
    r=json.loads((region/'results.json').read_text());t=json.loads((token/'results.json').read_text())
    lookup={i['index']:i for i in r['inputs']};data={}
    for item in t['inputs']:
        path=token/item['path'];entry=lookup[item['index']];p=region/entry['mask_path']
        assert digest(path)==item['sha256'] and digest(p)==entry['mask_sha256']
        with np.load(path,allow_pickle=False) as f:v=f['features'].astype(np.float64)
        v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
        data[item['index']]=(v,pooled(np.load(p,allow_pickle=False)))
    max_error=0;recalculated=[]
    for row,saved in zip(r['rows'],report['cases'],strict=True):
        assert row['index']==saved['index'] and row['split']==saved['split']
        ids=[row['roles'][role]['feature_index'] for role in ['content','reference']]
        assert ids==saved['feature_indices']
        supported=all(all(.02<=lookup[i][k]<=.98 for k in ['soft_person_fraction','hard_person_fraction']) for i in ids)
        assert supported==saved['supported']
        (q,a),(k,b)=[data[i] for i in ids]
        scores=(q@k.T)*10;scores-=scores.max(axis=1,keepdims=True)
        weights=np.exp(scores);weights/=weights.sum(axis=1,keepdims=True)
        hard=np.eye(64)[weights.argmax(1)]
        metrics={name:statistics(w,a,b) for name,w in [('soft',weights),('top1',hard)]}
        for name,values in metrics.items():
            for key,v in values.items():
                old=saved[name][key]
                if v is None:assert old is None
                else:
                    error=abs(v-old);max_error=max(max_error,error);assert error<=1e-6,(saved['index'],name,key,error)
        uniform=statistics(np.ones((64,64))/64,a,b)
        if uniform['balanced'] is not None:assert abs(uniform['balanced']-.5)<=1e-12
        recalculated.append(dict(split=row['split'],supported=supported,**metrics))
    passes=[]
    for split,expected in [('train',573),('validation',28)]:
        rows=[x for x in recalculated if x['split']==split and x['supported']];assert len(rows)==expected
        mean=float(np.mean([x['soft']['lift'] for x in rows]));positive=float(np.mean([x['soft']['lift']>0 for x in rows]))
        recorded=report['summary'][split]
        assert abs(mean-recorded['mean_lift'])<=1e-6 and positive==recorded['positive_fraction']
        passed=mean>=.1 and positive>=.75;assert passed==recorded['signal_pass'];passes.append(passed)
    assert all(passes)==report['signal_pass'] and len(recalculated)==1088
    result=dict(finished=True,rows=1088,cache_inputs=1365,maximum_statistic_difference=max_error,signal_pass=all(passes),
                results_sha256=digest(OUT/'results.json'),scope='cached automatic-label mechanism audit, not semantic truth or image quality')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
