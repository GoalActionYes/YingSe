"""Read saved pixels and reconstruct both hidden explanations independently."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/adaptive-reference-20261113/tst100k'
OUT=BASE/'style-task-ambiguity-v1'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    d=json.loads((OUT/'results.json').read_text());assert d['finished'] and not d['training_started'] and not d['product_modified']
    registry_path=BASE/'regional-correction-inputs-v1/results.json';assert digest(registry_path)==d['registry_sha256']
    reg=json.loads(registry_path.read_text());items={i['index']:i for i in reg['inputs']};selected=[];groups=set()
    for row in reg['rows']:
        if row['split']!='train' or row['group_id'] in groups:continue
        ids=[row['roles'][role]['feature_index'] for role in ['content','reference']]
        if not all(all(.02<=items[i][k]<=.98 for k in ['soft_person_fraction','hard_person_fraction']) for i in ids):continue
        selected.append(row['index']);groups.add(row['group_id'])
        if len(selected)==4:break
    assert selected==d['selection'] and [c['index'] for c in d['cases']]==[i for i in selected for _ in range(2)]
    for n,h in d['source_sha256'].items():assert digest(OUT/n)==h
    count=0
    for c in d['cases']:
        arrays={}
        for name,f in c['files'].items():
            p=OUT/f['path'];assert digest(p)==f['sha256']
            a=np.array(Image.open(p));assert a.dtype==np.uint8 and a.shape==(256,256,3)
            assert hashlib.sha256(a.tobytes()).hexdigest()==f['pixel_sha256'];arrays[name]=a.astype(np.int32);count+=1
        for role,name in [('content','target'),('reference','latent-A')]:
            p=ROOT/c['parents'][role]['path'];assert digest(p)==c['parents'][role]['sha256']
            original=np.array(Image.open(p).convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            assert hashlib.sha256(original.tobytes()).hexdigest()==c['parents'][role]['pixel_sha256']
            assert np.array_equal(original.clip(16,239),arrays[name])
        mask=ROOT/c['mask_path'];assert digest(mask)==c['mask_sha256']
        p=np.load(mask,allow_pickle=False)>.5;shift=np.array(c['delta'],np.int32)
        observed=arrays['latent-A']+(~p)[...,None]*shift
        assert np.array_equal(observed,arrays['observed-reference'])
        assert np.array_equal(arrays['latent-B']+shift,observed)
        assert np.array_equal(arrays['target'],arrays['answer-A'])
        assert np.array_equal(arrays['target']+shift,arrays['answer-B'])
        h=hashlib.sha256(arrays['target'].astype(np.uint8).tobytes()+observed.astype(np.uint8).tobytes()).hexdigest()
        assert h==c['input_A_sha256']==c['input_B_sha256']
        a,b=arrays['answer-A'],arrays['answer-B'];mae=float(np.abs(a-b).mean());assert mae==c['answer_rgb_mae'] and mae>0
        midpoint=(a+b)/2
        risk=(np.abs(midpoint-a).mean()+np.abs(midpoint-b).mean())/2
        assert risk==mae/2==c['equal_prior_mean_l1_lower_bound']
    assert count==48 and d['exact_input_collisions']==8 and not d['unique_objective_answer_from_rgb_only']
    result=dict(finished=True,png_files=48,exact_collisions=8,lower_bound_attained_by_midpoint=True,
                results_sha256=digest(OUT/'results.json'),scope='constructed two-history task only; not a natural-data impossibility result')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
