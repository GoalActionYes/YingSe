"""Cached mechanism assay, not a new color model or semantic ground truth."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/adaptive-reference-20261113/tst100k'
OUT=BASE/'semantic-category-assay-v1'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def measure(weights,left,right):
    m=weights@right
    person=float(left@m/left.sum()) if left.sum()>0 else None
    b=1-left
    background=float(b@(1-m)/b.sum()) if b.sum()>0 else None
    balanced=(person+background)/2 if person is not None and background is not None else None
    return dict(person=person,background=background,balanced=balanced,lift=balanced-.5 if balanced is not None else None)

def main():
    torch.set_num_threads(4);start=time.perf_counter();OUT.mkdir(exist_ok=False)
    region=BASE/'regional-correction-inputs-v1';tokens=BASE/'semantic-token-inputs-v1'
    r=json.loads((region/'results.json').read_text());t=json.loads((tokens/'results.json').read_text())
    assert r['finished'] and t['finished'] and t['registry_sha256']==digest(region/'results.json')
    assert len(r['rows'])==1088 and len(t['inputs'])==len(r['inputs'])==1365
    names=['assay_semantic_categories.py','semantic-category-assay-protocol.md']
    report=dict(scope='automatic person-label consistency of failed fixed correspondence, not semantic accuracy or quality',
                region_report_sha256=digest(region/'results.json'),token_report_sha256=digest(tokens/'results.json'),
                source_sha256={n:digest(Path(__file__).with_name(n)) for n in names},cases=[],finished=False,
                trained=False,product_integrated=False)
    (OUT/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for n in names:(OUT/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    registry={x['index']:x for x in r['inputs']};lookup={};panels=[]
    for item in t['inputs']:
        p=tokens/item['path'];other=registry[item['index']]
        assert digest(p)==item['sha256'] and item['pixel_sha256']==other['pixel_sha256']
        mask_path=region/other['mask_path'];assert digest(mask_path)==other['mask_sha256']
        with np.load(p,allow_pickle=False) as a:features=torch.from_numpy(a['features'].copy())
        mask=np.load(mask_path,allow_pickle=False);assert mask.shape==(256,256)
        small=np.array(Image.fromarray(mask).resize((252,252),Image.Resampling.BILINEAR))
        assert item['working_dimensions']==[252,252]
        probability=F.adaptive_avg_pool2d(F.avg_pool2d(torch.from_numpy(small)[None,None],14,14),(8,8)).reshape(64)
        lookup[item['index']]=(F.normalize(features,dim=-1),probability)
    for row in r['rows']:
        assert time.perf_counter()-start<120,'Frozen CPU budget exceeded'
        ids=[row['roles'][role]['feature_index'] for role in ['content','reference']]
        (q,left),(k,right)=[lookup[i] for i in ids]
        for role,i in zip(['content','reference'],ids,strict=True):assert row['roles'][role]['pixel_sha256']==registry[i]['pixel_sha256']
        weights=torch.softmax(q@k.T*10,dim=-1)
        hard=F.one_hot(weights.argmax(1),num_classes=64).float()
        ok=lambda i:all(.02<=registry[i][name]<=.98 for name in ['soft_person_fraction','hard_person_fraction'])
        supported=all(ok(i) for i in ids)
        entry=dict(index=row['index'],split=row['split'],feature_indices=ids,supported=supported,
                   soft=measure(weights.numpy().astype(float),left.numpy().astype(float),right.numpy().astype(float)),
                   top1=measure(hard.numpy().astype(float),left.numpy().astype(float),right.numpy().astype(float)))
        report['cases'].append(entry)
        if row['split']=='validation' and supported and len(panels)<4:
            panels.append((row['index'],left.numpy(),(weights@right).numpy(),right.numpy()))
    report['summary']={}
    for split,count,expected_support in [('train',1024,573),('validation',64,28)]:
        rows=[c for c in report['cases'] if c['split']==split];assert len(rows)==count
        selected=[c for c in rows if c['supported']];assert len(selected)==expected_support
        values=[c['soft']['lift'] for c in selected];assert all(v is not None for v in values)
        mean=float(np.mean(values));positive=float(np.mean(np.array(values)>0))
        report['summary'][split]=dict(all_rows=len(rows),supported=len(selected),mean_balanced=float(np.mean([c['soft']['balanced'] for c in selected])),
            mean_lift=mean,positive_fraction=positive,signal_pass=mean>=.1 and positive>=.75,
            top1_mean_balanced=float(np.mean([c['top1']['balanced'] for c in selected])))
    panel=Image.new('RGB',(768,4*280),'white');draw=ImageDraw.Draw(panel)
    for i,(index,*arrays) in enumerate(panels):
        draw.text((4,i*280+4),f'{index}: query person / matched reference person / reference person',fill='black')
        for j,a in enumerate(arrays):
            im=Image.fromarray(np.rint(a.reshape(8,8).clip(0,1)*255).astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST).convert('RGB')
            panel.paste(im,(j*256,i*280+24))
    assert len(panels)==4;panel.save(OUT/'category-panel.png')
    report.update(finished=True,seconds=time.perf_counter()-start,signal_pass=all(v['signal_pass'] for v in report['summary'].values()))
    assert all(digest(Path(__file__).with_name(n))==h for n,h in report['source_sha256'].items())
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(summary=report['summary'],seconds=report['seconds'],signal_pass=report['signal_pass'])),flush=True)

if __name__=='__main__':main()
