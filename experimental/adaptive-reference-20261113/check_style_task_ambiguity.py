"""Exact two-input counterexamples for proposed synthetic supervision only."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/adaptive-reference-20261113/tst100k'
OUT=BASE/'style-task-ambiguity-v1'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    registry_path=BASE/'regional-correction-inputs-v1/results.json'
    registry=json.loads(registry_path.read_text());inputs={r['index']:r for r in registry['inputs']}
    supported=lambda i:all(.02<=inputs[i][k]<=.98 for k in ['soft_person_fraction','hard_person_fraction'])
    rows=[];seen=set()
    for row in registry['rows']:
        if row['split']!='train' or row['group_id'] in seen:continue
        ids=[row['roles'][r]['feature_index'] for r in ['content','reference']]
        if not all(supported(i) for i in ids):continue
        rows.append(row);seen.add(row['group_id'])
        if len(rows)==4:break
    assert len(rows)==4
    names=['check_style_task_ambiguity.py','style-task-ambiguity-protocol.md']
    report=dict(scope='synthetic hidden-history ambiguity demonstration, not dataset error or natural quality',
                registry_sha256=digest(registry_path),source_sha256={n:digest(Path(__file__).with_name(n)) for n in names},
                selection=[r['index'] for r in rows],cases=[],finished=False,training_started=False,product_modified=False)
    for n in names:(OUT/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    (OUT/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def save(path,array):
        assert array.dtype==np.uint8
        p=OUT/path;p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(array).save(p)
        return dict(path=path,sha256=digest(p),pixel_sha256=hashlib.sha256(array.tobytes()).hexdigest())
    for row in rows:
        assert time.perf_counter()-start<30
        parents={};images={}
        for role in ['content','reference']:
            item=inputs[row['roles'][role]['feature_index']];p=ROOT/item['path'];assert digest(p)==item['parent_sha256']
            rgb=np.array(Image.open(p).convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==item['pixel_sha256']
            images[role]=rgb.clip(16,239).astype(np.int16)
            parents[role]=dict(path=item['path'],sha256=item['parent_sha256'],pixel_sha256=item['pixel_sha256'])
        refitem=inputs[row['roles']['reference']['feature_index']]
        mask_path=BASE/'regional-correction-inputs-v1'/refitem['mask_path'];assert digest(mask_path)==refitem['mask_sha256']
        person=np.load(mask_path,allow_pickle=False)>.5;assert person.any() and (~person).any()
        for sign in [1,-1]:
            delta=np.array([-8,4,-8],np.int16)*sign;t=images['content'];r=images['reference']
            observed=r+(~person)[...,None]*delta;alternative=observed-delta
            answer_a=t.copy();answer_b=t+delta
            assert all(a.min()>=0 and a.max()<=255 for a in [observed,alternative,answer_a,answer_b])
            assert np.array_equal(alternative+delta,observed)
            key=f"{row['index']}-{sign}"
            files={name:save(f'{key}/{name}.png',a.astype(np.uint8)) for name,a in
                   [('target',t),('observed-reference',observed),('latent-A',r),('latent-B',alternative),('answer-A',answer_a),('answer-B',answer_b)]}
            input_hash=hashlib.sha256(t.astype(np.uint8).tobytes()+observed.astype(np.uint8).tobytes()).hexdigest()
            mae=float(np.abs(answer_a-answer_b).mean())
            report['cases'].append(dict(index=row['index'],split='train',delta=delta.tolist(),parents=parents,
                mask_path=str(mask_path.relative_to(ROOT)),mask_sha256=digest(mask_path),person_fraction=float(person.mean()),
                files=files,input_A_sha256=input_hash,input_B_sha256=input_hash,answer_rgb_mae=mae,
                equal_prior_mean_l1_lower_bound=mae/2))
    assert len(report['cases'])==8
    panel=Image.new('RGB',(1536,4*280),'white');draw=ImageDraw.Draw(panel)
    for n,c in enumerate(report['cases'][::2]):
        draw.text((4,n*280+4),f"{c['index']}: target / same observed ref / latent A / latent B / answer A / answer B",fill='black')
        for j,name in enumerate(['target','observed-reference','latent-A','latent-B','answer-A','answer-B']):
            panel.paste(Image.open(OUT/c['files'][name]['path']),(j*256,n*280+24))
    panel.save(OUT/'diagnostic-panel.png')
    report.update(finished=True,seconds=time.perf_counter()-start,exact_input_collisions=8,
                  unique_objective_answer_from_rgb_only=False)
    assert all(digest(Path(__file__).with_name(n))==h for n,h in report['source_sha256'].items())
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(finished=True,indices=report['selection'],collisions=8,answer_mae=report['cases'][0]['answer_rgb_mae'],seconds=report['seconds'])))

if __name__=='__main__':main()
