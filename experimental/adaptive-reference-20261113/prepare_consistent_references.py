"""Prepare target-GT-derived reference augmentation from training originals only."""
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from capacity_audit import native_features,inverse_post,block_split,post_forward,render_coefficients
from reference_transfer_probe import crossfit_transfer
from consistent_reference import choose_training_reference

START_SHA='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';checkpoint=base/'multiref-control-v1/epoch-6.pth'
    if digest(cohort_path)!='124d8c670536719fc6f243b2d273849b70a89a164d8f84dbedc924b1d258862d' or digest(prep_path)!='b1bcb21c86a16aefe81aa3cd489836265b67b90c677247e42e67c5d5080fa04c' or digest(checkpoint)!=START_SHA:raise ValueError('Changed training inputs')
    load=lambda p:json.loads(p.read_text(encoding='utf-8'));cohort,prep=load(cohort_path),load(prep_path)
    if not prep['passed'] or prep['cohort_sha256']!=digest(cohort_path):raise ValueError('Invalid preparation')
    train=cohort['groups']['train'];lookup={r['original_path']:r for r in prep['files']};paths=sorted({r['content'] for r in train});groups=defaultdict(list)
    for row in train:groups[row['content']].append(row)
    if len(train)!=1024 or len(paths)!=256 or any(len(rows)!=4 or {r['style_index'] for r in rows}!={0,1,2,3} for rows in groups.values()):raise ValueError('Changed training groups')
    training_paths={r[k] for r in train for k in ['content','reference','gt']}
    out=base/'cohort-consistent-reference-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['prepare_consistent_references.py','consistent_reference.py','consistent-reference-protocol.md','reference_transfer_probe.py','capacity_audit.py','train_triplet_pilot.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Synthetic reference preparation from consumed training1024 target/GT; no validation/reserved generation; not independent acceptance',cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),checkpoint_sha256=START_SHA,source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=[],finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    model,api,manifest=author_model();state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite guide weights')
    model.load_state_dict(state,strict=True);model.eval().requires_grad_(False);report['author_revision']=manifest['revision'];mask=block_split(256,256,32);started=time.perf_counter()
    def read(relative):
        if relative not in training_paths:raise ValueError('Generation attempted a nontraining image')
        entry=lookup[relative];p=data/entry['prepared_path']
        if digest(p)!=entry['sha256']:raise ValueError('Changed training image')
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    for number,(content_path,rows) in enumerate(groups.items()):
        target=read(content_path);features=native_features(model,api,target);features.pop('guided_spatial');x=features['guided_global']
        for row in rows:
            truth=read(row['gt']);other=choose_training_reference(content_path,paths,row['index']);reference_original=read(other)
            values,_=inverse_post(truth,features['post_tables']);fit=crossfit_transfer(x,values,x,mask);coef=fit['coefficients']
            approximation=render_coefficients(model,api,target,coef);dense=post_forward(x@coef,features['post_tables']).reshape(target.shape)
            difference=int(np.abs(approximation.astype(int)-dense.astype(int)).max())
            if difference>1:raise ValueError('Native generated-map mismatch')
            generated=render_coefficients(model,api,reference_original,coef);p=out/f'{row["index"]}.png';Image.fromarray(generated).save(p)
            crossfit=post_forward(fit['reference_crossfit'],features['post_tables']).reshape(target.shape)
            case=dict(index=row['index'],content=content_path,gt=row['gt'],reference_original=other,generated_reference=p.name,sha256=digest(p),coefficients=coef.tolist(),target_approximation_mae=float(np.abs(approximation.astype(float)-truth).mean()),target_crossfit_mae=float(np.abs(crossfit.astype(float)-truth).mean()),native_max_code_difference=difference,generated_endpoint_fraction=float(((generated==0)|(generated==255)).mean()),parent_sha256={k:lookup[k]['sha256'] for k in [content_path,row['gt'],other]})
            report['cases'].append(case)
            canvas=Image.new('RGB',(768,550),'#222');draw=ImageDraw.Draw(canvas)
            for i,(label,image) in enumerate([('Training target',target),('Actual training GT',truth),('Target map approximation',approximation),('Other training original',reference_original),('Generated reference (not author)',generated),('Target GT crossfit diagnostic',crossfit)]):
                px=i%3*256;py=i//3*275;draw.text((px+3,py+3),label,fill='white');canvas.paste(Image.fromarray(image),(px,py+19))
            canvas.save(out/f'{row["index"]}.jpg',quality=95)
        del features
        (out/'progress.json').write_text(json.dumps(dict(groups=number+1,total_groups=256,cases=len(report['cases']))),encoding='utf-8')
        if number==0 or (number+1)%32==0:print(json.dumps(dict(groups=number+1,cases=len(report['cases']))),flush=True)
    if len(report['cases'])!=1024 or {r['index'] for r in report['cases']}!={r['index'] for r in train} or digest(checkpoint)!=START_SHA or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256']:raise ValueError('Incomplete or changed generation')
    report.update(finished=True,wall_seconds=time.perf_counter()-started,summary={name:dict(mean=float(np.mean([r[name] for r in report['cases']])),maximum=float(max(r[name] for r in report['cases']))) for name in ['target_approximation_mae','target_crossfit_mae','generated_endpoint_fraction']})
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,summary=report['summary'],wall_seconds=report['wall_seconds'])),flush=True)

if __name__=='__main__':main()
