"""Training-domain content sensitivity probe; never used to choose/promote models."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from capacity_audit import render_coefficients
from pairwise_context import enable_pairwise_context
from appearance_pairwise import enable_appearance_context,appearance_forward


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-multiref-v1';generated_dir=base/'cohort-consistent-reference-v1';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    cohort_path=data/'cohort.json';prep_path=data/'srgb-v1/manifest.json';cohort,prep=load(cohort_path),load(prep_path);generated=load(generated_dir/'results.json');rows=cohort['groups']['train'];lookup={r['original_path']:r for r in prep['files']};gen={r['index']:r for r in generated['cases']};originals=sorted({r['content'] for r in rows});checkpoint=base/'multiref-control-v1/epoch-6.pth'
    if len(rows)!=1024 or len(originals)!=256 or not generated['finished'] or generated['cohort_sha256']!=digest(cohort_path) or generated['prepared_manifest_sha256']!=digest(prep_path) or digest(checkpoint)!=generated['checkpoint_sha256']:raise ValueError('Changed probe inputs')
    for name in ['capacity_audit.py','train_triplet_pilot.py']:
        if digest(Path(__file__).with_name(name))!=generated['source_sha256'][name]:raise ValueError('Changed frozen-renderer implementation')
    paths={'C00':base/'appearance-pairwise-replica-control-v1','C10':base/'appearance-pairwise-replica-v1','C01':base/'appearance-invariance-C01-v2','C11':base/'appearance-invariance-C11-v2'};records={name:load(p/'results.json') for name,p in paths.items()}
    if any(not r['finished'] or r['cohort_sha256']!=digest(cohort_path) or r['prepared_manifest_sha256']!=digest(prep_path) for r in records.values()):raise ValueError('Incomplete or changed study')
    if any(records[n]['generation_manifest_sha256']!=digest(generated_dir/'results.json') for n in ['C01','C11']) or any(r['start_checkpoint_sha256']!=generated['checkpoint_sha256'] for r in records.values()):raise ValueError('Generation is not the frozen study input')
    out=base/'appearance-invariance-content-probe-v1';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    for name in ['diagnose_reference_content.py','appearance-invariance-diagnostic-protocol.md','capacity_audit.py','train_triplet_pilot.py','appearance_pairwise.py','pairwise_context.py','reference_context.py','training_checks.py','independent_coco.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='Post-selection training1024 same-constructed-transform content probe, not independent or release qualification',cohort_sha256=digest(cohort_path),prepared_manifest_sha256=digest(prep_path),generation_manifest_sha256=digest(generated_dir/'results.json'),source_sha256={p.name:digest(p) for p in frozen.iterdir()},checkpoints={name:dict(epoch=r['best_epoch'],sha256=r['selected_checkpoint_sha256'],results_sha256=digest(paths[name]/'results.json')) for name,r in records.items()},reference_b=[],cells={},finished=False,app_promoted=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def read(relative):
        item=lookup[relative];p=data/item['prepared_path']
        if digest(p)!=item['sha256']:raise ValueError('Changed prepared training input')
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    renderer,api,_=author_model();renderer.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);renderer.eval().requires_grad_(False)
    for i,row in enumerate(rows):
        item=gen[row['index']]
        if item['content']!=row['content'] or item['gt']!=row['gt'] or item['reference_original'] not in originals or any(lookup[p]['sha256']!=sha or digest(data/lookup[p]['prepared_path'])!=sha for p,sha in item['parent_sha256'].items()):raise ValueError('Changed original-generation parents')
        choices=[p for p in originals if p not in [row['content'],item['reference_original']]];key=hashlib.sha256(('yingse-reference-invariance-diagnostic-v1:'+str(row['index'])).encode()).digest();source=choices[int.from_bytes(key[:8],'big')%len(choices)]
        if digest(generated_dir/item['generated_reference'])!=item['sha256']:raise ValueError('Changed referenceA')
        coefficients=np.asarray(item['coefficients'])
        if coefficients.shape!=(32,3):raise ValueError('Expected the same frozen global transform')
        actual=render_coefficients(renderer,api,read(source),coefficients);p=out/f'{row["index"]}-referenceB.png';Image.fromarray(actual).save(p)
        report['reference_b'].append(dict(index=row['index'],source=source,parent_sha256=lookup[source]['sha256'],path=p.name,sha256=digest(p),endpoint_channel_fraction=float(((actual==0)|(actual==255)).mean())))
        if (i+1)%256==0:print(json.dumps(dict(generated_reference_b=i+1)),flush=True)
    del renderer
    model,api,_=author_model();enable_pairwise_context(model);enable_appearance_context(model);model.eval().requires_grad_(False)
    tensor=lambda rgb:(torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255).cuda();pixels=lambda x:np.rint(x[0].detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    def png(p):
        with Image.open(p) as im:return np.array(im.convert('RGB'))
    for name,path in paths.items():
        record=records[name];p=path/f'epoch-{record["best_epoch"]}.pth'
        if digest(p)!=record['selected_checkpoint_sha256']:raise ValueError('Selected model changed')
        model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True);cases=[]
        with torch.inference_mode():
            for i,(row,b) in enumerate(zip(rows,report['reference_b'])):
                item=gen[row['index']]
                if row['index']!=b['index'] or digest(out/b['path'])!=b['sha256'] or digest(generated_dir/item['generated_reference'])!=item['sha256']:raise ValueError('Reference order/hash differs')
                c=tensor(read(row['content']));truth=read(row['gt']);a=pixels(appearance_forward(model,api,c,tensor(png(generated_dir/item['generated_reference'])),disable_appearance=name[1]=='0')[0]);other=pixels(appearance_forward(model,api,c,tensor(png(out/b['path'])),disable_appearance=name[1]=='0')[0])
                ea=float(np.abs(a.astype(float)-truth).mean());eb=float(np.abs(other.astype(float)-truth).mean());cases.append(dict(index=row['index'],reference_content_sensitivity=float(np.abs(a.astype(float)-other).mean()),mae_reference_a=ea,mae_reference_b=eb,mean_mae=(ea+eb)/2,output_a_pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest(),output_b_pixel_sha256=hashlib.sha256(other.tobytes()).hexdigest()))
                if (i+1)%256==0:print(json.dumps(dict(cell=name,scored=i+1)),flush=True)
        summary={key:dict(mean=float(np.mean([r[key] for r in cases])),median=float(np.median([r[key] for r in cases]))) for key in ['reference_content_sensitivity','mae_reference_a','mae_reference_b','mean_mae']}
        report['cells'][name]=dict(cases=cases,summary=summary);(out/'progress.json').write_text(json.dumps(dict(cells=list(report['cells']))),encoding='utf-8');print(json.dumps(dict(cell=name,summary=summary)),flush=True)
    if any(digest(paths[name]/'results.json')!=snapshot['results_sha256'] for name,snapshot in report['checkpoints'].items()) or digest(cohort_path)!=report['cohort_sha256'] or digest(prep_path)!=report['prepared_manifest_sha256'] or digest(generated_dir/'results.json')!=report['generation_manifest_sha256'] or any(digest(Path(__file__).with_name(name))!=sha for name,sha in report['source_sha256'].items()):raise ValueError('Frozen diagnostic evidence changed')
    if any(digest(paths[name]/f'epoch-{snapshot["epoch"]}.pth')!=snapshot['sha256'] for name,snapshot in report['checkpoints'].items()):raise ValueError('Selected checkpoint changed during diagnostic')
    report['finished']=True;(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(finished=True,summaries={name:r['summary'] for name,r in report['cells'].items()})),flush=True)

if __name__=='__main__':main()
