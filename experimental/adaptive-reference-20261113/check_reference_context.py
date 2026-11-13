"""Development-only context ablations with frozen weights and correct histograms."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from training_checks import finite_output
from reference_context import enable_reference_context,reference_forward


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';run=base/'context-appearance-v1';data=base/'cohort-v2'
    record=json.loads((run/'results.json').read_text(encoding='utf-8'))
    if not record['finished']:raise ValueError('Training incomplete')
    epoch=record['best_epoch'];checkpoint=run/f'epoch-{epoch}.pth'
    expected=record['initial_checkpoint_sha256'] if epoch==16 else next(r['checkpoint_sha256'] for r in record['epochs'] if r['epoch']==epoch)
    if digest(checkpoint)!=expected:raise ValueError('Changed checkpoint')
    cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'));prep=json.loads((data/'srgb-v1/manifest.json').read_text(encoding='utf-8'))
    if digest(data/'cohort.json')!=record['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=record['prepared_manifest_sha256']:raise ValueError('Changed data')
    model,api,_=author_model();enable_reference_context(model)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite weights')
    model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
    out=run/'diagnostics-v1';out.mkdir(exist_ok=False)
    for name in ['check_reference_context.py','reference_context.py']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    lookup={e['original_path']:e for e in prep['files']}
    def read(name):
        e=lookup[name];path=data/e['prepared_path']
        if digest(path)!=e['sha256']:raise ValueError('Changed prepared image')
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    def tensor(rgb):return (torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255).cuda()
    def pixels(raw):return np.rint(finite_output(raw,(1,3,256,256))[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    samples=[(r,*[read(r[k]) for k in ['content','reference','gt']]) for r in cohort['groups']['validation']];rows=[];layout=[]
    with torch.inference_mode():
        reference_features=[model.img_projection(api['TF'].resize(tensor(s),(64,64),antialias=True)) for _,_,s,_ in samples]
        for i,(row,c,s,gt) in enumerate(samples):
            ct,st=tensor(c),tensor(s);images={'full':pixels(reference_forward(model,api,ct,st)[0])}
            images['no_context']=pixels(reference_forward(model,api,ct,st,True)[0])
            for label,replacement in [('zero_reference',torch.zeros_like(reference_features[i])),('wrong_reference',reference_features[(i+1)%len(samples)])]:
                handle=model.reference_adapter.register_forward_pre_hook(lambda module,args,replacement=replacement:(args[0],replacement))
                try:images[label]=pixels(reference_forward(model,api,ct,st)[0])
                finally:handle.remove()
            with Image.open(run/f'epoch-{epoch}/{row["index"]}.png') as im:saved=np.array(im.convert('RGB'))
            delta=int(np.abs(images['full'].astype(int)-saved.astype(int)).max())
            if delta>1:raise ValueError('Unexpected reproduction drift')
            rows.append(dict(index=row['index'],scores={k:float(np.abs(v.astype(float)-gt).mean()) for k,v in images.items()},
                reference_response_mae=float(np.abs(images['full'].astype(float)-images['wrong_reference']).mean()),
                reproduction_max_rgb8_delta=delta,reproduction_changed_values=int(np.count_nonzero(images['full']!=saved))))
            canvas=Image.new('RGB',(768,550),'#222');draw=ImageDraw.Draw(canvas)
            for n,(name,rgb) in enumerate([('Input',c),('Reference',s),('GT',gt),('Full context',images['full']),('No extra context',images['no_context']),('Wrong appearance only',images['wrong_reference'])]):
                x=n%3*256;y=n//3*275;draw.text((x+3,y+3),name,fill='white');canvas.paste(Image.fromarray(rgb),(x,y+19))
            canvas.save(out/f'{row["index"]}.jpg',quality=95)
            if i<4:
                rng=np.random.default_rng(20261116+i);shuffled=s.reshape(-1,3)[rng.permutation(256*256)].reshape(s.shape)
                other=tensor(shuffled);h1=api['batch_histogram_1D'](api['srgb_to_ycbcr'](st));h2=api['batch_histogram_1D'](api['srgb_to_ycbcr'](other))
                if not torch.equal(h1,h2):raise ValueError('Permutation changed marginal features')
                full=pixels(reference_forward(model,api,ct,other)[0]);none=pixels(reference_forward(model,api,ct,other,True)[0])
                layout.append(dict(index=row['index'],full_response_mae=float(np.abs(images['full'].astype(float)-full).mean()),
                    no_context_response_mae=float(np.abs(images['no_context'].astype(float)-none).mean())))
    report=dict(scope='64 previously used development pairs; cyclic wrong-reference appearance with true marginal histograms; no semantic or independent acceptance',
        checkpoint_sha256=digest(checkpoint),source_sha256={n:digest(out/n) for n in ['check_reference_context.py','reference_context.py']},cases=rows,layout_controls=layout,
        means={k:float(np.mean([r['scores'][k] for r in rows])) for k in ['full','no_context','zero_reference','wrong_reference']},
        mean_appearance_reference_response=float(np.mean([r['reference_response_mae'] for r in rows])),
        exact_reproduction=all(r['reproduction_changed_values']==0 for r in rows),maximum_reproduction_delta=max(r['reproduction_max_rgb8_delta'] for r in rows))
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:report[k] for k in ['means','mean_appearance_reference_response','layout_controls','exact_reproduction','maximum_reproduction_delta']}),flush=True)


if __name__=='__main__':main()
