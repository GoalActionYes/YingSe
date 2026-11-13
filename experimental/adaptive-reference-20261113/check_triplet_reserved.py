"""First frozen-checkpoint reserved evaluation, truth downloaded after predictions."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request,urlopen
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest,fetch
from triplet_color import to_srgb
from train_triplet_pilot import author_model
from training_checks import finite_output
from reference_color_app.pipeline import analyze
from reference_color_app.lut import apply_mapping

SHA='d08f07a41d4056bad26b480fd1d942197839b4c3f6e0145e796687b8bef1ff14'


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-v1';training=base/'training-v1'
    checkpoint=training/'epoch-8.pth'
    if digest(checkpoint)!=SHA:raise ValueError('Frozen checkpoint changed')
    cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'));selected=cohort['groups']['reserved'];revision=cohort['revision']
    out=base/'reserved-v2';out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    originals_root=out/'originals';originals_root.mkdir();originals_root=originals_root.resolve()
    for name in ['check_triplet_reserved.py','train_triplet_pilot.py','training_checks.py','triplet_color.py','triplet-reserved-protocol.md']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='First use of reserved32; family-disjoint academic data, not subject/video/product acceptance',
        checkpoint_sha256=SHA,cohort_sha256=digest(data/'cohort.json'),dataset_revision=revision,
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=[],assets=[],gt_downloaded=False)
    (out/'frozen.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    seen={e['pixel_sha256'] for e in json.loads((data/'srgb-v1/manifest.json').read_text(encoding='utf-8'))['files']};duplicates=[]
    gt_paths={r['gt'] for r in selected};source_paths={r[k] for r in selected for k in ['content','reference']}
    if source_paths & gt_paths:raise ValueError('Source path reveals a reserved GT')
    def prepare(paths,phase):
        req=Request(f'https://huggingface.co/api/datasets/ToneStyle/TST100K/paths-info/{revision}',data=json.dumps({'paths':sorted(paths)}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=60) as response:entries=json.load(response)
        if {e['path'] for e in entries}!=set(paths):raise ValueError('Incomplete file metadata')
        # Resolve/create parents once before concurrent Windows filesystem work.
        for entry in entries:
            relative=Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe dataset relative path')
            (originals_root/relative).parent.mkdir(parents=True,exist_ok=True)
        def one(entry):
            relative=entry['path'];original=originals_root/relative;prepared=out/'srgb'/(relative+'.png')
            if not original.resolve().is_relative_to(originals_root):raise ValueError(f'Unsafe dataset path: {original.resolve()} versus {originals_root}')
            sha=entry.get('lfs',{}).get('oid');fetch(f'https://huggingface.co/datasets/ToneStyle/TST100K/resolve/{revision}/{quote(relative)}',original,sha,entry['size'])
            if not sha:
                raw=original.read_bytes()
                if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=entry['oid']:raise ValueError('Git blob mismatch')
            with Image.open(original) as image:rgb,info=to_srgb(image)
            prepared.parent.mkdir(parents=True,exist_ok=True);rgb.save(prepared,icc_profile=rgb.info['icc_profile']);array=np.array(rgb)
            pixel_hash=hashlib.sha256(np.array(array.shape,dtype='<i8').tobytes()+array.tobytes()).hexdigest()
            return dict(path=relative,phase=phase,original_sha256=digest(original),prepared_path=str(prepared.relative_to(out)),sha256=digest(prepared),pixel_sha256=pixel_hash,preparation=info)
        with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,entries))
        for r in records:
            if r['pixel_sha256'] in seen:duplicates.append(r['path'])
        report['assets'].extend(records)
        return {r['path']:out/r['prepared_path'] for r in records}
    lookup=prepare(source_paths,'content-reference')
    (out/'inputs.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    model,api,_=author_model();model.eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    def array(path):
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    def tensor(rgb):return torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda().float()/255
    @torch.inference_mode()
    def predict(target,reference):
        c,s=tensor(target),tensor(reference);hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s));start=time.perf_counter()
        raw=finite_output(model(c,hc,hs)[0],c.shape);torch.cuda.synchronize();seconds=time.perf_counter()-start
        rgb=np.rint(raw[0].clamp(0,1).cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
        return rgb,seconds
    # First frozen fine-tuned and current predictions; neither receives GT.
    for item in selected:
        row=dict(index=item['index'],triplet=item,outputs={});report['cases'].append(row)
        try:
            target,reference=array(lookup[item['content']]),array(lookup[item['reference']]);candidate,seconds=predict(target,reference)
            current=apply_mapping(target,analyze(lookup[item['content']],lookup[item['reference']]),1)
            for label,rgb in [('input',target),('candidate',candidate),('current',current)]:
                path=out/f'{item["index"]}-{label}.png';Image.fromarray(rgb).save(path);row['outputs'][label]=dict(path=path.name,sha256=digest(path))
            row['model_seconds']=seconds
        except Exception as exc:row['failure']=dict(type=type(exc).__name__,message=str(exc))
        print(json.dumps(dict(index=item['index'],failure=row.get('failure'))),flush=True)
    # Restore the exact author checkpoint for the second, untrained control.
    original,_,_=author_model();model.load_state_dict(original.state_dict(),strict=True);del original
    for row in report['cases']:
        if 'failure' in row:continue
        item=row['triplet'];before,_=predict(array(lookup[item['content']]),array(lookup[item['reference']]));path=out/f'{item["index"]}-before.png';Image.fromarray(before).save(path);row['outputs']['before']=dict(path=path.name,sha256=digest(path))
    (out/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Truth only becomes accessible here, after all method outputs are frozen.
    lookup.update(prepare(gt_paths,'ground-truth-after-predictions'));report['gt_downloaded']=True
    report['predictions_sha256']=digest(out/'predictions.json')
    for row in report['cases']:
        if 'failure' in row:continue
        item=row['triplet'];truth=array(lookup[item['gt']]);images={}
        with Image.open(lookup[item['content']]) as c,Image.open(lookup[item['gt']]) as g:
            if c.size!=g.size:raise ValueError('Content/GT native shape mismatch')
        for label,entry in row['outputs'].items():
            if digest(out/entry['path'])!=entry['sha256']:raise ValueError('Prediction changed')
            with Image.open(out/entry['path']) as im:images[label]=np.array(im)
        row['scores']={k:float(np.abs(v.astype(float)-truth).mean()) for k,v in images.items()}
        row['gradient_mae']={k:float((np.abs(np.diff(v.astype(float),axis=0)-np.diff(truth.astype(float),axis=0)).mean()+np.abs(np.diff(v.astype(float),axis=1)-np.diff(truth.astype(float),axis=1)).mean())/2) for k,v in images.items()}
        panel=Image.new('RGB',(900,630),'#222222');draw=ImageDraw.Draw(panel)
        for n,(label,rgb) in enumerate([('Input',images['input']),('Reference',array(lookup[item['reference']])),('GT',truth),('Current',images['current']),('Before',images['before']),('Candidate',images['candidate'])]):
            x,y=n%3*300,n//3*310;panel.paste(Image.fromarray(rgb),(x,y+25));draw.text((x+5,y+5),label,fill='white')
        panel.save(out/f'{item["index"]}-comparison.jpg',quality=95)
    valid=[r for r in report['cases'] if 'scores' in r];baseline=np.array([r['scores']['current'] for r in valid]);summary={}
    for name in ['input','current','before','candidate']:
        actual=np.array([r['scores'][name] for r in valid]);summary[name]=dict(scored=len(valid),total=32,failed=32-len(valid),mean=float(actual.mean()),median=float(np.median(actual)),improved=int((actual<baseline-.1).sum()),regressed=int((actual>baseline+.1).sum()),severe=int(((actual>baseline+5)&(actual>baseline*1.25)).sum()),gradient_mae=float(np.mean([r['gradient_mae'][name] for r in valid])))
    report.update(summary=summary,exact_pixel_overlap_with_training_validation=duplicates,independence_exact_duplicate_gate_passed=not duplicates,
        reserved_now_consumed=True,all_predictions_before_gt_download=True,product_integrated=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(summary=summary,duplicates=duplicates)),flush=True)


if __name__=='__main__':main()
