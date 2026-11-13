"""First frozen-checkpoint reserved evaluation, truth downloaded after predictions."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import argparse
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request,urlopen
from urllib.error import URLError,HTTPError
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
from training_gates import reserved_gate
from reserved_checks import verify_frozen_predictions

SHA='d08f07a41d4056bad26b480fd1d942197839b4c3f6e0145e796687b8bef1ff14'


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group();mode.add_argument('--expanded',action='store_true');mode.add_argument('--multiref',action='store_true');args=parser.parse_args()
    checked=args.expanded or args.multiref
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/('cohort-v2' if args.expanded else 'cohort-v1');training=base/('training-expanded-control-v1' if args.expanded else 'training-v1')
    chosen_sha='17be34dfff3e778761c92e1442e75e8ce3bc2b0bb033543171eae40bc8e0b795' if args.expanded else SHA
    checkpoint=training/('epoch-16.pth' if args.expanded else 'epoch-8.pth')
    if args.multiref:
        data=base/'cohort-multiref-v1';training=base/'multiref-control-v1';checkpoint=training/'epoch-6.pth'
        chosen_sha='d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9'
    if digest(checkpoint)!=chosen_sha or digest(base/'training-v1/epoch-8.pth')!=SHA:raise ValueError('Frozen checkpoint changed')
    if checked:
        development=json.loads((training/'results.json').read_text(encoding='utf-8'))
        if not development['finished'] or not development['development_gate']['proceed_to_independent_evaluation'] or development['best_epoch']!=(6 if args.multiref else 16) or development['cohort_sha256']!=digest(data/'cohort.json') or development['prepared_manifest_sha256']!=digest(data/'srgb-v1/manifest.json'):raise ValueError('Development candidate not eligible')
    cohort_path=(base/'cohort-multiref-reserved-v1' if args.multiref else data)/'cohort.json'
    cohort=json.loads(cohort_path.read_text(encoding='utf-8'));selected=cohort['groups']['reserved'];revision=cohort['revision']
    if len(selected)!=32 or len({r['index'] for r in selected})!=32:raise ValueError('Incomplete reserved cohort')
    out=base/('reserved-multiref-v1' if args.multiref else 'reserved-expanded-v1' if args.expanded else 'reserved-v2');out.mkdir(exist_ok=False);frozen=out/'frozen-source';frozen.mkdir()
    originals_root=out/'originals';originals_root.mkdir();originals_root=originals_root.resolve()
    names=['check_triplet_reserved.py','train_triplet_pilot.py','training_checks.py','triplet_color.py','triplet-reserved-protocol.md','training_gates.py']
    if args.expanded:names+=['expanded-reserved-protocol.md']
    if args.multiref:names+=['multiref-independent-protocol.md','prepare_fresh_reserved.py','prepare_triplet_cohort.py']
    if checked:names+=['reserved_checks.py']
    for name in names:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    report=dict(scope='First use of fresh multiref reserved32; pretrained/subject independence unverified' if args.multiref else 'First use of cohort-v2 reserved32; independent of this finetuning, pretrained overlap unverified' if args.expanded else 'First use of reserved32; family-disjoint academic data, not subject/video/product acceptance',
        checkpoint_sha256=chosen_sha,cohort_sha256=digest(cohort_path),dataset_revision=revision,
        before_control='Current App epoch8' if checked else 'Original author model',
        source_sha256={p.name:digest(p) for p in frozen.iterdir()},cases=[],assets=[],gt_downloaded=False)
    if args.multiref:
        report.update(development_results_sha256=digest(training/'results.json'),development_cohort_sha256=digest(data/'cohort.json'),
            global_source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'reference_color_app').rglob('*.py')},phase_events=dict(started_ns=time.time_ns()))
    (out/'frozen.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    seen={e['pixel_sha256'] for e in json.loads((data/'srgb-v1/manifest.json').read_text(encoding='utf-8'))['files']};duplicates=[]
    if args.expanded:
        consumed=json.loads((base/'reserved-v2/results.json').read_text(encoding='utf-8'));seen.update(e['pixel_sha256'] for e in consumed['assets'])
    if args.multiref:
        for name in ['cohort-v1','cohort-v2']:
            seen.update(e['pixel_sha256'] for e in json.loads((base/name/'srgb-v1/manifest.json').read_text(encoding='utf-8'))['files'])
        for name in ['reserved-v2','reserved-expanded-v1']:
            seen.update(e['pixel_sha256'] for e in json.loads((base/name/'results.json').read_text(encoding='utf-8'))['assets'])
    gt_paths={r['gt'] for r in selected};source_paths={r[k] for r in selected for k in ['content','reference']}
    if source_paths & gt_paths:raise ValueError('Source path reveals a reserved GT')
    def prepare(paths,phase):
        req=Request(f'https://huggingface.co/api/datasets/ToneStyle/TST100K/paths-info/{revision}',data=json.dumps({'paths':sorted(paths)}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=60) as response:entries=json.load(response)
        if {e['path'] for e in entries}!=set(paths) or len(entries)!=len(paths):raise ValueError('Incomplete file metadata')
        if checked and (sum(e['size'] for e in entries)+sum(e.get('original_bytes',0) for e in report['assets'])>256*1024*1024 or any(e['size']>20*1024*1024 for e in entries)):raise ValueError('Reserved assets exceed frozen resource bounds')
        # Resolve/create parents once before concurrent Windows filesystem work.
        for entry in entries:
            relative=Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe dataset relative path')
            (originals_root/relative).parent.mkdir(parents=True,exist_ok=True)
        def one(entry):
            relative=entry['path'];original=originals_root/relative;prepared=out/'srgb'/(relative+'.png')
            if not original.resolve().is_relative_to(originals_root):raise ValueError(f'Unsafe dataset path: {original.resolve()} versus {originals_root}')
            sha=entry.get('lfs',{}).get('oid')
            for attempt in range(3):
                try:
                    fetch(f'https://huggingface.co/datasets/ToneStyle/TST100K/resolve/{revision}/{quote(relative)}',original,sha,entry['size']);break
                except (URLError,TimeoutError) as error:
                    if attempt==2 or isinstance(error,HTTPError) and error.code not in (429,500,502,503,504):raise
                    time.sleep(2**attempt)
            if not sha:
                raw=original.read_bytes()
                if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=entry['oid']:raise ValueError('Git blob mismatch')
            with Image.open(original) as image:rgb,info=to_srgb(image)
            prepared.parent.mkdir(parents=True,exist_ok=True);rgb.save(prepared,icc_profile=rgb.info['icc_profile']);array=np.array(rgb)
            pixel_hash=hashlib.sha256(np.array(array.shape,dtype='<i8').tobytes()+array.tobytes()).hexdigest()
            return dict(path=relative,phase=phase,original_bytes=entry['size'],original_sha256=digest(original),prepared_path=str(prepared.relative_to(out)),sha256=digest(prepared),pixel_sha256=pixel_hash,preparation=info)
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
    def tensor(rgb):
        value=torch.from_numpy(rgb.copy()).permute(2,0,1)[None]
        return (value.float()/255).cuda() if checked else value.cuda().float()/255
    @torch.inference_mode()
    def predict(target,reference):
        c,s=tensor(target),tensor(reference);hc=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));hs=api['batch_histogram_1D'](api['srgb_to_ycbcr'](s));start=time.perf_counter()
        raw,grid=model(c,hc,hs);finite_output(grid,(1,13,8,16,16));raw=finite_output(raw,c.shape);torch.cuda.synchronize();seconds=time.perf_counter()-start
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
    if checked:model.load_state_dict(torch.load(base/'training-v1/epoch-8.pth',map_location='cpu',weights_only=True),strict=True)
    for row in report['cases']:
        if 'failure' in row:continue
        item=row['triplet'];before,_=predict(array(lookup[item['content']]),array(lookup[item['reference']]));path=out/f'{item["index"]}-before.png';Image.fromarray(before).save(path);row['outputs']['before']=dict(path=path.name,sha256=digest(path))
    (out/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if checked:verify_frozen_predictions(out,report['cases'])
    predictions_sha=digest(out/'predictions.json')
    if args.multiref:report['phase_events']['predictions_frozen_ns']=time.time_ns()
    # Truth only becomes accessible here, after all method outputs are frozen.
    lookup.update(prepare(gt_paths,'ground-truth-after-predictions'));report['gt_downloaded']=True
    if args.multiref:report['phase_events']['gt_preparation_finished_ns']=time.time_ns()
    if digest(out/'predictions.json')!=predictions_sha:raise ValueError('Frozen predictions changed during GT preparation')
    report['predictions_sha256']=predictions_sha
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
        for n,(label,rgb) in enumerate([('Input',images['input']),('Reference',array(lookup[item['reference']])),('GT',truth),('Legacy global',images['current']),('Current App8' if checked else 'Author',images['before']),('Candidate',images['candidate'])]):
            x,y=n%3*300,n//3*310;panel.paste(Image.fromarray(rgb),(x,y+25));draw.text((x+5,y+5),label,fill='white')
        panel.save(out/f'{item["index"]}-comparison.jpg',quality=95)
    valid=[r for r in report['cases'] if 'scores' in r];baseline=np.array([r['scores']['current'] for r in valid]);summary={}
    for name in ['input','current','before','candidate']:
        actual=np.array([r['scores'][name] for r in valid]);summary[name]=dict(scored=len(valid),total=32,failed=32-len(valid),mean=float(actual.mean()),median=float(np.median(actual)),improved=int((actual<baseline-.1).sum()),regressed=int((actual>baseline+.1).sum()),severe=int(((actual>baseline+5)&(actual>baseline*1.25)).sum()),gradient_mae=float(np.mean([r['gradient_mae'][name] for r in valid])))
    report.update(summary=summary,exact_pixel_overlap_with_training_validation=duplicates,independence_exact_duplicate_gate_passed=not duplicates,
        reserved_now_consumed=True,all_predictions_before_gt_download=True,product_integrated=False)
    if checked:
        report['duplicate_checked_sets']=['Expanded train/validation','Previously consumed reserved32'] if args.expanded else ['All cohort-v1/v2/multiref train/validation','Both consumed reserved32 sets']
        report['independent_gate']=reserved_gate(report['cases'],duplicates)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(summary=summary,duplicates=duplicates)),flush=True)


if __name__=='__main__':main()
