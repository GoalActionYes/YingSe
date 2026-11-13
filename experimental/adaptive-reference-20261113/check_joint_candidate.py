"""Development-only ablation and direction checks for the frozen joint pilot."""
import json
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from training_checks import finite_output
from joint_features import enable_joint_input,joint_forward,conditional_color


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='diagnostics-v4');args=parser.parse_args()
    if Path(args.run).name!=args.run or args.run in ('','.','..'):raise ValueError('Expected output directory name')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';run=base/'training-joint-v1'
    record=json.loads((run/'results.json').read_text(encoding='utf-8'))
    if not record['finished']:raise ValueError('Training is not complete')
    epoch=record['best_epoch'];checkpoint=run/f'epoch-{epoch}.pth'
    expected=record.get('initial_checkpoint_sha256') if epoch==8 else next(r['checkpoint_sha256'] for r in record['epochs'] if r['epoch']==epoch)
    if expected is None or digest(checkpoint)!=expected:raise ValueError('Changed or unverified selected checkpoint')
    data=base/'cohort-v1';cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'))
    prep=json.loads((data/'srgb-v1/manifest.json').read_text(encoding='utf-8'))
    if digest(data/'cohort.json')!=record['cohort_sha256'] or digest(data/'srgb-v1/manifest.json')!=record['prepared_manifest_sha256']:raise ValueError('Changed data manifest')
    lookup={e['original_path']:e for e in prep['files']};model,api,_=author_model();enable_joint_input(model)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite selected weights')
    model.load_state_dict(state,strict=True);model.eval().requires_grad_(False);torch.set_num_threads(4)
    out=run/args.run;out.mkdir(exist_ok=False);rows=[]
    def read(name):
        entry=lookup[name];path=data/entry['prepared_path']
        if digest(path)!=entry['sha256']:raise ValueError('Changed prepared file')
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    # Match training's CPU float normalization before the CUDA transfer.
    def tensor(rgb):return (torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255).cuda()
    def pixels(raw):return np.rint(finite_output(raw,(1,3,256,256))[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
    with torch.inference_mode():
        for row in cohort['groups']['validation']:
            content,reference,gt=[read(row[role]) for role in ['content','reference','gt']];c,s=tensor(content),tensor(reference)
            start=time.perf_counter();full=pixels(joint_forward(model,api,c,s)[0]);elapsed=time.perf_counter()-start
            with Image.open(run/f'epoch-{epoch}/{row["index"]}.png') as im:saved=np.array(im.convert('RGB'))
            reproduction_delta=int(np.abs(full.astype(int)-saved.astype(int)).max())
            if reproduction_delta>1:raise ValueError('Diagnostic differs from frozen output by more than one code value')
            ablated=pixels(joint_forward(model,api,c,s,disable_joint=True)[0])
            rows.append(dict(index=row['index'],full_mae=float(np.abs(full.astype(float)-gt).mean()),no_joint_mae=float(np.abs(ablated.astype(float)-gt).mean()),
                response_mae=float(np.abs(full.astype(float)-ablated).mean()),seconds=elapsed,
                reproduction_max_rgb8_delta=reproduction_delta,reproduction_changed_values=int(np.count_nonzero(full!=saved))))
            if len(rows)==1:
                yc=api['srgb_to_ycbcr'](s);features=[conditional_color(yc).cpu().numpy() for _ in range(12)]
                repeat_feature_delta=max(float(np.abs(value-features[0]).max()) for value in features)
            canvas=Image.new('RGB',(768,550),'#222');draw=ImageDraw.Draw(canvas)
            with Image.open(base/f'training-v1/epoch-8/{row["index"]}.png') as im:original=np.array(im.convert('RGB'))
            with Image.open(base/f'training-v2/epoch-16/{row["index"]}.png') as im:control=np.array(im.convert('RGB'))
            for i,(label,rgb) in enumerate([('Input',content),('Reference',reference),('GT',gt),('Original epoch8',original),('Continued control',control),('Joint candidate',full)]):
                x=i%3*256;y=i//3*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(rgb),(x,y+19))
            canvas.save(out/f'{row["index"]}-comparison.jpg',quality=95)
        a=np.empty((256,256,3),np.uint8);b=np.empty_like(a)
        a[:128]=[63,92,118];a[128:]=[193,163,139];b[:128]=[113,83,60];b[128:]=[143,172,198]
        target=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],256,axis=0);target=np.repeat(target,3,axis=2)
        c,ta,tb=tensor(target),tensor(a),tensor(b)
        ya,yb=api['srgb_to_ycbcr'](ta),api['srgb_to_ycbcr'](tb)
        ha,hb=[api['batch_histogram_1D'](v) for v in [ya,yb]]
        if not torch.equal(ha,hb):raise ValueError('Invalid equal-marginal counterexample')
        oa,ga=joint_forward(model,api,c,ta);ob,gb=joint_forward(model,api,c,tb)
        pa,pb=pixels(oa),pixels(ob)
        output_a,output_b=api['srgb_to_ycbcr'](tensor(pa)),api['srgb_to_ycbcr'](tensor(pb));difference=(output_a-output_b)[0].cpu().permute(1,2,0).numpy()
        dark=difference[:,0:80].mean((0,1));light=difference[:,176:].mean((0,1))
        directions=[bool(dark[1]>0),bool(dark[2]<0),bool(light[1]<0),bool(light[2]>0)]
        counter=dict(marginal_histogram_delta=float((ha-hb).abs().max()),joint_feature_delta=float((conditional_color(ya)-conditional_color(yb)).abs().max()),
            grid_delta=float((ga-gb).abs().max()),output_changed_values=int(np.count_nonzero(pa!=pb)),
            dark_a_minus_b_ycbcr=dark.tolist(),light_a_minus_b_ycbcr=light.tolist(),expected_direction_checks=directions,correct_directions=sum(directions))
        canvas=Image.new('RGB',(512,550),'#222');draw=ImageDraw.Draw(canvas)
        for i,(label,rgb) in enumerate([('A cool-dark / warm-light',a),('B warm-dark / cool-light',b),('Output A',pa),('Output B',pb)]):
            x=i%2*256;y=i//2*275;draw.text((x+3,y+3),label,fill='white');canvas.paste(Image.fromarray(rgb),(x,y+19))
        canvas.save(out/'counterexample.png')
    report=dict(scope='Development ablation/direction diagnostic only; no independent acceptance or app integration',
        reproduces_selected_validation_bytes=all(r['reproduction_changed_values']==0 for r in rows),tf32=False,
        maximum_reproduction_rgb8_delta=max(r['reproduction_max_rgb8_delta'] for r in rows),repeat_joint_feature_delta=repeat_feature_delta,
        checkpoint_sha256=digest(checkpoint),script_sha256=digest(Path(__file__)),cases=rows,counterexample=counter,
        means={key:float(np.mean([r[key] for r in rows])) for key in ['full_mae','no_joint_mae','response_mae']})
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(means=report['means'],counterexample=counter)),flush=True)


if __name__=='__main__':main()
