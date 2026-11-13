"""Check reference dependence against no-op and shuffled-reference controls."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from independent_coco import ROOT,digest
from train_triplet_pilot import author_model
from training_checks import finite_output


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=base/'cohort-v1';run=base/'training-v1'
    report=json.loads((run/'results.json').read_text(encoding='utf-8'));epoch=report['best_epoch'];cohort=json.loads((data/'cohort.json').read_text(encoding='utf-8'))
    manifest=json.loads((data/'srgb-v1/manifest.json').read_text(encoding='utf-8'));lookup={e['original_path']:data/e['prepared_path'] for e in manifest['files']}
    model,api,_=author_model();checkpoint=run/f'epoch-{epoch}.pth';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval()
    def tensor(path):
        with Image.open(path) as im:array=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
        return torch.from_numpy(array.copy()).permute(2,0,1)[None].cuda().float()/255
    samples=[(r,tensor(lookup[r['content']]),tensor(lookup[r['reference']]),tensor(lookup[r['gt']])) for r in cohort['groups']['validation']]
    output=run/'conditioning';output.mkdir(exist_ok=False);rows=[]
    with torch.inference_mode():
        histograms=[api['batch_histogram_1D'](api['srgb_to_ycbcr'](s)) for _,_,s,_ in samples];mean_hist=torch.stack(histograms).mean(0)
        for i,(row,c,s,gt) in enumerate(samples):
            source_hist=api['batch_histogram_1D'](api['srgb_to_ycbcr'](c));predictions={}
            for label,hist in [('correct',histograms[i]),('shuffled',histograms[(i+1)%len(samples)]),('mean_hist',mean_hist)]:
                raw=finite_output(model(c,source_hist,hist)[0],c.shape);predictions[label]=raw.clamp(0,1)
            def array(x):return np.rint(x[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
            truth=array(gt);rgb={k:array(v) for k,v in predictions.items()};rgb['input']=array(c)
            scores={k:float(np.abs(v.astype(float)-truth).mean()) for k,v in rgb.items()}
            rows.append(dict(index=row['index'],scores=scores,reference_response_mae=float(np.abs(rgb['correct'].astype(float)-rgb['shuffled']).mean())))
            for label in ['correct','shuffled','mean_hist']:Image.fromarray(rgb[label]).save(output/f'{row["index"]}-{label}.png')
    result=dict(scope='32 development validation conditioning controls, not independent acceptance',checkpoint_sha256=digest(checkpoint),
        source_sha256=digest(Path(__file__)),cases=rows,means={k:float(np.mean([r['scores'][k] for r in rows])) for k in ['input','correct','shuffled','mean_hist']},
        shuffled_reference_policy='Cyclic shift +1, no GT-guided choice',product_integrated=False)
    (output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result['means']),flush=True)


if __name__=='__main__':main()
