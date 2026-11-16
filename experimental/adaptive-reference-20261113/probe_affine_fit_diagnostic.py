"""Fixed first16 old training targets, all four styles, GT-assisted feasibility."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine
from affine_fit_diagnostic import fit


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    b=ROOT/'outputs/adaptive-reference-20261113/tst100k';data=b/'cohort-multiref-v1';cache=b/'regional-correction-inputs-v1'
    registry=json.loads((cache/'results.json').read_text());assert registry['finished']
    assert digest(data/'cohort.json')==registry['cohort_sha256'];assert digest(data/'srgb-v1/manifest.json')==registry['prepared_manifest_sha256']
    cohort=json.loads((data/'cohort.json').read_text());lookup={e['original_path']:e for e in json.loads((data/'srgb-v1/manifest.json').read_text())['files']}
    groups=list(dict.fromkeys(x['group_id'] for x in cohort['groups']['train']))[:16]
    rows=[x for x in cohort['groups']['train'] if x['group_id'] in groups];assert len(rows)==64
    records={r['index']:r for r in registry['rows'] if r['split']=='train'};inputs={e['index']:e for e in registry['inputs']}
    out=b/'affine-fit-diagnostic-v1';out.mkdir(exist_ok=False)
    report=dict(scope='GT-assisted constructive affine fit on64 consumed training rows, not prediction or optimal bound',groups=groups,cohort_sha256=registry['cohort_sha256'],source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['affine-fit-diagnostic-protocol.md','affine_fit_diagnostic.py','probe_affine_fit_diagnostic.py','test_affine_fit_diagnostic.py']},cases=[],finished=False,product_integrated=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2));engine=SpatialEngine('multiref-v1');report['model_sha256']=engine.checkpoint_sha
    def read(name):
        e=lookup[name];p=data/e['prepared_path'];assert digest(p)==e['sha256']
        with Image.open(p) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    for i,row in enumerate(rows):
        rgb,ref=read(row['content']),read(row['reference']);raw=engine.prepare(rgb,ref).float_output(rgb)
        m=inputs[records[row['index']]['roles']['content']['feature_index']];assert digest(cache/m['mask_path'])==m['mask_sha256'];prob=np.load(cache/m['mask_path'])
        gt=read(row['gt']);outputs={'baseline':np.rint(raw.clip(0,1)*255).astype(np.uint8)};record=dict(index=row['index'],group_id=row['group_id'],style_index=row['style_index'],methods={})
        for name,regional in [('global',False),('regional',True)]:
            actual,coef,maximum=fit(rgb,prob,raw,gt.astype(float)/255,regional)
            outputs[name]=np.rint(actual.clip(0,1)*255).astype(np.uint8)
            p=out/f"{row['index']}-{name}-coefficients.npy";np.save(p,coef,allow_pickle=False)
            record['methods'][name]=dict(coefficients_sha256=digest(p),maximum_residual_codes=maximum*255)
        for name,pixels in outputs.items():
            p=out/f"{row['index']}-{name}.png";Image.fromarray(pixels).save(p);record['methods'].setdefault(name,{})
            record['methods'][name].update(path=p.name,sha256=digest(p))
        report['cases'].append(record)
        if i<4:
            canvas=Image.new('RGB',(1280,280),'#22252a');draw=ImageDraw.Draw(canvas)
            for col,(name,pixels) in enumerate([('source',rgb),*outputs.items(),('GT',gt)]):canvas.paste(Image.fromarray(pixels),(col*256,24));draw.text((col*256+3,4),name,fill='white')
            canvas.save(out/f'review-{i+1}.png')
    for row,record in zip(rows,report['cases']):
        gt=read(row['gt']).astype(float)
        for name,entry in record['methods'].items():
            p=out/entry['path'];assert digest(p)==entry['sha256']
            with Image.open(p) as im:entry['mae']=float(np.abs(np.array(im).astype(float)-gt).mean())
    summary={}
    baseline=np.array([r['methods']['baseline']['mae'] for r in report['cases']])
    for name in ['global','regional']:
        a=np.array([r['methods'][name]['mae'] for r in report['cases']]);summary[name]=dict(baseline_mean=float(baseline.mean()),mean=float(a.mean()),improvement_percent=float((1-a.mean()/baseline.mean())*100),improved_over_point_one=int((baseline-a>.1).sum()),worsened_over_point_one=int((a-baseline>.1).sum()))
    report['summary']=summary;assert all(digest(Path(__file__).with_name(n))==h for n,h in report['source_sha256'].items());report['finished']=True
    (out/'results.json').write_text(json.dumps(report,indent=2));(ROOT/'docs/research/2026-11-17-affine-fit-evidence.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
