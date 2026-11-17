"""Describe all fixed expert-label differences; these are not prediction scores."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import zlib
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from triplet_color import to_srgb

BASE=ROOT/'outputs/adaptive-reference-20261113'
OUT=BASE/'ppr-expert-contrast-v1'
OLD=BASE/'ppr-pixel-sample-v1'


def compare(first,second):
    d=second.astype(np.float64)-first.astype(np.float64)
    return dict(exact=bool(np.array_equal(first,second)),mae=float(np.abs(d).mean()),max_difference=int(np.abs(d).max()),mean_rgb_second_minus_first=d.mean(axis=(0,1)).tolist())


def main():
    report=json.loads((OUT/'results.json').read_text());plan=json.loads((OUT/'plan.json').read_text())
    old=json.loads((OLD/'results.json').read_text());assert old['finished'] and report['finished']
    assert digest(OUT/'plan.json')==report['plan_sha256']
    for n,h in plan['source_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    assert report['read_bytes']==sum(r['bytes'] for r in report['requests'])<=16*1024**2
    assert len(report['requests'])<=64
    all_files=[dict(r,expert='a') for r in old['files'] if r['role']=='target_a']+report['files']
    arrays={};metadata=[]
    for row in all_files:
        path=ROOT/row['local_path'];raw=path.read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'] and zlib.crc32(raw)==row['crc']
        with Image.open(path) as im:
            bits=list(im.tag_v2.get(258,()));size=list(im.size)
            rgb,policy=to_srgb(im);array=np.array(rgb)
            small=np.array(rgb.resize((256,256),Image.Resampling.BILINEAR))
        assert bits==[8,8,8]
        arrays[row['expert'],row['id']]=(array,small)
        metadata.append(dict(id=row['id'],expert=row['expert'],size=size,bits_per_sample=bits,icc_policy=policy['policy'],pixel_sha256=hashlib.sha256(array.tobytes()).hexdigest()))
    records=[]
    for key in plan['ids']:
        mask_row=next(r for r in old['files'] if r['role']=='mask' and r['id']==key)
        with Image.open(ROOT/mask_row['local_path']) as im:alpha=np.array(im.convert('L'),dtype=np.float64)/255
        for first,second in itertools.combinations('abc',2):
            a,aa=arrays[first,key];b,bb=arrays[second,key]
            assert a.shape==b.shape and a.shape[:2]==alpha.shape
            delta=b.astype(np.float64)-a.astype(np.float64)
            regions={}
            for name,w in [('human',alpha),('background',1-alpha)]:
                assert w.sum()>0
                regions[name]=dict(mean_rgb_second_minus_first=((delta*w[:,:,None]).sum(axis=(0,1))/w.sum()).tolist(),mae=float((np.abs(delta)*w[:,:,None]).sum()/(w.sum()*3)))
            records.append(dict(id=key,pair=first+'_'+second,native=compare(a,b),model_256=compare(aa,bb),offline_soft_mask_regions=regions))
    sys.path.insert(0,str(ROOT/'.tools/ModFlows-audit/vendor'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(4,4,figsize=(14,12),layout='constrained')
    for row,key in enumerate(plan['ids']):
        src=next(r for r in old['files'] if r['role']=='source' and r['id']==key)
        with Image.open(ROOT/src['local_path']) as im:source,_=to_srgb(im)
        axes[row,0].imshow(source)
        for col,expert in enumerate('abc',1):axes[row,col].imshow(arrays[expert,key][0])
        for col,title in enumerate(['Source RGB8 proxy','Official expert A','Official expert B','Official expert C']):
            axes[row,col].set_title(key+' | '+title,fontsize=10);axes[row,col].axis('off')
    fig.suptitle('Four fixed training-origin groups: official labels only, not model predictions',fontsize=13)
    fig.savefig(OUT/'expert-labels.png',dpi=140);plt.close(fig)
    summary={pair:dict(cases=4,exact_native=sum(r['native']['exact'] for r in records if r['pair']==pair),native_mae=[r['native']['mae'] for r in records if r['pair']==pair],model_256_mae=[r['model_256']['mae'] for r in records if r['pair']==pair]) for pair in ['a_b','a_c','b_c']}
    result=dict(finished=True,acquisition_sha256=digest(OUT/'results.json'),driver_sha256=digest(__file__),color_policy_sha256=digest(Path(__file__).with_name('triplet_color.py')),metadata=metadata,cases=records,summary=summary,figure_sha256=digest(OUT/'expert-labels.png'),scope='expert label contrasts, not quality or conditional learnability scores',training_started=False)
    (OUT/'measurement.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
