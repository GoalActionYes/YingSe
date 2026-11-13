"""Construct opposite luma/chroma relationships with equal marginal features."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from independent_coco import ROOT,digest
from spatial_render import SpatialEngine


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/reference-joint-limit-v1';out.mkdir(exist_ok=False)
    matrix=np.array([[.2126,.7152,.0722],[-.1146,-.3854,.5],[.5,-.4542,-.0458]],float)
    def color(y,cb,cr):
        center=np.array([(y+.5)/128,(cb+.5)/128-.5,(cr+.5)/128-.5])
        near=np.rint(np.linalg.solve(matrix,center)*255).astype(int)
        for delta in itertools.product(range(-2,3),repeat=3):
            rgb=near+delta
            if (rgb<0).any() or (rgb>255).any():continue
            value=matrix@(rgb/255);bins=np.floor((value+np.array([0,.5,.5]))*128).astype(int)
            if np.array_equal(bins,[y,cb,cr]):return rgb.astype(np.uint8)
        raise ValueError('No exact-bin RGB8 color found')
    dark_cool=color(44,72,56);light_warm=color(84,56,72)
    dark_warm=color(44,56,72);light_cool=color(84,72,56)
    a=np.empty((256,256,3),np.uint8);b=np.empty_like(a)
    a[:128]=dark_cool;a[128:]=light_warm;b[:128]=dark_warm;b[128:]=light_cool
    target=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],256,axis=0);target=np.repeat(target,3,axis=2)
    engine=SpatialEngine()
    histograms=[];relationships=[]
    for rgb in [a,b]:
        yc=engine.api['srgb_to_ycbcr'](engine.tensor(rgb))
        histograms.append(engine.api['batch_histogram_1D'](yc))
        values=yc[0].cpu().permute(1,2,0).numpy().reshape(-1,3)
        relationships.append(np.cov(values.T).tolist())
    if not torch.equal(*histograms):raise ValueError('Constructed references do not have equal actual author marginal histograms')
    first=engine.prepare(target,a);second=engine.prepare(target,b)
    outputs=[first.apply(target),second.apply(target)]
    report=dict(scope='Constructive representation counterexample, not natural-picture quality or semantic acceptance',
        colors=dict(dark_cool=dark_cool.tolist(),light_warm=light_warm.tolist(),dark_warm=dark_warm.tolist(),light_cool=light_cool.tolist()),
        histogram_max_delta=float((histograms[0]-histograms[1]).abs().max()),reference_ycbcr_covariances=relationships,
        grid_max_delta=float((first.grid-second.grid).abs().max()),output_changed_values=int(np.count_nonzero(outputs[0]!=outputs[1])),
        output_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in outputs],script_sha256=digest(Path(__file__)),
        limitation='Independent Y/Cb/Cr marginal histograms lose luma/chroma dependence; identical features force identical mapping for the same target')
    canvas=Image.new('RGB',(768,280),'#222');draw=ImageDraw.Draw(canvas)
    for i,(label,rgb) in enumerate([('A: cool shadows, warm lights',a),('B: warm shadows, cool lights',b),('Identical model output',outputs[0])]):
        canvas.paste(Image.fromarray(rgb),(i*256,24));draw.text((i*256+3,4),label,fill='white')
    canvas.save(out/'counterexample.png')
    for name,rgb in [('reference-a',a),('reference-b',b),('target',target),('output-a',outputs[0]),('output-b',outputs[1])]:Image.fromarray(rgb).save(out/(name+'.png'))
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
