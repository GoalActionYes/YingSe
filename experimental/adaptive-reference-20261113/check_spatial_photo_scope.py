"""Fixed first six development photos at native size; reference-layout probe.

No tuning, no new independent-test claim, and no subjective quality pass gate.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
from independent_coco import ROOT,digest
sys.path.insert(0,str(ROOT))
from reference_color_app.pipeline import analyze
from reference_color_app.lut import apply_mapping
from spatial_render import SpatialEngine


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/tst100k/native-photo-v1';out.mkdir(exist_ok=False)
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k/cohort-v1'
    cohort=json.loads((base/'cohort.json').read_text(encoding='utf-8'))
    manifest=json.loads((base/'srgb-v1/manifest.json').read_text(encoding='utf-8'))
    if digest(base/'cohort.json')!=manifest['cohort_sha256']:raise ValueError('Changed cohort')
    lookup={e['original_path']:e for e in manifest['files']};engine=SpatialEngine();results=[];permutation=None
    def read(name):
        entry=lookup[name];path=base/entry['prepared_path']
        if digest(path)!=entry['sha256']:raise ValueError('Changed prepared photo')
        with Image.open(path) as image:return np.array(image.convert('RGB')),path
    for row in cohort['groups']['validation'][:6]:
        content,cp=read(row['content']);reference,rp=read(row['reference']);gt,gp=read(row['gt'])
        if content.shape!=gt.shape:raise ValueError('Unaligned native GT dimensions')
        start=time.perf_counter();prepared=engine.prepare(content,reference);candidate=prepared.apply(content);seconds=time.perf_counter()-start
        current=apply_mapping(content,analyze(cp,rp),1)
        scores={name:float(np.abs(rgb.astype(float)-gt.astype(float)).mean()) for name,rgb in [('input',content),('current',current),('candidate',candidate)]}
        clipping={name:float(np.mean((rgb==0)|(rgb==255))) for name,rgb in [('input',content),('gt',gt),('current',current),('candidate',candidate)]}
        canvas=Image.new('RGB',(960,460),'#222');draw=ImageDraw.Draw(canvas)
        for index,(name,rgb) in enumerate([('Input',content),('Reference',reference),('GT',gt),('Current',current),('Candidate',candidate)]):
            x=index%3*320;y=index//3*230;draw.text((x+4,y+4),name,fill='white');tile=Image.fromarray(rgb);tile.thumbnail((312,204));canvas.paste(tile,(x+4,y+23))
        canvas.save(out/f'{row["index"]}-comparison.jpg',quality=95)
        results.append({'index':row['index'],'dimensions':[content.shape[1],content.shape[0]],'seconds':seconds,'mae':scores,'endpoint_channel_fraction':clipping})
        if permutation is None:
            ref=np.array(Image.fromarray(reference).resize((256,256),Image.Resampling.BILINEAR))
            rng=np.random.default_rng(20261113);shuffled=ref.reshape(-1,3)[rng.permutation(256*256)].reshape(ref.shape)
            first=engine.prepare(content,ref);second=engine.prepare(content,shuffled)
            a=first.apply(content);b=second.apply(content)
            permutation={'seed':20261113,'reference_dimensions':[256,256],
                'max_grid_delta':float((first.grid-second.grid).abs().max()),
                'changed_output_values':int(np.count_nonzero(a!=b)),
                'output_a_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'output_b_sha256':hashlib.sha256(b.tobytes()).hexdigest(),
                'scope':'Reference pixels shuffled after fixed resize, keeping joint color samples; demonstrates lack of reference spatial/semantic layout conditioning'}
            Image.fromarray(ref).save(out/'reference-original.png');Image.fromarray(shuffled).save(out/'reference-shuffled.png')
        print(json.dumps(results[-1]),flush=True)
    report={'scope':'First six previously used development-validation photos, native RGB8; no training or independent acceptance',
        'cohort_sha256':digest(base/'cohort.json'),'checkpoint_sha256':prepared.diagnostics['checkpoint_sha256'],
        'script_sha256':digest(Path(__file__)),'cases':results,'reference_permutation':permutation}
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(permutation),flush=True)


if __name__=='__main__':main()
