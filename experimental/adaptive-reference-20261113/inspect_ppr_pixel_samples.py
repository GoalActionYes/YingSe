"""Verify fixed local members and render an offline geometry diagnostic, not quality scores."""
import hashlib
import json
from pathlib import Path
import sys
import zlib
import numpy as np
from PIL import Image
from independent_coco import ROOT, digest

OUT=ROOT/'outputs/adaptive-reference-20261113/ppr-pixel-sample-v1'


def main():
    report=json.loads((OUT/'results.json').read_text());plan=json.loads((OUT/'plan.json').read_text())
    assert report['finished'] and not report['pixels_decoded'] and len(report['files'])==12
    assert digest(OUT/'plan.json')==report['plan_sha256']
    assert digest(Path(__file__).with_name('sample_ppr_remote_zip.py'))==plan['driver_sha256']
    assert digest(Path(__file__).with_name('ppr-pixel-sample-protocol.md'))==plan['protocol_sha256']
    assert report['read_bytes']==sum(r['bytes'] for r in report['requests'])<=32*1024**2
    assert len(report['requests'])<=96
    arrays={};records=[]
    for role in ['source','target_a','mask']:
        index={r['name']:r for r in json.loads((OUT/(role+'-index.json')).read_text())}
        rows=[r for r in report['files'] if r['role']==role]
        assert [r['id'] for r in rows]==plan['ids']
        for row in rows:
            path=ROOT/row['local_path'];raw=path.read_bytes();entry=index[row['member']]
            assert len(raw)==row['bytes']==entry['bytes']
            assert hashlib.sha256(raw).hexdigest()==row['sha256']
            assert zlib.crc32(raw)==row['crc']==entry['crc']
            with Image.open(path) as im:
                tags=getattr(im,'tag_v2',{})
                bits=tags.get(258);bits=[bits] if isinstance(bits,int) else list(bits) if bits is not None else None
                meta=dict(id=row['id'],role=role,size=list(im.size),format=im.format,mode=im.mode,bits_per_sample=bits,orientation=int(tags.get(274,1)))
                array=np.array(im.convert('L' if role=='mask' else 'RGB'))
            arrays[role,row['id']]=array
            if role=='mask':meta['values']=np.unique(array).tolist();meta['nonzero_fraction']=float((array>0).mean())
            records.append(meta)
    geometry=[]
    for key in plan['ids']:
        selected=[r for r in records if r['id']==key]
        same=len({tuple(r['size']) for r in selected})==1
        geometry.append(dict(id=key,same_raw_dimensions=same,raw_orientations=[r['orientation'] for r in selected]))
    sys.path.insert(0,str(ROOT/'.tools/ModFlows-audit/vendor'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(4,4,figsize=(14,12),layout='constrained')
    for row,key in enumerate(plan['ids']):
        source=arrays['source',key];target=arrays['target_a',key];mask=arrays['mask',key]
        axes[row,0].imshow(source);axes[row,1].imshow(target);axes[row,2].imshow(mask,cmap='gray',vmin=0,vmax=255)
        axes[row,3].imshow(target)
        if mask.shape==target.shape[:2] and mask.min()<mask.max():axes[row,3].contour(mask,levels=[127.5],colors=['#ff6020'],linewidths=.7)
        for col,title in enumerate(['Source (8-bit display proxy)','Expert A (8-bit)','Human mask','Expert A + mask boundary']):
            axes[row,col].set_title(key+' | '+title,fontsize=9);axes[row,col].axis('off')
    fig.suptitle('Fixed training-origin samples: offline pairing / geometry only; no model prediction',fontsize=13)
    fig.savefig(OUT/'geometry-diagnostic.png',dpi=140);plt.close(fig)
    result=dict(finished=True,report_sha256=digest(OUT/'results.json'),inspector_sha256=digest(__file__),files=records,geometry=geometry,member_crc_and_sha_verified=12,pixel_display_dtype='uint8; source TIFF bit-depth tags retained separately',pixel_quality_scoring=False,training_started=False,figure_sha256=digest(OUT/'geometry-diagnostic.png'))
    (OUT/'inspection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
