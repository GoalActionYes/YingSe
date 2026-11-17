"""Cross-check native TIFF depth and the explicit display proxy using cached OpenCV."""
import hashlib
import json
import sys
import numpy as np
from PIL import Image
from independent_coco import ROOT, digest

OUT=ROOT/'outputs/adaptive-reference-20261113/ppr-pixel-sample-v1'


def main():
    sys.path.insert(0,str(ROOT/'.tools/StyleGallery-audit/vendor'))
    import cv2
    report=json.loads((OUT/'results.json').read_text())
    assert report['finished'] and len(report['files'])==12
    rows=[]
    for row in report['files']:
        path=ROOT/row['local_path'];raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        array=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
        assert array is not None
        with Image.open(path) as im:pil=np.array(im.convert('L' if row['role']=='mask' else 'RGB'))
        if row['role']=='source':
            assert array.dtype==np.uint16 and array.ndim==3
            proxy=(array[:,:,::-1]>>8).astype(np.uint8)
        elif row['role']=='target_a':
            assert array.dtype==np.uint8 and array.ndim==3
            proxy=array[:,:,::-1]
        else:
            assert array.dtype==np.uint8 and array.ndim==2
            proxy=array
        assert proxy.shape==pil.shape
        difference=int(np.abs(proxy.astype(np.int16)-pil.astype(np.int16)).max())
        assert difference==0
        rows.append(dict(id=row['id'],role=row['role'],native_dtype=str(array.dtype),native_shape=list(array.shape),max_display_proxy_difference_vs_pillow=difference,native_pixel_sha256=hashlib.sha256(array.tobytes()).hexdigest()))
    audit=dict(finished=True,driver_sha256=digest(__file__),opencv_version=cv2.__version__,source_16bit_target_8bit_confirmed=True,source_display_policy='OpenCV RGB uint16 >> 8 exactly matches Pillow RGB proxy for all four samples',rows=rows,pixel_quality_scoring=False,training_started=False)
    (OUT/'decoder-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='rows'}))


if __name__=='__main__':main()
