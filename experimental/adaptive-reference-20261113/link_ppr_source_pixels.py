"""Test fixed official/old source identity without fitting or altering old preparation."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import link_ppr_training_metadata as link
from triplet_color import to_srgb

BASE=link.ROOT/'outputs/adaptive-reference-20261113'
SAMPLES=BASE/'ppr-pixel-sample-v1'
OUT=BASE/'ppr-source-pixel-link-v1'


def difference(a,b):
    if a.shape!=b.shape:return dict(same_shape=False,exact=False)
    delta=np.abs(a.astype(np.int32)-b.astype(np.int32))
    return dict(same_shape=True,exact=bool(np.array_equal(a,b)),mae=float(delta.mean()),max_difference=int(delta.max()))


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    links_path=BASE/'ppr-group-exclusions-v3/results.json'
    assert link.digest(links_path)=='8783deef13634bd1433964d251e77af3cba765c280c5b4741cdf13318263f716'
    manifest_path=link.DATA/'srgb-v1/manifest.json'
    assert link.digest(manifest_path)=='da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f'
    plan=json.loads((SAMPLES/'plan.json').read_text());sample=json.loads((SAMPLES/'results.json').read_text())
    assert sample['finished'] and link.digest(SAMPLES/'plan.json')==sample['plan_sha256']
    links=json.loads(links_path.read_text());manifest={r['original_path']:r for r in json.loads(manifest_path.read_text())['files']}
    sources={n:link.digest(Path(__file__).with_name(n)) for n in ['link_ppr_source_pixels.py','triplet_color.py','ppr-source-pixel-link-protocol.md']}
    frozen=OUT/'frozen-source';frozen.mkdir()
    for n in sources:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    config=dict(ids=plan['ids'],source_sha256=sources,manifest_sha256=link.digest(manifest_path),links_sha256=link.digest(links_path),sample_results_sha256=link.digest(SAMPLES/'results.json'))
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    sys.path.insert(0,str(link.ROOT/'.tools/StyleGallery-audit/vendor'))
    import cv2
    rows=[]
    for key in plan['ids']:
        assert time.perf_counter()-start<60
        choices=[r for r in links['training_candidates'] if r['ppr_candidate_id']==key]
        assert len(choices)==1 and not choices[0]['exclude_for_known_consumed_group']
        entry=manifest[choices[0]['path']]
        old=next((link.BASE/root/'inputs'/entry['original_path'] for root in ['cohort-training-coverage-v2','cohort-training-coverage-v1','cohort-multiref-v1','cohort-v2','cohort-v1'] if (link.BASE/root/'inputs'/entry['original_path']).is_file()),None)
        assert old is not None and link.digest(old)==entry['original_sha256']
        prepared=link.DATA/entry['prepared_path'];assert link.digest(prepared)==entry['sha256']
        chosen=[r for r in sample['files'] if r['role']=='source' and r['id']==key];assert len(chosen)==1
        official=link.ROOT/chosen[0]['local_path'];assert link.digest(official)==chosen[0]['sha256']
        old_native=cv2.imdecode(np.frombuffer(old.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
        official_native=cv2.imdecode(np.frombuffer(official.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
        assert old_native is not None and official_native is not None
        with Image.open(old) as im:old_rgb=np.array(im.convert('RGB'));old_srgb,old_policy=to_srgb(im)
        with Image.open(official) as im:official_rgb=np.array(im.convert('RGB'));official_srgb,official_policy=to_srgb(im)
        with Image.open(prepared) as im:existing=im.convert('RGB').copy()
        sizes=[list(x.shape[:2][::-1]) for x in [old_rgb,official_rgb]]
        model_old=np.array(existing.resize((256,256),Image.Resampling.BILINEAR))
        model_official=np.array(official_srgb.resize((256,256),Image.Resampling.BILINEAR))
        rows.append(dict(id=key,old_path=entry['original_path'],old_original_sha256=entry['original_sha256'],official_sha256=chosen[0]['sha256'],same_file_bytes=old.read_bytes()==official.read_bytes(),native_dtypes=[str(old_native.dtype),str(official_native.dtype)],native_pixels=difference(old_native,official_native),rgb8_pixels=difference(old_rgb,official_rgb),raw_sizes=sizes,prepared_old_pixels=difference(np.array(old_srgb),np.array(existing)),prepared_official_pixels=difference(np.array(official_srgb),np.array(existing)),model_input_256=difference(model_old,model_official),icc_policies=[old_policy['policy'],official_policy['policy']]))
    report=dict(config_sha256=link.digest(OUT/'config.json'),rows=rows,finished=True,seconds=time.perf_counter()-start,opencv_version=cv2.__version__,training_started=False,quality_scoring=False)
    assert all(link.digest(Path(__file__).with_name(n))==h for n,h in sources.items())
    (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
