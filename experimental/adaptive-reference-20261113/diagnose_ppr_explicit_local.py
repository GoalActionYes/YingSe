"""Second fixed truth-assisted test; reuse the first round's bases and signal rule."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import json
from pathlib import Path
import time
import zlib
import numpy as np
from PIL import Image
import diagnose_ppr_spatial_information as core
import link_ppr_training_metadata as link
from triplet_color import to_srgb

BASE=link.ROOT/'outputs/adaptive-reference-20261113'
SAMPLES=BASE/'ppr-explicit-local-pixels-v1'
OUT=BASE/'ppr-explicit-local-fit-v1'


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=False)
    plan=json.loads((SAMPLES/'plan.json').read_text());raw=json.loads((SAMPLES/'results.json').read_text())
    assert raw['finished'] and len(raw['files'])==9 and link.digest(SAMPLES/'plan.json')==raw['plan_sha256']
    assert raw['read_bytes']==sum(r['bytes'] for r in raw['requests'])<=16*1024**2
    links_path=BASE/'ppr-group-exclusions-v3/results.json';assert link.digest(links_path)=='8783deef13634bd1433964d251e77af3cba765c280c5b4741cdf13318263f716'
    links=json.loads(links_path.read_text())
    manifest_path=link.DATA/'srgb-v1/manifest.json';assert link.digest(manifest_path)=='da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f'
    manifest={r['original_path']:r for r in json.loads(manifest_path.read_text())['files']}
    names=['diagnose_ppr_explicit_local.py','diagnose_ppr_spatial_information.py','ppr-explicit-local-pixel-protocol.md','triplet_color.py']
    config=dict(ids=plan['ids'],source_sha256={n:link.digest(Path(__file__).with_name(n)) for n in names},sample_sha256=link.digest(SAMPLES/'results.json'),ridge=1e-6,block=8,threads=2,round=2)
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    frozen=OUT/'frozen-source';frozen.mkdir()
    for n in names:(frozen/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    yy,xx=np.indices((256,256));fold=((xx//8+yy//8)%2).reshape(-1)
    rows=[];fits=[];predictions={};inputs={};identities=[]
    for number,key in enumerate(plan['ids']):
        assert time.monotonic()-start<180
        choices=[r for r in links['training_candidates'] if r['ppr_candidate_id']==key];assert len(choices)==1 and not choices[0]['exclude_for_known_consumed_group']
        entry=manifest[choices[0]['path']]
        old=next((link.BASE/p/'inputs'/entry['original_path'] for p in ['cohort-training-coverage-v2','cohort-training-coverage-v1','cohort-multiref-v1','cohort-v2','cohort-v1'] if (link.BASE/p/'inputs'/entry['original_path']).is_file()),None)
        assert old and link.digest(old)==entry['original_sha256']
        values={};metadata=[]
        for role in ['source','target_a','mask']:
            row=next(r for r in raw['files'] if r['id']==key and r['role']==role);path=link.ROOT/row['local_path'];data=path.read_bytes()
            assert link.digest(path)==row['sha256'] and zlib.crc32(data)==row['crc'] and len(data)==row['bytes']
            if role=='source':assert data==old.read_bytes()
            with Image.open(path) as im:
                bits=getattr(im,'tag_v2',{}).get(258,());bits=[bits] if isinstance(bits,int) else list(bits)
                metadata.append(dict(role=role,size=list(im.size),bits_per_sample=bits))
                if role=='mask':values[role]=np.array(im.convert('L').resize((256,256),Image.Resampling.BILINEAR),dtype=np.float64)/255
                else:
                    image,_=to_srgb(im);values[role]=np.array(image.resize((256,256),Image.Resampling.BILINEAR))
        assert len({tuple(m['size']) for m in metadata})==1
        identities.append(dict(id=key,old_source_path=entry['original_path'],source_file_bytes_identical=True,source_sha256=entry['original_sha256'],metadata=metadata))
        source=values['source'];alpha=values['mask'];target=values['target_a'];y=target.reshape(-1,3)/255
        cubic=core.basis(source,3);quintic=core.basis(source,5)
        shuffled=np.random.default_rng(20261118+number).permutation(alpha.reshape(-1))
        matrices={'global3':cubic,'global5':quintic,'human3':np.c_[cubic,cubic*alpha.reshape(-1,1)],'position3':np.c_[cubic,cubic*(xx.reshape(-1,1)/255)],'permuted3':np.c_[cubic,cubic*shuffled[:,None]]}
        metrics={}
        for method,x in matrices.items():
            result=np.empty_like(y)
            for parity in [0,1]:
                train=fold==parity;test=~train
                penalty=np.eye(x.shape[1])*1e-6;penalty[0,0]=0
                gram=x[train].T@x[train]/train.sum()+penalty
                weights=np.linalg.solve(gram,x[train].T@y[train]/train.sum());assert np.isfinite(weights).all()
                result[test]=x[test]@weights
                fits.append(dict(id=key,method=method,fold=parity,features=x.shape[1],regularized_condition_number=float(np.linalg.cond(gram))))
            image=np.rint(np.clip(result,0,1)*255).astype(np.uint8).reshape(256,256,3)
            predictions[key+'_'+method]=image;metrics[method]=core.metrics(image,target,alpha)
        inputs[key+'_source']=source;inputs[key+'_alpha']=alpha;inputs[key+'_a']=target
        rows.append(dict(id=key,expert='a',metrics=metrics))
    np.savez_compressed(OUT/'predictions.npz',**predictions);np.savez_compressed(OUT/'inputs.npz',**inputs)
    report=dict(finished=True,config_sha256=link.digest(OUT/'config.json'),source_sha256=config['source_sha256'],truth_assisted=True,software_inference=False,rows=rows,fits=fits,source_identities=identities,decision=core.signal(rows),seconds=time.monotonic()-start,product_promoted=False,product_candidate_trained=False)
    assert all(link.digest(Path(__file__).with_name(n))==h for n,h in config['source_sha256'].items())
    (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(rows=rows,decision=report['decision'],seconds=report['seconds'])))


if __name__=='__main__':main()
