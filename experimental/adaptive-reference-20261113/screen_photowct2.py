"""Author notebook default ZCA plus guided filtering, consumed dev64 only."""
import os
os.environ['TF_USE_LEGACY_KERAS']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import ast,json,sys,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[2]
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)


def main():
    source=ROOT/'.tools/PhotoWCT2-audit';previous=Path.cwd()
    preflight=json.loads((ROOT/'docs/research/2026-11-17-photowct2-runtime-evidence.json').read_text())
    for manifest in preflight['manifests'].values():
        for e in manifest['files']:assert digest(source/e['path'])==e['sha256']
    os.chdir(source);sys.path.insert(0,str(source))
    try:
        from utils.model_relu import VggEncoder,VggDecoder
        encoder,decoder=VggEncoder(),VggDecoder();decoder(0,tf.zeros((1,64,64,64)),skip=None)
        encoder.load_weights('ckpts/ckpts-relu/encoder').assert_consumed()
        decoder.load_weights('ckpts/ckpts-relu/decoder').assert_consumed()
    finally:os.chdir(previous)
    notebook=json.loads((source/'relu_demo.ipynb').read_text())
    functions={}
    loop=None
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        tree=ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node,ast.FunctionDef):functions[node.name]=node
            if isinstance(node,ast.For) and 'cont_img' in ast.unparse(node):loop=node
    env={'tf':tf,'np':np}
    names=['inv_sqrt_cov','stylize_core','stylize_zca']
    for name in names:exec(compile(ast.Module(body=[functions[name]],type_ignores=[]),'<author-notebook>','exec'),env)
    assert loop is not None
    # Load the original postprocessor by explicit file path, without copying its code.
    import importlib.util
    spec=importlib.util.spec_from_file_location('author_gif',source/'photo_gif.py');gif=importlib.util.module_from_spec(spec);spec.loader.exec_module(gif)
    smoothing=gif.GIFSmoothing(r=50,eps=(.02*255)**2)
    base=ROOT/'outputs/adaptive-reference-20261113/tst100k'
    registry_path=base/'regional-correction-inputs-v1/results.json';registry=json.loads(registry_path.read_text())
    rows=[r for r in registry['rows'] if r['split']=='validation'];assert len(rows)==64
    inputs={e['index']:e for e in registry['inputs']}
    old=base/'gradestudio-screen-v1';old_report=json.loads((old/'results.json').read_text());assert old_report['finished'] and old_report['registry_sha256']==digest(registry_path)
    baselines={r['index']:r['outputs']['baseline'] for r in old_report['rows']}
    prepared=base/'photowct2-inputs-v1';prepared_report=json.loads((prepared/'results.json').read_text())
    assert prepared_report['registry_sha256']==digest(registry_path)
    out=base/'photowct2-screen-v3';out.mkdir(exist_ok=False)
    report=dict(scope='Consumed development64, author relu/ZCA/guided filter, full strength',
        baseline_report_sha256=digest(old/'results.json'),registry_sha256=digest(registry_path),prepared_report_sha256=digest(prepared/'results.json'),
        author_sha256={n:digest(source/n) for n in ['relu_demo.ipynb','photo_gif.py']},
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['screen_photowct2.py','photowct2-screen-protocol.md']},
        runtime=tf.__version__,rows=[],finished=False,product_integrated=False,training=False)
    (out/'frozen-config.json').write_text(json.dumps(report,indent=2))
    def read(path,sha):
        assert digest(path)==sha
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    def encode(x):
        a=encoder(0,x);b=encoder(1,a);c=encoder(2,b[0]);d=encoder(3,c[0]);return [a,b,c,d]
    def render(content,reference):
        y=encode(content);s=encode(reference);x=y[-1][0]
        for level in [3,2,1,0]:
            sf=s[level][0] if level else s[0]
            sf=tf.reshape(tf.transpose(sf,[0,3,1,2]),[1,sf.shape[-1],-1])
            x=env['stylize_core'](x,sf,opt='zca')
            x=decoder(level,x,skip=y[level][1] if level else None)
        return tf.clip_by_value(x,0,1)
    def postprocess(raw,folder):
        # OpenCV's narrow Windows filename API cannot read the Chinese absolute path.
        # Its original file-reading path works with the ASCII filename in the same cwd.
        before=Path.cwd()
        try:
            os.chdir(folder)
            return smoothing.process(raw,'input.png')
        finally:os.chdir(before)
    for i,row in enumerate(rows):
        arrays=[]
        for role in ['content','reference']:
            e=prepared_report['inputs'][str(row['roles'][role]['feature_index'])];a=read(prepared/e['path'],e['sha256'])
            assert hashlib.sha256(a.tobytes()).hexdigest()==e['pixel_sha256'];arrays.append(a)
        folder=out/str(row['index']);folder.mkdir();guide=folder/'input.png';Image.fromarray(arrays[0]).save(guide)
        content,reference=[a[None].astype(np.float32)/255 for a in arrays]
        start=time.perf_counter();raw=render(content,reference).numpy()
        assert raw.shape==(1,256,256,3) and np.isfinite(raw).all()
        final=postprocess(raw[0],folder);actual=np.array(final)
        if i==0:
            assert np.array_equal(actual,np.array(postprocess(render(content,reference).numpy()[0],folder)))
            # Execute the author's original loop body with only file loading/saving replaced.
            from types import SimpleNamespace
            native=dict(env,enc_dec=SimpleNamespace(encoder=encoder,decoder=decoder),cont_img=content,style_img=reference,opt='zca')
            nodes=[n for n in loop.body if isinstance(n,ast.Assign) and not any(isinstance(t,ast.Name) and t.id in ['cont_img','style_img','opt'] for t in n.targets)]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'<author-loop>','exec'),native)
            assert np.array_equal(raw,native['x'].numpy())
            report['first_author_loop_float_exact']=True;report['first_rgb_repeat_exact']=True
        Image.fromarray(np.uint8(raw[0]*255)).save(folder/'raw.png');final.save(folder/'result.png')
        b=baselines[row['index']];assert digest(old/b['path'])==b['sha256']
        report['rows'].append(dict(index=row['index'],seconds=time.perf_counter()-start,
            result_path=str((folder/'result.png').relative_to(out)),result_sha256=digest(folder/'result.png'),
            raw_sha256=digest(folder/'raw.png'),baseline_path=b['path'],baseline_sha256=b['sha256']))
        (out/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(predictions=i+1)),flush=True)
    report.update(finished=True);(out/'results.json').write_text(json.dumps(report,indent=2));print('All64 predictions saved; GT scoring is separate',flush=True)


if __name__=='__main__':main()
