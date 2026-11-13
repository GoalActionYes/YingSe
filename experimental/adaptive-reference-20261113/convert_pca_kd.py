"""Run fixed author TensorFlow blocks, export safe arrays and independent fixtures."""
import os
os.environ['TF_USE_LEGACY_KERAS']='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTRAOP_THREADS']='4'
os.environ['TF_NUM_INTEROP_THREADS']='2'
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest


def main():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    source=ROOT/'.tools/PCA-KD-source';manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['revision']!='658d4a247b1aed87b9b3f19d8f8ae555e29fe31b':raise ValueError('Unexpected source revision')
    for entry in manifest['files']:
        if digest(source/entry['path'])!=entry['sha256']:raise ValueError('Author source/weight file changed')
    out=ROOT/'.tools/PCA-KD-converted'
    if out.exists():raise ValueError('Converted destination already exists; do not overwrite evidence')
    arrays={};fixtures={};layer_info=[]
    # Windows TensorFlow's checkpoint reader rejected the Unicode workspace path.
    # Only byte-identical copies go into this private ASCII temporary directory.
    parent=Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix='yingse-pca-',ignore_cleanup_errors=True) as temporary:
        stage=Path(temporary).resolve()
        if not stage.is_relative_to(parent):raise ValueError('Temporary stage escapes task temp parent')
        for entry in manifest['files']:
            if entry['path'].startswith('VGG backbone/'):
                relative=Path(entry['path']).relative_to('VGG backbone')
                target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/entry['path'],target)
                if digest(target)!=entry['sha256']:raise ValueError('Staged asset differs')
        previous=Path.cwd()
        try:
            os.chdir(stage);sys.path.insert(0,str(stage))
            from utils.lightweight_model import VggSEncoder,VggSDecoder
            encoder,decoder=VggSEncoder(),VggSDecoder()
            # Published weights name the four training-only eigenbases conv_aux;
            # the current author demo class calls them eigenbases. A checkpoint
            # view restores every object strictly without changing author math.
            tf.train.Checkpoint(btnecks=encoder.btnecks,conv_aux=encoder.eigenbases).read(
                str(stage/'ckpts/sencoder')).assert_consumed()
            tf.train.Checkpoint(btnecks=decoder.btnecks).read(
                str(stage/'ckpts/sdecoder')).assert_consumed()
            for kind,model in [('encoder',encoder),('decoder',decoder)]:
                for block_index,block in enumerate(model.btnecks):
                    convolutions=[layer for layer in block.layers if isinstance(layer,tf.keras.layers.Conv2D)]
                    for conv_index,layer in enumerate(convolutions):
                        weights=layer.get_weights()
                        if len(weights)!=2 or not all(np.isfinite(x).all() for x in weights):raise ValueError('Invalid restored convolution')
                        key=f'{kind}_{block_index}_{conv_index}'
                        arrays[key+'_kernel'],arrays[key+'_bias']=weights
                        layer_info.append(dict(key=key,source_name=layer.name,kernel_shape=list(weights[0].shape),bias_shape=list(weights[1].shape)))
            def square_root(cov,inverse=False):
                s,u,_=tf.linalg.svd(cov+tf.eye(cov.shape[-1]))
                rank=tf.reduce_sum(tf.cast(s>1e-5,tf.int32))
                values=tf.sqrt(s[:,:rank]);values=1/values if inverse else values
                u=u[:,:,:rank]
                return u@tf.linalg.diag(values)@tf.transpose(u,[0,2,1])
            def zca(content,style):
                shape=content.shape;c=tf.reshape(tf.transpose(content,[0,3,1,2]),[1,shape[-1],-1])
                s=tf.reshape(tf.transpose(style,[0,3,1,2]),[1,style.shape[-1],-1])
                cm,sm=tf.reduce_mean(c,-1,keepdims=True),tf.reduce_mean(s,-1,keepdims=True)
                c,s=c-cm,s-sm
                cc,ss=c@tf.transpose(c,[0,2,1])/c.shape[-1],s@tf.transpose(s,[0,2,1])/s.shape[-1]
                transformed=square_root(ss)@square_root(cc,True)@c+sm
                return tf.transpose(tf.reshape(transformed,[1,shape[-1],shape[1],shape[2]]),[0,2,3,1])
            rng=np.random.default_rng(20261113)
            tests=[]
            tests.append(('noise',rng.integers(0,256,(1,64,80,3),dtype=np.uint8).astype(np.float32)/255,
                rng.integers(0,256,(1,72,88,3),dtype=np.uint8).astype(np.float32)/255))
            ramp=np.linspace(0,1,64*80,dtype=np.float32).reshape(1,64,80,1).repeat(3,axis=-1)
            tests.append(('ramp',ramp,np.clip(ramp*np.array([.8,1,1.1],dtype=np.float32)+.03,0,1)))
            data=ROOT/'outputs/adaptive-reference-20261113/pst50'
            photos=[]
            for path,size in [(data/'content_709/in1.png',(80,64)),(data/'paired_style/tar1.png',(88,72))]:
                with Image.open(path) as image:photos.append(np.array(image.convert('RGB').resize(size,Image.Resampling.BILINEAR),dtype=np.float32)[None]/255)
            tests.append(('natural',*photos))
            def save(name,value):fixtures[name]=np.asarray(value).transpose(0,3,1,2).copy()
            for name,c,s in tests:
                save(name+'_content',c);save(name+'_style',s);cs=[];ss=[];skips=[]
                for block in range(4):
                    c,skip=encoder.call(block,c);s,_=encoder.call(block,s)
                    cs.append(c);ss.append(s);skips.append(skip)
                    save(f'{name}_enc{block}',c);save(f'{name}_skip{block}',skip);save(f'{name}_style{block}',s)
                value=cs[-1]
                for block in reversed(range(4)):
                    value=zca(value,ss[block]);save(f'{name}_zca{block}',value)
                    value=decoder.call(block,value,skips[block]);save(f'{name}_dec{block}',value)
                save(name+'_output',tf.clip_by_value(value,0,1))
            import utils.lightweight_model as author
            author.csvfile.close()
        finally:os.chdir(previous)
    out.mkdir(exist_ok=False)
    np.savez_compressed(out/'weights.npz',**arrays);np.savez_compressed(out/'oracle.npz',**fixtures)
    shutil.copyfile(source/'LICENSE.md',out/'LICENSE.md')
    provenance=dict(source=manifest,source_manifest_sha256=digest(source/'manifest.json'),tensorflow=tf.__version__,
        keras='Legacy tf_keras2.20.1; author source unchanged',all_checkpoint_objects_restored=True,
        checkpoint_adapter='Strict checkpoint view: saved conv_aux maps to author eigenbases; all four training-only matrices restored',
        converted_layers=layer_info,weight_arrays=len(arrays),inference_parameters=sum(x.size for x in arrays.values()),
        weights_sha256=digest(out/'weights.npz'),oracle_sha256=digest(out/'oracle.npz'),
        converter_sha256=digest(Path(__file__)),unused_training_eigenbases=4,
        scope='CPU author-code fixtures and converted inference convolutions; not quality validation')
    (out/'manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in provenance.items() if k not in ('source','converted_layers')}),flush=True)


if __name__=='__main__':main()
