"""Strict author checkpoint loading and reconstruction only; no quality claim."""
import os
os.environ['TF_USE_LEGACY_KERAS']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json,sys,hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'.tools/PhotoWCT2-audit'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={n:json.loads((source/n).read_text()) for n in ['source-check.json','checkpoint-check.json']}
for manifest in checks.values():
    assert manifest['revision']=='834a7701769de1cf0d7b6cee250f583f4cca1e8f'
    for e in manifest['files']:assert digest(source/e['path'])==e['sha256']
previous=Path.cwd()
try:
    os.chdir(source);sys.path.insert(0,str(source))
    from utils.model_relu import VggEncoder,VggDecoder
    encoder,decoder=VggEncoder(),VggDecoder()
    # Author decoder block0 is a lazy Sequential; create its variables before strict restore.
    decoder(0,tf.zeros((1,64,64,64)),skip=None)
    encoder.load_weights('ckpts/ckpts-relu/encoder').assert_consumed()
    decoder.load_weights('ckpts/ckpts-relu/decoder').assert_consumed()
    rgb=np.random.default_rng(20261130).random((1,64,64,3),dtype=np.float32)
    x1=encoder(0,rgb);x2=encoder(1,x1);x3=encoder(2,x2[0]);x4=encoder(3,x3[0])
    result=decoder(3,x4[0],skip=x4[1]);result=decoder(2,result,skip=x3[1])
    result=decoder(1,result,skip=x2[1]);result=decoder(0,result,skip=None).numpy()
    assert result.shape==rgb.shape and np.isfinite(result).all()
finally:os.chdir(previous)
report=dict(scope='Author architecture/checkpoint runtime preflight only; not style inference or quality evaluation',
    tensorflow=tf.__version__,legacy_keras=True,manifests=checks,strict_checkpoint_consumed=True,
    reconstruction_shape=list(result.shape),finite=True,source_sha256=digest(Path(__file__)),
    product_integrated=False,training=False)
(ROOT/'docs/research/2026-11-17-photowct2-runtime-evidence.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='manifests'}))
