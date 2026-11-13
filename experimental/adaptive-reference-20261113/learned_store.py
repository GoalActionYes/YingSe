"""Research-only persistent model mappings, isolated from the stable Store cache."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile
import zlib
import numpy as np
from reference_color_app import media,pipeline
from reference_color_app.contracts import Mapping
from reference_color_app.storage import Store,VERSION,_id,_json
from reference_color_app.lut import _layers
from canonical_model import ROOT,CHECKPOINT_SHA,UPSTREAM_COMMIT


def identity_cube():
    axis=np.linspace(0,1,64,dtype=np.float32)
    return np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)


def current_model_tag():
    directory=Path(__file__).parent
    digest=hashlib.sha256(UPSTREAM_COMMIT.encode())
    for name in ('learned_store.py','canonical_model.py','model_provider.py'):
        digest.update((directory/name).read_bytes())
    if pipeline.file_sha256(ROOT/'.tools/CanonCGT/pretrained/E2E.pth')!=CHECKPOINT_SHA:
        raise ValueError('模型文件缺失或校验失败')
    digest.update(CHECKPOINT_SHA.encode())
    for name in ('models/base.py','models/networks/end_to_end_finetuning.py',
                 'models/networks/Estimator/Estimator_modules.py','models/networks/Estimator/Estimator_blocks.py'):
        digest.update((ROOT/'.tools/CanonCGT'/name).read_bytes())
    return digest.hexdigest()


class LearnedStore(Store):
    def __init__(self,root,provider,identity_tag=None):
        super().__init__(root)
        self.provider=provider;self.identity_tag=identity_tag

    def _model_identity(self,target,reference,time_seconds):
        identity,_=super()._identity(target,reference,time_seconds)
        identity.update(experimental_engine='canonical-e2e-v1',model_tag=self.identity_tag or current_model_tag())
        return identity,hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def analyze_engine(self,target_id,reference_id,time_seconds=0.,engine='baseline'):
        if engine=='baseline':return super().analyze(target_id,reference_id,time_seconds)
        if engine!='model':raise ValueError('未知调色方法')
        if isinstance(time_seconds,(bool,str)) or not np.isscalar(time_seconds) or not math.isfinite(float(time_seconds)) or float(time_seconds)<0:
            raise ValueError('无效分析时间')
        time_seconds=float(time_seconds)
        with self.lock:
            target=self.get_media(target_id); reference=self.get_media(reference_id)
            if reference['kind']!='image':raise ValueError('参考素材必须为图片')
            identity,key=self._model_identity(target,reference,time_seconds)
            cache_hit=True
            try:mapping=self._load(key,identity)
            except (OSError,ValueError,KeyError,TypeError,EOFError,zipfile.BadZipFile,zlib.error):
                cache_hit=False
                rgb=media.decode_image(target['path']) if target['kind']=='image' else media.decode_frame(target['path'],time_seconds)[0]
                ref=media.decode_image(reference['path'])
                if np.array_equal(rgb,ref):
                    cube=identity_cube()
                    mapping=Mapping(np.stack([cube,cube]),dict(engine='model',exact_identity=True,
                                    reason='Decoded RGB equality bypasses the learned model'))
                else:
                    if self.provider is None:raise ValueError('模型尚未准备好')
                    # Input resampling changes this model's inferred grading.
                    # Estimate from decoded originals; only previews may be reduced.
                    mapping=self.provider.compile(rgb,ref)
                layers=_layers(mapping)
                diagnostics=dict(mapping.diagnostics,engine='model',
                    strength_policy='Existing two-layer strength response; not linear interpolation',
                    mapping_scope='One global 64-cube; not semantic spatial transfer',
                    inputs=dict(target=dict(metadata=target['metadata'],sha256=target['sha256'],original_dimensions=[rgb.shape[1],rgb.shape[0]]),
                                reference=dict(metadata=reference['metadata'],sha256=reference['sha256'],original_dimensions=[ref.shape[1],ref.shape[0]])))
                mapping=Mapping(layers,pipeline.json_safe(diagnostics))
                temp=self.root/'cache'/('.'+_id()+'.npz')
                try:
                    np.savez_compressed(temp,layers=layers)
                    os.replace(temp,self.root/'cache'/(key+'.npz'))
                finally:temp.unlink(missing_ok=True)
                _json(self.root/'cache'/(key+'.json'),dict(version=VERSION,identity=identity,
                      layer_sha256=pipeline.layer_sha256(layers),diagnostics=mapping.diagnostics))
            identifier=_id()
            record=dict(id=identifier,target_id=target_id,reference_id=reference_id,analysis_time=time_seconds,
                        codefingerprint=identity['codefingerprint'],cache_key=key,identity=identity,diagnostics=mapping.diagnostics,
                        cache_hit=cache_hit)
            _json(self._record('mappings',identifier),record)
            return copy.deepcopy(record)

    def get_mapping(self,mapping_id):
        record=self.get_mapping_record(mapping_id)
        if 'experimental_engine' not in record['identity']:return super().get_mapping(mapping_id)
        identity,key=self._model_identity(self.get_media(record['target_id']),self.get_media(record['reference_id']),record['analysis_time'])
        if key!=record['cache_key'] or identity!=record['identity']:raise ValueError('模型映射已过期，请重新分析')
        return self._load(key,identity)
