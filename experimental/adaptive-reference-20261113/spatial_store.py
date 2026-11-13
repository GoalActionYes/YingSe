"""Spatial-grid persistence, separate from all stable/global workspaces."""
import copy
import hashlib
import json
import math
import io
import os
from collections import OrderedDict
from pathlib import Path
import threading
import zipfile
import zlib
import numpy as np
from PIL import Image,ImageOps,ImageCms
import torch
from reference_color_app import media,pipeline
from reference_color_app.storage import Store,VERSION,_id,_json
from reference_color_app.contracts import Mapping
from reference_color_app.adjustments import normalize
from reference_color_app.lut import _strength
from learned_store import identity_cube
from spatial_render import PreparedSpatial,CHECKPOINT_SHA,finish_spatial
from triplet_color import to_srgb


class SpatialStore(Store):
    def __init__(self,root,engine,identity_tag=None):
        super().__init__(root);self.engine=engine;self.render_lock=threading.RLock();self.preview_lock=threading.RLock();self.preview_cache=OrderedDict()
        self.identity_lock=threading.RLock();self.probe_cache=OrderedDict()
        names=['spatial_render.py','spatial_store.py','triplet_color.py','training_checks.py']
        self.engine_tag=identity_tag or hashlib.sha256((getattr(engine,'checkpoint_sha',CHECKPOINT_SHA)+''.join(pipeline.file_sha256(Path(__file__).with_name(n)) for n in names)).encode()).hexdigest()

    def _identity(self,target,reference,time_seconds):
        # Callers supply freshly SHA-verified owned media. Reuse only the
        # metadata probe, keyed by those bytes and the current decoder/code.
        fingerprint=pipeline.algorithm_fingerprint()
        decoder_sha=pipeline.file_sha256(media.ffmpeg_path()) if target['kind']=='video' else None
        decoder={}
        for name,record in [('target',target),('reference',reference)]:
            key=(str(Path(record['path']).resolve()),record['sha256'],record['kind'],decoder_sha if record['kind']=='video' else None,fingerprint)
            with self.identity_lock:
                metadata=self.probe_cache.get(key)
                if metadata is not None:self.probe_cache.move_to_end(key)
            if metadata is None:
                metadata=media.probe_media(record['path']);metadata.pop('metadata_text',None)
                with self.identity_lock:
                    self.probe_cache[key]=copy.deepcopy(metadata)
                    while len(self.probe_cache)>32:self.probe_cache.popitem(last=False)
            decoder[name]=copy.deepcopy(metadata)
        if decoder_sha is not None:decoder['ffmpeg_sha256']=decoder_sha
        identity=dict(version=VERSION,target=target['sha256'],reference=reference['sha256'],time_seconds=time_seconds,
            codefingerprint=fingerprint,decoder=decoder,spatial_engine=self.engine_tag)
        return identity,hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()

    def decode(self,record,time_seconds=0.):
        if record['kind']=='video':return media.decode_frame(record['path'],time_seconds)
        media.decode_image(record['path'])  # Preserve the app's precision checks.
        with Image.open(record['path']) as image:
            oriented=ImageOps.exif_transpose(image);profile_bytes=oriented.info.get('icc_profile')
            # The upload contract permits ordinary RGBA/palette/gray inputs.
            # Match the existing RGB conversion; honor native gray ICCs before
            # promotion rather than applying a gray profile to RGB channels.
            if profile_bytes and oriented.mode in ('L','LA'):
                profile=ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
                if profile.profile.xcolor_space.strip()=='GRAY':
                    oriented=ImageCms.profileToProfile(oriented.getchannel('L'),profile,ImageCms.createProfile('sRGB'),outputMode='RGB',renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
            normalized=oriented.convert('RGB')
            prepared,info=to_srgb(normalized)
        return np.array(prepared),dict(record['metadata'],spatial_color_preparation=info)

    def analyze(self,target_id,reference_id,time_seconds=0.):
        if isinstance(time_seconds,(bool,str)) or not math.isfinite(float(time_seconds)) or float(time_seconds)<0:raise ValueError('无效分析时间')
        time_seconds=float(time_seconds)
        with self.lock:
            target,reference=self.get_media(target_id),self.get_media(reference_id)
            if reference['kind']!='image':raise ValueError('参考素材必须是图片')
            identity,key=self._identity(target,reference,time_seconds);cache_hit=False
            try:
                self._load(key,identity);self._grid(key);cache_hit=True
            except (ValueError,OSError,KeyError,EOFError,zipfile.BadZipFile,zlib.error):
                rgb,meta=self.decode(target,time_seconds);ref,rmeta=self.decode(reference)
                with self.render_lock:prepared=self.engine.prepare(rgb,ref)
                grid=prepared.grid.detach().cpu().numpy().astype(np.float32)
                cube=identity_cube();layers=np.stack([cube,cube])
                diagnostics=dict(engine='spatial',exact_identity=bool(np.array_equal(rgb,ref)),model_identity=self.engine_tag,
                    grid_sha256=hashlib.sha256(grid.tobytes()).hexdigest(),grid_shape=list(grid.shape),
                    mapping_scope='Spatial bilateral grid; identity LUT is a project transport placeholder, never an export',
                    strength_policy='Linear original/graded blend; manual controls applied afterwards',
                    research_scope='Noncommercial academic prototype',analysis=prepared.diagnostics,
                    inputs=dict(target=dict(metadata=meta,sha256=target['sha256']),reference=dict(metadata=rmeta,sha256=reference['sha256'])))
                temp=self.root/'cache'/('.'+_id()+'.npz')
                try:
                    np.savez_compressed(temp,layers=layers,grid=grid);os.replace(temp,self.root/'cache'/(key+'.npz'))
                finally:temp.unlink(missing_ok=True)
                _json(self.root/'cache'/(key+'.json'),dict(version=VERSION,identity=identity,layer_sha256=pipeline.layer_sha256(layers),diagnostics=diagnostics))
            mapping=self._load(key,identity);identifier=_id()
            record=dict(id=identifier,target_id=target_id,reference_id=reference_id,analysis_time=time_seconds,
                codefingerprint=identity['codefingerprint'],cache_key=key,identity=identity,diagnostics=mapping.diagnostics,cache_hit=cache_hit)
            _json(self._record('mappings',identifier),record);return copy.deepcopy(record)

    def _grid(self,key):
        meta=json.loads((self.root/'cache'/(key+'.json')).read_text(encoding='utf-8'))
        with np.load(self.root/'cache'/(key+'.npz'),allow_pickle=False) as archive:grid=archive['grid'].copy()
        if grid.shape!=(1,13,8,16,16) or grid.dtype!=np.float32 or not np.isfinite(grid).all() or hashlib.sha256(grid.tobytes()).hexdigest()!=meta['diagnostics']['grid_sha256']:
            raise ValueError('局部调色缓存损坏，请重新分析')
        return grid

    def _validated_grid(self,mapping_id,verified_target=None):
        record=self.get_mapping_record(mapping_id)
        target=verified_target if verified_target is not None else self.get_media(record['target_id'])
        if target['id']!=record['target_id']:raise ValueError('Preview source does not match mapping')
        reference=self.get_media(record['reference_id']);identity,key=self._identity(target,reference,record['analysis_time'])
        if identity!=record['identity'] or key!=record['cache_key']:raise ValueError('调色模型或素材已更新，请重新分析')
        self._load(key,identity)
        return self._grid(key),record

    def get_prepared(self,mapping_id):
        grid,record=self._validated_grid(mapping_id)
        return PreparedSpatial(self.engine,torch.from_numpy(grid).to(self.engine.device),record['diagnostics']['analysis'])

    def render(self,mapping_id,rgb,strength,adjustments=None,preview=False,verified_target=None):
        strength=float(_strength(strength));settings=normalize(adjustments)
        grid,record=self._validated_grid(mapping_id,verified_target)
        key=(mapping_id,rgb.shape,hashlib.sha256(rgb.tobytes()).hexdigest())
        graded=None
        if strength==0 or record['diagnostics']['exact_identity']:graded=rgb
        elif preview:
            with self.preview_lock:
                if key in self.preview_cache:
                    graded=self.preview_cache.pop(key);self.preview_cache[key]=graded
        if graded is None:
            with self.render_lock:
                prepared=PreparedSpatial(self.engine,torch.from_numpy(grid).to(self.engine.device),record['diagnostics']['analysis'])
                graded=prepared.apply(rgb,1)
            if preview:
                with self.preview_lock:
                    self.preview_cache[key]=graded
                    while len(self.preview_cache)>2:self.preview_cache.popitem(last=False)
        return finish_spatial(rgb,graded,strength,settings)

    def save_project(self,mapping_id,strength,output_settings=None):
        if output_settings and output_settings.get('kind')=='cube':raise ValueError('局部调色无法完整保存为普通CUBE，请导出图片或视频')
        self.get_prepared(mapping_id)
        return super().save_project(mapping_id,strength,output_settings)
