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
import numpy as np
from PIL import Image,ImageOps,ImageCms
import torch
from reference_color_app import media,pipeline
from reference_color_app.storage import Store,VERSION,_id,_json
from reference_color_app.contracts import Mapping
from reference_color_app.adjustments import normalize,transform
from reference_color_app.lut import _strength
from learned_store import identity_cube
from spatial_render import PreparedSpatial,CHECKPOINT_SHA
from triplet_color import to_srgb


class SpatialStore(Store):
    def __init__(self,root,engine,identity_tag=None):
        super().__init__(root);self.engine=engine;self.render_lock=threading.RLock();self.preview_cache=OrderedDict()
        names=['spatial_render.py','spatial_store.py','triplet_color.py','training_checks.py']
        self.engine_tag=identity_tag or hashlib.sha256((CHECKPOINT_SHA+''.join(pipeline.file_sha256(Path(__file__).with_name(n)) for n in names)).encode()).hexdigest()

    def _identity(self,target,reference,time_seconds):
        identity,_=super()._identity(target,reference,time_seconds)
        identity.update(spatial_engine=self.engine_tag)
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
            except (ValueError,OSError,KeyError,EOFError):
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

    def get_prepared(self,mapping_id):
        self.get_mapping(mapping_id);record=self.get_mapping_record(mapping_id);grid=self._grid(record['cache_key'])
        return PreparedSpatial(self.engine,torch.from_numpy(grid).to(self.engine.device),record['diagnostics']['analysis'])

    def render(self,mapping_id,rgb,strength,adjustments=None,preview=False):
        strength=float(_strength(strength));settings=normalize(adjustments)
        prepared=self.get_prepared(mapping_id);record=self.get_mapping_record(mapping_id)
        key=(mapping_id,rgb.shape,hashlib.sha256(rgb.tobytes()).hexdigest())
        with self.render_lock:
            if strength==0 or record['diagnostics']['exact_identity']:graded=rgb
            elif preview and key in self.preview_cache:
                graded=self.preview_cache.pop(key);self.preview_cache[key]=graded
            else:
                graded=prepared.apply(rgb,1)
                if preview:
                    self.preview_cache[key]=graded
                    while len(self.preview_cache)>2:self.preview_cache.popitem(last=False)
            value=rgb.astype(np.float32)/255*(1-strength)+graded.astype(np.float32)/255*strength
            return np.rint(transform(value,settings)*255).astype(np.uint8)

    def save_project(self,mapping_id,strength,output_settings=None):
        if output_settings and output_settings.get('kind')=='cube':raise ValueError('局部调色无法完整保存为普通CUBE，请导出图片或视频')
        self.get_prepared(mapping_id)
        return super().save_project(mapping_id,strength,output_settings)
