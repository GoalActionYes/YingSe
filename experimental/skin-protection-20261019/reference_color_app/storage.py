"""Owned immutable inputs and versioned, verified mapping persistence."""
import copy, hashlib, json, math, os, shutil, threading, uuid, zipfile, zlib
from pathlib import Path
import numpy as np
from . import pipeline, media
from .contracts import Mapping

VERSION=1

def _id(): return uuid.uuid4().hex

def _json(path, value):
    temp=path.with_name('.'+_id()+'.tmp')
    try:
        with temp.open('w',encoding='utf-8') as f:
            json.dump(pipeline.json_safe(value),f,allow_nan=False,ensure_ascii=False); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    finally: temp.unlink(missing_ok=True)

class Store:
    def __init__(self,root):
        self.root=Path(root).resolve(); self.lock=threading.RLock()
        for part in ('media','mappings','cache','projects','jobs'): (self.root/part).mkdir(parents=True,exist_ok=True)
    def _record(self,folder,identifier):
        if not isinstance(identifier,str) or len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier): raise ValueError('Invalid ID')
        return self.root/folder/(identifier+'.json')
    def add_media(self,path,name=None):
        source=Path(path); identifier=_id(); suffix='.input'
        dest=self.root/'media'/(identifier+suffix); temp=dest.with_suffix('.tmp')
        try:
            shutil.copyfile(source,temp); metadata=media.probe_media(temp)
            digest=pipeline.file_sha256(temp); os.replace(temp,dest)
            record=dict(id=identifier,name=str(name or source.name),kind=metadata['kind'],sha256=digest,metadata=metadata,path=str(dest))
            _json(self._record('media',identifier),record); return copy.deepcopy(record)
        except Exception:
            temp.unlink(missing_ok=True); dest.unlink(missing_ok=True); raise
    def get_media(self,media_id):
        record=json.loads(self._record('media',media_id).read_text(encoding='utf-8'))
        path=Path(record['path']).resolve()
        if path.parent!=self.root/'media' or not path.is_file() or pipeline.file_sha256(path)!=record['sha256']: raise ValueError('Owned media missing or modified')
        return record
    def _identity(self,target,reference,time_seconds):
        decoder={'target':media.probe_media(target['path']),'reference':media.probe_media(reference['path'])}
        # FFmpeg diagnostic text can contain process-specific pointer addresses.
        for metadata in decoder.values(): metadata.pop('metadata_text',None)
        if target['kind']=='video': decoder['ffmpeg_sha256']=pipeline.file_sha256(media.ffmpeg_path())
        identity=dict(version=VERSION,target=target['sha256'],reference=reference['sha256'],time_seconds=time_seconds,codefingerprint=pipeline.algorithm_fingerprint(),decoder=decoder)
        return identity,hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
    def _load(self,key,expected=None):
        meta=json.loads((self.root/'cache'/(key+'.json')).read_text(encoding='utf-8'))
        if meta['version']!=VERSION or (expected is not None and meta['identity']!=expected): raise ValueError('Stale cache')
        with np.load(self.root/'cache'/(key+'.npz'),allow_pickle=False) as data: layers=data['layers'].copy()
        if layers.shape!=(2,64,64,64,3) or layers.dtype!=np.float32 or not np.isfinite(layers).all() or np.any(layers<0) or np.any(layers>1) or pipeline.layer_sha256(layers)!=meta['layer_sha256']: raise ValueError('Invalid mapping cache')
        layers.setflags(write=False)
        return Mapping(layers,meta['diagnostics'])
    def analyze(self,target_id,reference_id,time_seconds=0.0):
        if isinstance(time_seconds,(bool,str)) or not math.isfinite(float(time_seconds)) or float(time_seconds)<0: raise ValueError('Invalid analysis time')
        time_seconds=float(time_seconds)
        with self.lock:
            target=self.get_media(target_id); reference=self.get_media(reference_id)
            identity,key=self._identity(target,reference,time_seconds)
            try: mapping=self._load(key,identity)
            except (OSError,ValueError,KeyError,TypeError,EOFError,zipfile.BadZipFile,zlib.error):
                mapping=pipeline.analyze(target['path'],reference['path'],time_seconds)
                temp=self.root/'cache'/('.'+_id()+'.npz')
                try:
                    np.savez_compressed(temp,layers=np.asarray(mapping.layers,dtype=np.float32)); os.replace(temp,self.root/'cache'/(key+'.npz'))
                finally: temp.unlink(missing_ok=True)
                _json(self.root/'cache'/(key+'.json'),dict(version=VERSION,identity=identity,layer_sha256=pipeline.layer_sha256(mapping.layers),diagnostics=mapping.diagnostics))
            identifier=_id(); record=dict(id=identifier,target_id=target_id,reference_id=reference_id,analysis_time=time_seconds,codefingerprint=identity['codefingerprint'],cache_key=key,identity=identity,diagnostics=mapping.diagnostics)
            _json(self._record('mappings',identifier),record); return copy.deepcopy(record)
    def get_mapping_record(self,mapping_id):
        return json.loads(self._record('mappings',mapping_id).read_text(encoding='utf-8'))
    def get_mapping(self,mapping_id):
        record=self.get_mapping_record(mapping_id)
        identity,key=self._identity(self.get_media(record['target_id']),self.get_media(record['reference_id']),record['analysis_time'])
        if key!=record['cache_key'] or identity!=record['identity']: raise ValueError('Mapping is stale; analyze again')
        return self._load(key,identity)
    def save_project(self,mapping_id,strength,output_settings=None):
        from .lut import _strength
        strength=float(_strength(strength)); self.get_mapping(mapping_id)
        record=dict(id=_id(),version=VERSION,mapping=self.get_mapping_record(mapping_id),strength=strength,output_settings=output_settings or {})
        _json(self._record('projects',record['id']),record); return copy.deepcopy(record)
