"""Opt-in research adapter. Disabled/rejected inputs use the real baseline.

No App defaults change. This is not a complete native-engine reconstruction.
"""
from pathlib import Path
import sys,math,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'skin-protection-20261019'))
from reference_color_app import pipeline as baseline
from reference_color_app import media
from reference_color_app.contracts import Mapping

VERSION='baseline-preserving-composition-adapter-v1'

def fingerprint():
 h=hashlib.sha256()
 for name in ('candidate_pipeline.py','person_gate.py','composition.py','dense_features.py','balance.py'):
  h.update(name.encode()+b'\0'+Path(__file__).with_name(name).read_bytes())
 return h.hexdigest()

def analyze(target_path,reference_path,time_seconds=0.,*,enabled=False):
 if type(enabled) is not bool:raise ValueError('enabled must be a boolean')
 try:valid_time=not isinstance(time_seconds,(bool,str)) and math.isfinite(float(time_seconds)) and time_seconds>=0
 except (ValueError,TypeError,OverflowError):valid_time=False
 if not valid_time:
  raise ValueError('Invalid analysis time')
 if not enabled:return baseline.analyze(target_path,reference_path,time_seconds)
 from PIL import Image
 import numpy as np
 from person_gate import segment,fullmask
 from composition import extract_pair
 from reference_color_app.lut import build_mapping
 if media.probe_media(reference_path)['kind']!='image':raise ValueError('The reference must be an image')
 tm=media.probe_media(target_path)
 a=media.decode_image(target_path) if tm['kind']=='image' else media.decode_frame(target_path,time_seconds)[0]
 b=media.decode_image(reference_path)
 def shrink(rgb):
  im=Image.fromarray(rgb);im.thumbnail((640,640));return np.asarray(im)
 a,b=shrink(a),shrink(b)
 am,_,ad=segment(a);bm,_,bd=segment(b)
 context=dict(version=VERSION,enabled=True,target_evidence=ad,reference_evidence=bd,
              candidate_fingerprint=fingerprint(),native_byte_compatible=False)
 if ad['person_detected'] and bd['person_detected']:
  source,reference,diag=extract_pair(a,b,fullmask(am,a),fullmask(bm,b),.5)
  context['composition']=diag
  if diag['applied']:
   mapping=build_mapping(source,reference)
   return Mapping(mapping.layers,baseline.json_safe(dict(mapping.diagnostics,candidate=dict(context,applied=True),
     source_packet=source.values,reference_packet=reference.values,
     analysis_max_edge=640,layer_sha256=baseline.layer_sha256(mapping.layers))))
 original=baseline.analyze(target_path,reference_path,time_seconds)
 return Mapping(original.layers,dict(original.diagnostics,candidate=dict(context,applied=False,
   fallback='Original baseline pipeline; missing eligibility, reliable regions, or composition difference')))
