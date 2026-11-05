"""Additional diagnostic controls for the throwaway region-matching probe."""
from pathlib import Path
import sys,json,hashlib,time
sys.path.insert(0,str(Path(__file__).parent))
import run_probe as probe
import numpy as np
from PIL import Image
target=probe.media.decode_image(probe.OUT/'target.png')
reference=probe.media.decode_image(probe.OUT/'reference-controlled.png')
ones=np.ones(target.shape[:2],np.float32)
dense_global=probe.build_mapping(probe.extract_region(target,0,ones),probe.extract_region(reference,1,ones))
dense_result=probe.apply_mapping(target,dense_global,.8)
mask=probe.media.decode_image(probe.OUT/'person-mask.png')[...,0]/255
metrics={}
for label,selection in [('person',mask>.9),('background',mask<.1)]:
    metrics[label]=float(np.abs(dense_result.astype(float)-reference)[selection].mean())
identity=probe.fit(target,target)
probe.RegionalSession.mapping=identity
identity_result,_=probe.RegionalSession().process(target,target)
identity_global=probe.apply_mapping(target,identity['global_map'],.8)
probe.save('same-reference-regional.png',identity_result)
results=dict(same_sampling_global_reference_mae=metrics,
    same_reference_identity_mae=dict(global_mapping=float(np.abs(identity_global.astype(float)-target).mean()),
                                     regional_mapping=float(np.abs(identity_result.astype(float)-target).mean())),
    interpretation='Same-reference nonzero drift is measured, not treated as an identity guarantee. Sampling policy differs from the shipped baseline; report the dense whole-image control separately.')
baseline=json.loads((probe.ROOT/'work/skin-probe-20261019/stable-baseline-hashes.json').read_text('utf8'))
assert all(hashlib.sha256((probe.ROOT/name).read_bytes()).hexdigest()==digest for name,digest in baseline.items())
results['stable_frozen_files_unchanged']=len(baseline)
(probe.OUT/'controls.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),'utf8')
print(json.dumps(results,indent=2))
