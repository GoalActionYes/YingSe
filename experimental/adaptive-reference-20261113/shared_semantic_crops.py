"""Ablation: one full-target canonicalization, two reference-conditioned restylers."""
import time
import numpy as np
from canonical_model import validate_rgb
from semantic_crops import choose_crops,blend_residual,valid_probability


class SharedCanonicalCandidate:
    def __init__(self,model):self.model=model

    def predict(self,target,reference,target_probability,reference_probability):
        start=time.perf_counter()
        valid_probability(target,target_probability);valid_probability(reference,reference_probability)
        if np.array_equal(target,reference):
            return target.copy(),target.copy(),dict(applied=False,exact_identity=True,seconds=time.perf_counter()-start)
        global_rgb,_=self.model.predict(target,reference)
        source=choose_crops(target,target_probability);ref=choose_crops(reference,reference_probability)
        diagnostic=dict(target_crops=source,reference_crops=ref,applied=False,
                        scope='Shared canonicalization with reference crop style; unvalidated regional adaptation')
        if not source['accepted'] or not ref['accepted']:
            return global_rgb,global_rgb,dict(diagnostic,seconds=time.perf_counter()-start)
        torch=self.model.torch;network=self.model.model
        outputs=[]
        with torch.inference_mode():
            neutral=network.TrilinearInterpolation(self.model._tensor(target),self.model.last_luts[0])
            for role in ('foreground','background'):
                left,top,right,bottom=ref[role]
                condition=network.Embedding_Net(self.model._tensor(reference[top:bottom,left:right]))
                result=network.Restyler(neutral,condition=condition)['result']
                if not torch.isfinite(result).all():raise ValueError('Invalid regional restyler output')
                outputs.append(result[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy())
        result=blend_residual(global_rgb,*outputs,target_probability)
        return result,global_rgb,dict(diagnostic,applied=True,seconds=time.perf_counter()-start)
