"""Optional, isolated author-model baseline. Not an integrated YingSe engine.

Uses official CanonCGT E2E checkpoint, pinned by hash. No unsafe pickle load,
no partial model loading, no downloads at inference time. Dependency setup is
separate in .tools/adaptive-env; stable app requirements remain unchanged.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT_SHA='916f7ad5028d3fef51cf915bb51febc9508e7a378c0babe9ff3d3992b8c594f7'
UPSTREAM_COMMIT='229a7d3ec24af19ea5d97bb136f0317762bb9261'
NETWORK=dict(n_points=17,hidden_dim=64,style_token_dim=64,n_down_layer=3,n_attn_module=3,
             n_enc_block=[2,2,2],n_dec_block=[2,2,2],n_enc_heads=[4,4,4],n_dec_heads=[1,1,1])


def validate_rgb(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
        raise ValueError('Expected nonempty RGB8')


class CanonicalModel:
    def __init__(self,device=None):
        import sys
        import subprocess
        import torch
        upstream=ROOT/'.tools/CanonCGT'
        commit=subprocess.run(['git','-C',str(upstream),'rev-parse','HEAD'],check=True,
                              capture_output=True,text=True,timeout=15).stdout.strip()
        if commit!=UPSTREAM_COMMIT:
            raise ValueError('Author source revision differs from the reviewed baseline')
        dirty=subprocess.run(['git','-C',str(upstream),'diff','--quiet','HEAD'],timeout=15)
        if dirty.returncode!=0:
            raise ValueError('Author tracked source has local modifications')
        checkpoint=upstream/'pretrained/E2E.pth'
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA:
            raise ValueError('Author checkpoint checksum mismatch')
        sys.path.insert(0,str(upstream))
        from models.networks.Estimator import Estimator_blocks as blocks
        from models.networks.end_to_end_finetuning import CanonCGT_E2E
        self.torch=torch
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        # The complete author checkpoint supplies backbone weights. Avoid the
        # redundant ImageNet download in the upstream model constructor.
        original=blocks.mobilenet_v2
        blocks.mobilenet_v2=lambda weights=None: original(weights=None)
        try:
            self.model=CanonCGT_E2E(SimpleNamespace(network=NETWORK))
        finally:
            blocks.mobilenet_v2=original
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or 'model_state_dict' not in state:
            raise ValueError('Unexpected author checkpoint structure')
        weights=state['model_state_dict']
        expected=set(self.model.state_dict())
        missing=expected-set(weights)
        extra=set(weights)-expected
        if missing or any(not key.startswith('style_centroids.') for key in extra):
            raise ValueError(f'Incomplete model state: missing={sorted(missing)}, extra={sorted(extra)}')
        self.model.load_state_dict({key:weights[key] for key in expected},strict=True)
        self.model.to(self.device).eval()
        self.provenance=dict(author_repo='https://github.com/Jinwon-Ko/CanonCGT',
                             author_commit=UPSTREAM_COMMIT,stage='E2E, not final SSL',
                             checkpoint_sha256=CHECKPOINT_SHA,ignored_centroid_keys=sorted(extra),
                             torch_version=torch.__version__,device=str(self.device),
                             all_inference_weights_loaded=True,
                             limitation='Two learned global LUT stages; does not establish region-dependent mapping')

    def _tensor(self,rgb):
        validate_rgb(rgb)
        return self.torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1)[None].float().to(self.device)/255

    def predict(self,target,reference):
        torch=self.torch
        x=self._tensor(target); r=self._tensor(reference)
        if self.device.type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            outputs=self.model(x,r)
            result=outputs['restyled']
        if self.device.type=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        if result.shape!=x.shape or not torch.isfinite(result).all():
            raise ValueError('Invalid author model output')
        self.last_luts=(outputs['canonicalize_LUT'].detach(),outputs['restylize_LUT'].detach())
        rgb=result[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()
        return rgb,dict(forward_seconds=seconds,includes_input_upload=False,includes_output_download=False)
