"""Final author SSL model, separate from the E2E research-workbench backend."""
import subprocess
import sys
import torch
from types import SimpleNamespace
from canonical_model import ROOT,CanonicalModel,UPSTREAM_COMMIT,NETWORK
from independent_coco import digest
from prepare_canonical_final import SSL_SHA


def load_final_state(model,payload):
    if not isinstance(payload,dict) or not isinstance(payload.get('model_state_dict'),dict):
        raise ValueError('Expected exact final checkpoint state dictionary')
    state=payload['model_state_dict']; expected=model.state_dict()
    if set(state)!=set(expected): raise ValueError('Missing or extra final inference state keys')
    for name,value in state.items():
        if (not isinstance(value,torch.Tensor) or value.shape!=expected[name].shape
                or value.dtype!=expected[name].dtype or not torch.isfinite(value).all()):
            raise ValueError('Invalid final inference tensor: '+name)
    model.load_state_dict(state,strict=True)


class CanonicalFinal(CanonicalModel):
    def __init__(self,device=None):
        upstream=ROOT/'.tools/CanonCGT'
        commit=subprocess.run(['git','-C',str(upstream),'rev-parse','HEAD'],check=True,capture_output=True,text=True,timeout=15).stdout.strip()
        if commit!=UPSTREAM_COMMIT or subprocess.run(['git','-C',str(upstream),'diff','--quiet','HEAD'],timeout=15).returncode!=0:
            raise ValueError('Pinned author source differs')
        checkpoint=ROOT/'.tools/CanonCGT-final/SSL_updated_251111.pth'
        if digest(checkpoint)!=SSL_SHA: raise ValueError('Final checkpoint checksum mismatch')
        sys.path.insert(0,str(upstream))
        from models.networks.Estimator import Estimator_blocks as blocks
        from models.networks.SSL_training import CanonCGT_SSL
        original=blocks.mobilenet_v2
        blocks.mobilenet_v2=lambda weights=None: original(weights=None)
        try: self.model=CanonCGT_SSL(SimpleNamespace(network=NETWORK))
        finally: blocks.mobilenet_v2=original
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        load_final_state(self.model,payload)
        self.torch=torch; self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device).eval()
        self.provenance=dict(author_repo='https://github.com/Jinwon-Ko/CanonCGT',author_commit=UPSTREAM_COMMIT,
            stage='Author final self-supervised refinement',checkpoint_name=checkpoint.name,checkpoint_sha256=SSL_SHA,
            source_download='Official README public Drive package',all_inference_weights_loaded=True,
            inference_state_keys=len(self.model.state_dict()),epoch=payload.get('epoch'),
            torch_version=torch.__version__,device=str(self.device),
            limitation='Two global LUT stages, not per-region semantic grading or recovered Jianying code')
