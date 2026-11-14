"""Optional original grade encoder only; no canonicalizer or LUT inference here."""
import torch
from training_checks import finite_output


class FrozenGradeEncoder(torch.nn.Module):
    def __init__(self,encoder,provenance):
        super().__init__()
        if not isinstance(encoder,torch.nn.Module) or not list(encoder.parameters()):raise ValueError('Expected an encoder with parameters')
        self.encoder=encoder;self.provenance=dict(provenance);self.requires_grad_(False);self.eval()

    def train(self,mode=True):
        if type(mode)!=bool:raise ValueError('Expected boolean mode')
        return super().train(False)

    @torch.no_grad()
    def forward(self,rgb):
        p=next(self.encoder.parameters())
        if not isinstance(rgb,torch.Tensor) or rgb.ndim!=4 or rgb.shape[1]!=3 or min(rgb.shape)<1 or rgb.dtype!=p.dtype or rgb.device!=p.device or not torch.isfinite(rgb).all() or (rgb<0).any() or (rgb>1).any():raise ValueError('Expected finite unit RGB BCHW on encoder device')
        return finite_output(self.encoder(rgb),(rgb.shape[0],64)).detach()


def load_frozen_grade_encoder(device='cuda'):
    from canonical_model import CanonicalModel,ROOT,CHECKPOINT_SHA,UPSTREAM_COMMIT
    from independent_coco import digest
    original=CanonicalModel(device=device);upstream=ROOT/'.tools/CanonCGT'
    names=['README.md','LICENSE','models/networks/Estimator/Estimator_blocks.py','models/networks/Estimator/Estimator_modules.py','models/networks/end_to_end_finetuning.py','configs/Stage1_style_encoder.yaml','configs/Stage2_end_to_end_finetuning.yaml']
    provenance=dict(original.provenance,feature='Unmodified original Embedding_Net only; no canonicalization or LUT inference',input='author bilinear224 RGB0..1, mean0/std1, no ImageNet renormalization',dimensions=64,checkpoint_sha256=CHECKPOINT_SHA,author_commit=UPSTREAM_COMMIT,upstream_source_sha256={name:digest(upstream/name) for name in names},loader_sha256=digest(ROOT/'experimental/adaptive-reference-20261113/canonical_model.py'),known_source_overlap='FiveK source overlaps TST100K; exact photograph/preset overlap unresolved; no independent generalization claim')
    return FrozenGradeEncoder(original.model.Embedding_Net,provenance)
