"""Pinned, strict, inference-only SA-LUT research adapter (not product backend).

Uses the author's S-Lab non-commercial licensed source from a local checkout.
Source is not copied into this module. Preserve the upstream LICENSE when sharing
that checkout. All inference parameters must load; no random encoder fallback.
"""
import ast
import hashlib
import math
from pathlib import Path
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from salut_interpolation import TrilinearInterpolation, QuadrilinearInterpolation4D, efficient_attention

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '.tools/SA-LUT-source/SA-LUT'
CHECKPOINT = ROOT / '.downloads/salut/epoch=100-step=4127466.ckpt.state.pt'
CHECKPOINT_SHA = '5ad3685cf6bbf14230d71580b9b93f03e76bf6d7dfaceecd0a3527341cc92f2b'
SOURCE_HASHES = {
    'core/module/model.py': '296a544acbac514b7de190acf3b30a3d40e5cb8a81aa7ba55ebb16c06a7dd198',
    'core/module/net.py': 'c2f202eb30db82721296af272fc077b8e90f3584de3153d4281ace111adbba39',
    'core/module/clut4d.py': '05c2dbe7ca4cc5f7c82e3bf8a03e1f34384a3db028248740ffcdd47273722ef6',
    'core/dataset/utils.py': '1a790a17453ed2320b6fa3fb7e3930ff359e093e4eb65905dd2286f31f4446c6',
    'core/assets/Standard.cube': 'f99223675b29933952da2153bdb3137dd749d12964d0753db85e47576ca4578d',
}


def verify_source(root):
    for relative, expected in SOURCE_HASHES.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'SA-LUT source changed: {relative}')


def adapted_model_source(root):
    verify_source(root)
    source = (root / 'core/module/model.py').read_text(encoding='utf-8')
    old_load = '        vgg.load_state_dict(torch.load("ckpts/vgg_normalised.pth", weights_only=False))'
    old_attention = '''        attn_scores = torch.bmm(Q_flat, K_flat)
        attn_scores = attn_scores / (C_attn**0.5) * self.attn_temperature
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention
        aggregated = torch.bmm(attn_weights, V_flat)'''
    for old, new in ((old_load, '        # Encoder loaded strictly from complete inference checkpoint.'),
                     (old_attention, '        aggregated = efficient_attention(Q_flat, K_flat, V_flat, self.attn_temperature)')):
        if source.count(old) != 1:
            raise ValueError('Pinned source adaptation no longer matches')
        source = source.replace(old, new)
    tree = ast.parse(source)
    kept = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == 'quadrilinear4d' for a in node.names):
            continue
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('core.'):
            continue
        if isinstance(node, ast.ClassDef) and node.name in ('QuadrilinearInterpolation_Function', 'QuadrilinearInterpolation_4D'):
            continue
        if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test):
            continue
        kept.append(node)
    tree.body = kept
    return ast.unparse(tree)


def filter_inference_state(state):
    allowed = ('vlog2stylenet.', 'style_discriminator.', 'lpips_loss.', 'vgg_loss.')
    if any(not key.startswith(allowed) for key in state):
        raise ValueError('Unknown checkpoint namespace')
    result = {k.removeprefix('vlog2stylenet.'): v for k, v in state.items() if k.startswith('vlog2stylenet.')}
    if not any(k.startswith('encoder.enc_') for k in result):
        raise ValueError('Complete encoder required')
    return result


def _execute_file(path, module_name):
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
    return module


def load_inference_weights(model, state):
    expected = model.state_dict()
    missing = set(expected)-set(state)
    # This upstream checkpoint predates a deterministic loss-weight buffer.
    # It only weights the discarded TV training loss, never LUTs/context/output.
    allowed = {'tvmn.weight_c'}
    if missing-allowed or set(state)-set(expected):
        raise ValueError(f'Incomplete inference checkpoint: missing={sorted(missing)}, extra={sorted(set(state)-set(expected))}')
    if missing & set(dict(model.named_parameters())):
        raise ValueError('A learned parameter may never be restored from initialization')
    model.load_state_dict({**state, **{k: expected[k] for k in missing}}, strict=True)
    return sorted(missing)


class SALUTModel:
    def __init__(self, device='cuda', source=SOURCE, checkpoint=CHECKPOINT):
        verify_source(source)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA:
            raise ValueError('Checkpoint checksum mismatch')
        state = filter_inference_state(torch.load(checkpoint, map_location='cpu', weights_only=True))
        net = _execute_file(source / 'core/module/net.py', 'salut_pinned_net')
        clut = _execute_file(source / 'core/module/clut4d.py', 'salut_pinned_clut')
        utils_tree = ast.parse((source / 'core/dataset/utils.py').read_text(encoding='utf-8'))
        reader = next(n for n in utils_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'read_3dlut_from_file')
        reader_namespace = {'np': np, 'torch': torch, 'math': math}
        exec(compile(ast.Module(body=[reader], type_ignores=[]), '<pinned-cube-reader>', 'exec'), reader_namespace)
        namespace = {
            '__file__': str(source / 'core/module/model.py'), '__name__': 'salut_pinned_model',
            'net': net, 'CLUT4D': clut.CLUT4D, 'TV_4D': clut.TV_4D,
            'identity4d_tensor': clut.identity4d_tensor,
            'read_3dlut_from_file': reader_namespace['read_3dlut_from_file'],
            'TrilinearInterpolation': TrilinearInterpolation,
            'QuadrilinearInterpolation_4D': QuadrilinearInterpolation4D,
            'efficient_attention': efficient_attention,
        }
        exec(compile(adapted_model_source(source), '<pinned-salut-model>', 'exec'), namespace)
        model = namespace['VLog2StyleNet4D'](dim=17, num_basis=64)
        self.restored_loss_buffers = load_inference_weights(model, state)
        self.loaded_keys = len(state)
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)

    def _tensor(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected RGB uint8 image')
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)/255

    @torch.inference_mode()
    def predict(self, target, reference):
        content = self._tensor(target)
        style = self._tensor(reference)
        # Preserve author's inference CLI preprocessing and full-res context upsampling.
        content_small = F.interpolate(content, (512, 512), mode='bilinear', align_corners=True)
        style_small = F.interpolate(style, (512, 512), mode='bilinear', align_corners=True)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        _, lut, context = self.model(style_small, content_small)
        full_context = F.interpolate(context, target.shape[:2], mode='bilinear', align_corners=True)
        _, output = QuadrilinearInterpolation4D()(lut[0], torch.cat((full_context, content), dim=1))
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        rgb = output[0].clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
        diagnostics = {'forward_seconds': elapsed, 'loaded_keys': self.loaded_keys,
                       'restored_training_loss_buffers': self.restored_loss_buffers,
                       'context_min': float(context.min()), 'context_max': float(context.max()),
                       'context_std': float(context.std()), 'lut_shape': list(lut.shape),
                       'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if self.device.type == 'cuda' else None}
        self.last_lut, self.last_context = lut, context
        return rgb, diagnostics
