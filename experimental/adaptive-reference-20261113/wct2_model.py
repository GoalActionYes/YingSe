"""Research-only adapter for pinned official WCT2 cat5 weights.

Upstream notices remain in the local source checkout. NAVER MIT-style files are
combined with NVIDIA CC-BY-NC-SA-4.0 utility code: do not claim wholly MIT.
No fine-tuning, no segmentation, no product integration in this adapter.
"""
import ast
import hashlib
import os
from pathlib import Path
import time
import types
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '.tools/WCT2-source'
HASHES = {
    'model.py': '8a7c69dd42aa537b3af648f6a554a5ad5112c97aa224e9afe57f1664edd30f25',
    'transfer.py': '6fdc16cd6715237d14b5e6f3aa7d811fb6bd6a709e7ced2999b56ac751e0cb68',
    'utils/core.py': '0d5cf4f5c5cb457e4ce34ae43f9dddecab2515d1294ac3511f4f1fdffb25c271',
    'utils/io.py': '6c529fada794f2c592d78eb305c2632ff5ca1a76d7719d36675e909acfd5bf63',
    'model_checkpoints/wave_encoder_cat5_l4.pth': 'ebd55e09aa4f4cf6d65a0b31651f4635d23f3f9724b54791b2f072e3b371725b',
    'model_checkpoints/wave_decoder_cat5_l4.pth': '85086ea2da11144999446874600c419bff49c218433d1b2a4ac315c9c491f151',
}


def verify_source():
    for relative, expected in HASHES.items():
        if hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('WCT2 source or checkpoint changed: '+relative)


def adapted_transfer_class():
    verify_source()
    tree = ast.parse((SOURCE / 'transfer.py').read_text(encoding='utf-8'))
    selected = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WCT2')
    for node in ast.walk(selected):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch.load':
            node.keywords.append(ast.keyword(arg='weights_only', value=ast.Constant(value=True)))
    return ast.unparse(ast.fix_missing_locations(selected))


def prepare_rgb(rgb, device, max_edge=640):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or max_edge < 16:
        raise ValueError('Expected RGB uint8 image and max_edge >=16')
    im = Image.fromarray(rgb)
    im.thumbnail((max_edge, max_edge), Image.Resampling.BILINEAR)
    working = np.array(im)
    h, w = working.shape[:2]
    if min(h, w) < 16:
        raise ValueError('Working image too narrow for three wavelet levels')
    x = torch.from_numpy(working.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)/255
    return F.pad(x, (0, (-w)%16, 0, (-h)%16), mode='replicate'), working


def restore_rgb(tensor, shape):
    if tensor.ndim != 4 or tensor.shape[0:2] != (1, 3) or not torch.isfinite(tensor).all():
        raise ValueError('Invalid WCT2 output')
    h, w = shape[:2]
    if tensor.shape[-2] < h or tensor.shape[-1] < w:
        raise ValueError('Decoded output is smaller than working image')
    return tensor[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()


def _execute(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
    return module


class WCT2Model:
    def __init__(self, device='cuda', max_edge=640):
        verify_source()
        architecture = _execute(SOURCE / 'model.py', 'wct2_pinned_architecture')
        transforms = _execute(SOURCE / 'utils/core.py', 'wct2_pinned_core')
        io_tree = ast.parse((SOURCE / 'utils/io.py').read_text(encoding='utf-8'))
        label_fn = next(n for n in io_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_label_info')
        namespace = {'np': np, 'torch': torch, 'os': os, 'WaveEncoder': architecture.WaveEncoder,
                     'WaveDecoder': architecture.WaveDecoder, 'feature_wct': transforms.feature_wct}
        exec(compile(ast.Module(body=[label_fn], type_ignores=[]), '<pinned-label-check>', 'exec'), namespace)
        exec(compile(adapted_transfer_class(), '<pinned-wct2-transfer>', 'exec'), namespace)
        self.device = torch.device(device)
        self.model = namespace['WCT2'](model_path=str(SOURCE / 'model_checkpoints'),
                                      transfer_at=['encoder', 'skip', 'decoder'], option_unpool='cat5', device=device)
        self.model.encoder.eval()
        self.model.decoder.eval()
        self.max_edge = max_edge

    @torch.inference_mode()
    def predict(self, target, reference):
        x, working = prepare_rgb(target, self.device, self.max_edge)
        r, _ = prepare_rgb(reference, self.device, self.max_edge)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output = self.model.transfer(x, r, np.empty(0), np.empty(0), alpha=1)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        rgb = restore_rgb(output, working.shape)
        return rgb, {'forward_seconds': elapsed, 'working_shape': list(working.shape),
                     'padded_shape': list(x.shape), 'input_resized': working.shape != target.shape,
                     'transfer_at': ['encoder', 'skip', 'decoder'], 'alpha': 1,
                     'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if self.device.type == 'cuda' else None}
