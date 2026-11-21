"""Check VCG's repacked checkpoint against the authors' model family without loading it."""

from argparse import ArgumentParser
from hashlib import sha1, sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch
from accelerate.utils import set_module_tensor_to_device
import huggingface_hub


AUTHOR_FILES = {
    'LICENSE': '261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64',
    'models/ReferenceNet.py': '2bca8c0106f238d6e3f9c47001d444b1e30e2b76',
    'models/ReferenceNet_attention.py': '6465ad97b998a39c61bb1258e2eb65389766432b',
    'models/ReferenceNet_attention_diff.py': '30ae357ec0153831496334396aabe1ad3ad777db',
    'models/attention.py': 'b17416e5068562b7ad74eb5ce5ebeb71b3ef1ca9',
    'models/ImageEncoder.py': '83ab8c9769e99574df165fd52e3401ae187b328a',
    'pipeline.py': '6fbe3168e8275379433aa6ff14b00056a4a150de',
    'configs/inference/inference.yaml': '9fbb99e44c56174942c57f8ead0180d695465bd9',
}
HEADER = 'ae5f5dba740c72909745d34ae464d5f69d50f8b46b1092a3d93b43067bd6d5b8'
DIFFUSERS = '00d63135ac78a670cdacfe2a1c921d3cf75fe4f248202f3fe947121b254ad440'
SAMPLES = {
    'clip': ('clip_vision.vision_model.embeddings.class_embedding',
             'e1faf3ae0a580864507394b2220d3d9032f21e954ef2045f53f388023cece0b2'),
    'reference': ('referencenet.conv_in.bias',
                  '10eedd6c2fd3b214bfccca6fe49143c39965ab139f7591c6946b30eb65b6cfcb'),
    'diffuser': ('l_diffuser.conv_in.bias',
                 '66d71c9a9d6f010fe5ebb95ff01ebc0de9b8bb7c9bf4f926962f423e27000a8d'),
}
VAE_NAMES = {'query': 'to_q', 'key': 'to_k', 'value': 'to_v', 'proj_attn': 'to_out.0'}


def offline_download(*args, **kwargs):
    raise RuntimeError('The model loading preflight only uses local files')


def create_models(source, author):
    wheel = source / 'diffusers-0.21.4-py3-none-any.whl'
    if sha256(wheel.read_bytes()).hexdigest() != DIFFUSERS:
        raise ValueError('Unexpected Diffusers package')
    for relative, expected in AUTHOR_FILES.items():
        content = (author / relative).read_bytes()
        prefix = f'blob {len(content)}'.encode() + bytes(1)
        if sha1(prefix + content).hexdigest() != expected:
            raise ValueError(f'Unexpected upstream blob: {relative}')
    sys.path.insert(0, str(source / 'vendor-0214'))
    sys.path.insert(0, str(author))
    # Diffusers 0.21 imports this retired symbol. This audit never calls it.
    huggingface_hub.cached_download = offline_download
    import diffusers
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    from models.ReferenceNet import ReferenceNet
    from pipeline import InferencePipeline

    if diffusers.__version__ != '0.21.4':
        raise ValueError('Wrong isolated Diffusers version')
    if InferencePipeline.__name__ != 'InferencePipeline':
        raise ValueError('Unexpected author pipeline')
    unet = dict(sample_size=64, cross_attention_dim=768)
    vae = dict(
        down_block_types=('DownEncoderBlock2D',) * 4,
        up_block_types=('UpDecoderBlock2D',) * 4,
        block_out_channels=(128, 256, 512, 512), layers_per_block=2,
        latent_channels=4, sample_size=512,
    )
    with torch.device('meta'):
        return {
            'clip_vision': CLIPVisionModelWithProjection(CLIPVisionConfig(
                hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                num_attention_heads=12, image_size=224, patch_size=32,
            )),
            'vae': AutoencoderKL(**vae),
            'referencenet': ReferenceNet(**unet),
            'l_diffuser': UNet2DConditionModel(**unet, in_channels=6, out_channels=3),
        }


def compare(model, component, index):
    prefix = component + '.'
    expected = model.state_dict()
    found = {}
    aliases = {}
    wrong_shape = {}
    for name, info in index.items():
        if not name.startswith(prefix):
            continue
        local = name.removeprefix(prefix)
        if component == 'vae' and local not in expected and '.attentions.' in local:
            path, module, suffix = local.rsplit('.', 2)
            if module in VAE_NAMES:
                renamed = f'{path}.{VAE_NAMES[module]}.{suffix}'
                aliases[local] = renamed
                local = renamed
        if local in found:
            raise ValueError(f'Non-injective key translation: {name}')
        found[local] = info
        if local in expected and list(expected[local].shape) != info['shape']:
            wrong_shape[local] = {'model': list(expected[local].shape), 'file': info['shape']}
    return {
        'checkpoint_tensors': len(found), 'model_tensors': len(expected),
        'aliases': aliases, 'missing': sorted(expected.keys() - found.keys()),
        'unexpected': sorted(found.keys() - expected.keys()),
        'wrong_shape': wrong_shape,
    }


def verify_cuda(source, index, models):
    samples = {}
    for label, (key, expected) in SAMPLES.items():
        path = source / f'{label}.tensor'
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Unexpected tensor receipt: {key}')
        info = index[key]
        data_type = {'F16': '<f2', 'F32': '<f4'}[info['dtype']]
        values = np.memmap(path, mode='r', dtype=data_type, shape=tuple(info['shape']))
        tensor = torch.from_numpy(np.array(values))
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Non-finite tensor: {key}')
        component, local = key.split('.', 1)
        model = models[component]
        if local not in model.state_dict():
            raise ValueError(f'Sample has no model slot: {key}')
        set_module_tensor_to_device(model, local, 'cuda', value=tensor, dtype=tensor.dtype)
        on_device = model.state_dict()[local]
        if on_device.device.type != 'cuda' or not torch.equal(on_device.cpu(), tensor):
            raise ValueError(f'CUDA round trip changed {key}')
        set_module_tensor_to_device(model, local, 'meta')
        if not model.state_dict()[local].is_meta:
            raise ValueError(f'Model slot failed to unload: {key}')
        samples[label] = {'key': key, 'shape': info['shape'], 'dtype': info['dtype'],
                          'sha256': expected, 'bytes': tensor.nbytes}
    return samples


def run(source, author, out):
    raw = (source / 'header.bin').read_bytes()
    if sha256(raw).hexdigest() != HEADER:
        raise ValueError('Unexpected checkpoint header')
    limit = 8 + int.from_bytes(raw[:8], 'little')
    index = json.loads(raw[8:limit])
    models = create_models(source, author)
    components = {name: compare(model, name, index) for name, model in models.items()}
    matches = all(not any(row[key] for key in ('missing', 'unexpected', 'wrong_shape'))
                  for row in components.values())
    cuda = verify_cuda(source, index, models)
    result = {'header_sha256': HEADER, 'diffusers_sha256': DIFFUSERS, 'author_blobs': AUTHOR_FILES,
              'components': components, 'shape_match': matches, 'cuda_samples': cuda}
    with out.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for name in ('source', 'author', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
