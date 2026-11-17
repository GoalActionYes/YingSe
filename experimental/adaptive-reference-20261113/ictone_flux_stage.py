"""Stream the real frozen Fill/LoRA into synchronous disk groups, not full RAM."""
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / '.tools/ICTone-quantized-audit'
sys.path[:0] = [str(ASSETS / 'vendor'), str(ROOT / '.tools/StyleGallery-audit/vendor'),
                str(ROOT / '.tools/ModFlows-audit/vendor')]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['DIFFUSERS_GGUF_CUDA_KERNELS'] = 'false'


def meta_checkpoint(metadata):
    import torch
    import gguf
    from diffusers.quantizers.gguf.utils import GGUFParameter
    result = {}
    for row in metadata['parsed']['fill']['tensors']:
        shape = tuple(reversed(row['ggml_dims']))
        kind = gguf.GGMLQuantizationType(row['ggml_type'])
        if kind == gguf.GGMLQuantizationType.Q3_K:
            assert len(shape) == 2 and shape[1] % 256 == 0
            result[row['name']] = GGUFParameter(torch.empty((shape[0], shape[1] // 256 * 110),
                                                        dtype=torch.uint8, device='meta'), quant_type=kind)
        else:
            assert kind in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
            result[row['name']] = torch.empty(shape, device='meta',
                    dtype=torch.float32 if kind == gguf.GGMLQuantizationType.F32 else torch.float16)
    return result


def source_group(name):
    if name.startswith('transformer_blocks.'):
        return 'double_blocks.' + name.split('.')[1] + '.'
    if name.startswith('single_transformer_blocks.'):
        return 'single_blocks.' + name.split('.')[1] + '.'
    return 'root'


def build_empty_flux_groups(metadata, config, directory):
    import torch
    from accelerate import init_empty_weights
    from diffusers import FluxTransformer2DModel
    from diffusers.hooks import apply_group_offloading
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
    from diffusers.quantizers.gguf.utils import GGUFParameter, _replace_with_gguf_linear
    from ictone_gguf_runtime import release_disk_group_storage
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assert not list(directory.iterdir()), 'new empty staging directory required'
    converted = convert_flux_transformer_checkpoint_to_diffusers(meta_checkpoint(metadata))
    with init_empty_weights():
        model = FluxTransformer2DModel.from_config(config).eval().to(dtype=torch.bfloat16)
    _replace_with_gguf_linear(model, torch.bfloat16, converted)
    for name, value in converted.items():
        parent, _, leaf = name.rpartition('.')
        model.get_submodule(parent)._parameters[leaf] = (value if isinstance(value, GGUFParameter)
                else torch.nn.Parameter(value.to(torch.bfloat16), requires_grad=False))
    adapter = {name.removeprefix('transformer.'): torch.empty(row['shape'], dtype=torch.bfloat16, device='meta')
               for name, row in metadata['parsed']['lora']['tensors'].items()}
    model.load_lora_adapter(adapter, prefix=None, adapter_name='ictone', low_cpu_mem_usage=True)
    model.requires_grad_(False)
    identities = {}
    for name, old in list(model.named_parameters()):
        identities[name] = dict(shape=list(old.quant_shape if isinstance(old, GGUFParameter) else old.shape),
                                dtype=str(old.dtype), quant_type=int(old.quant_type) if isinstance(old, GGUFParameter) else None)
        parent, _, leaf = name.rpartition('.')
        if isinstance(old, GGUFParameter):
            placeholder = GGUFParameter(torch.empty(0, dtype=torch.uint8), quant_type=old.quant_type)
            placeholder.quant_shape = old.quant_shape
        else:
            placeholder = torch.nn.Parameter(torch.empty(0, dtype=old.dtype), requires_grad=False)
        model.get_submodule(parent)._parameters[leaf] = placeholder
    apply_group_offloading(model, onload_device=torch.device('cuda'), offload_device=torch.device('cpu'),
                           offload_type='block_level', num_blocks_per_group=1, use_stream=False,
                           offload_to_disk_path=str(directory))
    raw_groups = release_disk_group_storage(model)
    names_by_id = {id(p): name for name, p in model.named_parameters()}
    rows = metadata['parsed']['fill']['tensors']
    groups = []
    for group in raw_groups:
        names = {p: names_by_id[id(p)] for p in group.tensor_to_key}
        prefixes = {source_group(name) for name in names.values()}
        assert len(prefixes) == 1, prefixes
        prefix = prefixes.pop()
        inputs = [row['name'] for row in rows if (not row['name'].startswith(('double_blocks.', 'single_blocks.'))
                    if prefix == 'root' else row['name'].startswith(prefix))]
        assert inputs
        groups.append((group, names, inputs))
    return model, groups, identities


def stage_real_flux(directory):
    import numpy as np
    import psutil
    import torch
    import gguf
    from safetensors import safe_open
    from safetensors.torch import save_file
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
    from diffusers.quantizers.gguf.utils import GGUFParameter
    metadata = json.loads((ROOT / 'outputs/adaptive-reference-20261113/ictone-quantized-metadata-v1/results.json').read_text())
    config = json.loads((ASSETS / 'models/components/transformer/config.json').read_text())
    for label in ['fill', 'lora']:
        verified = json.loads((ASSETS / f'assets-{label}-verified.json').read_text())
        assert verified['finished'] and len(verified['records']) == 1
    fill_path = ASSETS / 'models/fill/flux1-fill-dev-Q3_K_S.gguf'
    lora_path = ASSETS / 'models/lora/pytorch_lora_weights.safetensors'
    reader = gguf.GGUFReader(str(fill_path))
    tensors = {row.name: row for row in reader.tensors}
    assert len(tensors) == 780
    started = time.monotonic()
    model, groups, identities = build_empty_flux_groups(metadata, config, Path(directory) / 'groups')
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    records = []
    with fill_path.open('rb') as handle, safe_open(str(lora_path), framework='pt', device='cpu') as lora:
        for group, names, inputs in groups:
            assert time.monotonic()-started <= 600, '600s staging budget exhausted'
            shadow = meta_checkpoint(metadata)
            for name in inputs:
                row = tensors[name]
                handle.seek(row.data_offset)
                body = handle.read(row.n_bytes)
                assert len(body) == row.n_bytes
                value = torch.from_numpy(np.frombuffer(body, dtype=row.data.dtype).reshape(row.data.shape).copy())
                shadow[name] = GGUFParameter(value, quant_type=row.tensor_type) if row.tensor_type == gguf.GGMLQuantizationType.Q3_K else value
            converted = convert_flux_transformer_checkpoint_to_diffusers(shadow)
            payload = {}
            state_rows = []
            for parameter, name in names.items():
                if '.lora_A.' in name or '.lora_B.' in name:
                    original = 'transformer.' + name.replace('.ictone.', '.')
                    value = lora.get_tensor(original)
                else:
                    value = converted[name.replace('.base_layer.', '.')]
                assert not value.is_meta, name
                if isinstance(value, GGUFParameter):
                    assert tuple(value.quant_shape) == tuple(identities[name]['shape'])
                    assert int(value.quant_type) == identities[name]['quant_type']
                    value = value.as_tensor()
                else:
                    assert list(value.shape) == identities[name]['shape'], name
                    value = value.to(torch.bfloat16)
                assert str(value.dtype) == identities[name]['dtype'], (name, value.dtype, identities[name])
                value = value.contiguous()
                key = group.tensor_to_key[parameter]
                payload[key] = value
                state_rows.append(dict(name=name, key=key, stored_shape=list(value.shape),
                                       **identities[name], sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()))
            save_file(payload, group.safetensors_file_path)
            path = Path(group.safetensors_file_path)
            peak_rss = max(peak_rss, process.memory_info().rss)
            records.append(dict(file=path.name, bytes=path.stat().st_size, state=state_rows))
            print(json.dumps(dict(staged_groups=len(records), total=len(groups), bytes=path.stat().st_size)), flush=True)
            del payload, converted, shadow, value, body
            gc.collect()
    result = dict(finished=True, scope='frozen real Fill and LoRA state staging only; no image or GPU inference',
                  groups=records, parameter_count=len(identities),
                  staged_bytes=sum(x['bytes'] for x in records),
                  wall_seconds=time.monotonic()-started, sampled_peak_rss_bytes=peak_rss,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (Path(directory) / 'stage.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return model, result
