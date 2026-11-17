"""Match all header shapes via Diffusers conversion without allocating model weights."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.tools/ICTone-quantized-audit/vendor'),
                str(ROOT / '.tools/StyleGallery-audit/vendor'), str(ROOT / '.tools/ModFlows-audit/vendor')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    out = ROOT / 'outputs/adaptive-reference-20261113' / args.run
    out.mkdir(exist_ok=False)
    source = Path(args.metadata)
    metadata = json.loads(source.read_text(encoding='utf-8'))
    import torch
    import gguf
    from accelerate import init_empty_weights
    from diffusers import FluxTransformer2DModel
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
    tensors = metadata['parsed']['fill']['tensors']
    checkpoint = {x['name']: torch.empty(tuple(reversed(x['ggml_dims'])), device='meta') for x in tensors}
    converted = convert_flux_transformer_checkpoint_to_diffusers(checkpoint)
    with init_empty_weights():
        model = FluxTransformer2DModel(in_channels=384, out_channels=64, guidance_embeds=True)
    actual = model.state_dict()
    assert actual.keys() == converted.keys(), (actual.keys() - converted.keys(), converted.keys() - actual.keys())
    mismatches = {k: [list(actual[k].shape), list(v.shape)] for k, v in converted.items()
                  if actual[k].shape != v.shape or v.device.type != 'meta'}
    assert not mismatches, mismatches
    lora = metadata['parsed']['lora']['tensors']
    checked = []
    for key, value in lora.items():
        module, side = key.removeprefix('transformer.').split('.lora_')
        base = actual[module + '.weight']
        shape = value['shape']
        assert len(base.shape) == 2 and len(shape) == 2 and value['dtype'] == 'BF16'
        if side == 'A.weight':
            assert shape == [256, base.shape[1]]
            assert lora['transformer.' + module + '.lora_B.weight']['shape'] == [base.shape[0], 256]
        else:
            assert side == 'B.weight' and shape == [base.shape[0], 256]
        checked.append(dict(key=key, adapter_shape=shape, base_shape=list(base.shape)))
    packed_bytes = 0
    for tensor in tensors:
        elements = 1
        for dim in tensor['ggml_dims']:
            elements *= dim
        block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(tensor['ggml_type'])]
        assert elements % block == 0
        packed_bytes += elements // block * size
    adapter_bytes = metadata['parsed']['lora']['dtype_elements']['BF16'] * 2
    result = dict(finished=True, scope='all meta shapes only; no full checkpoint payload or forward verification',
                  metadata_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gguf_tensors=len(tensors), converted_model_tensors=len(actual),
                  checked_lora_tensors=checked, packed_parameter_bytes=packed_bytes,
                  bf16_adapter_bytes=adapter_bytes, base_plus_adapter_bytes=packed_bytes + adapter_bytes,
                  base_config_source='Diffusers 0.36 defaults with in_channels=384, out_channels=64 and guidance_embeds=True; all header shapes checked')
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'checked_lora_tensors'}, indent=2))


if __name__ == '__main__':
    main()
