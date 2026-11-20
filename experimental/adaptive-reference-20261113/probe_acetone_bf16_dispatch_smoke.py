"""Compare AceTone BF16 mixed-disk dispatch with all-GPU generation in a tiny model."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch

from probe_acetone_architecture_preflight import batch, tiny_config
import probe_acetone_bf16_first_portrait as probe


def run(assets, scratch):
    sys.path.insert(0, str((assets / "upstream").resolve()))
    from model.config_acetone import AceToneConfig
    from model.modeling_acetone import AceToneVLM

    torch.manual_seed(351967)
    source = assets / "checkpoint"
    processor = probe.load_processor(source)
    config = tiny_config(AceToneConfig, probe.load_json(source / "config.json"), processor.tokenizer)
    config.tie_word_embeddings = False
    config.text_config.tie_word_embeddings = False
    model = AceToneVLM(config).eval().to(dtype=torch.bfloat16)
    tensors = {key.replace("model.language_model.", "model.").replace("model.visual.", "visual."):
               value.contiguous() for key, value in model.state_dict().items()}
    scratch.mkdir()
    shard = "weights.safetensors"
    save_file(tensors, scratch / shard)
    (scratch / "config.json").write_text(config.to_json_string(), encoding="utf-8")
    registry = {key: shard for key in tensors}
    (scratch / "model.safetensors.index.json").write_text(json.dumps({"weight_map": registry}), encoding="utf-8")
    probe.SHARDS = {shard: None}
    probe.GPU_BUDGET = sum(value.numel() * 2 for value in tensors.values()) - 1024
    loaded, setup = probe.load_weights(scratch, assets)
    assert setup["disk_tensor_count"] > 0
    images = [Image.fromarray(np.full((56, 56, 3), level, dtype=np.uint8)) for level in (32, 221)]
    inputs = batch(processor, images).to("cuda")
    kwargs = {"max_new_tokens": 3, "min_new_tokens": 3, "do_sample": False}
    with torch.inference_mode():
        expected = model.to("cuda").generate(**inputs, **kwargs)
        actual = loaded.generate(**inputs, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    ids = actual[0, inputs.input_ids.shape[1]:].tolist()
    assert len(ids) == 3
    report = setup | {"checks": "mixed and all-GPU generated identical token IDs", "token_ids": ids}
    (scratch / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("scratch", type=Path)
    run(**vars(parser.parse_args()))
