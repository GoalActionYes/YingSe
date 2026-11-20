"""Verify the pinned AceTone code and two-image token interface without weights."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch
from transformers import Qwen2_5_VLProcessor


WEIGHTS_BYTES = 8131575808


def load_json(path):
    return json.loads(path.read_bytes())


def parameter_map(model, index):
    mapping = model._checkpoint_conversion_mapping
    converted = {}
    for key in sorted(index):
        name = key
        for pattern, replacement in mapping.items():
            name, hits = re.subn(pattern, replacement, name)
            if hits:
                break
        if name in converted:
            raise ValueError(f"Duplicate parameter after conversion: {name}")
        converted[name] = key
    actual = set(model.state_dict())
    if set(converted) != actual:
        raise ValueError(f"Checkpoint names differ: {sorted(set(converted) - actual)} / "
                         f"{sorted(actual - set(converted))}")
    return converted


def tiny_config(config_class, raw, tokenizer):
    text = {key: raw[key] for key in ("bos_token_id", "eos_token_id", "max_position_embeddings", "vocab_size")}
    text.update(hidden_size=48, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1,
                num_hidden_layers=2, rope_scaling={"rope_type": "default", "mrope_section": [4, 4, 4]})
    vision = raw["vision_config"] | {"depth": 2, "fullatt_block_indexes": [1], "hidden_size": 48,
                                     "intermediate_size": 96, "num_heads": 4, "out_hidden_size": 48}
    tokens = {key: raw[key] for key in ("image_token_id", "video_token_id", "vision_start_token_id",
                                        "vision_end_token_id")}
    return config_class(text_config=text, vision_config=vision, mm_vocab_size=256,
                        tie_word_embeddings=True, pad_token_id=tokenizer.pad_token_id,
                        attn_implementation="sdpa", **tokens)


def parse_codes(token_ids, first, last, begin, end):
    if token_ids.count(begin) != 1 or token_ids.count(end) != 1:
        raise ValueError("Expected one pair of table delimiters")
    left, right = token_ids.index(begin), token_ids.index(end)
    body = token_ids[left + 1:right]
    if len(body) != 64 or any(code < first or code > last for code in body):
        raise ValueError("Expected exactly 64 valid table codes")
    return [code - first for code in body]


def tokenizer_check(tokenizer, model):
    first, last = model.vq_range
    identifiers = [tokenizer.convert_tokens_to_ids(f"<MM{i}>") for i in range(256)]
    if identifiers != list(range(first, last + 1)):
        raise ValueError("Tokenizer code IDs do not match published model")
    begin, end = [tokenizer.convert_tokens_to_ids(token) for token in ("<SoT>", "<EoT>")]
    raw = [begin, *identifiers[:63], identifiers[-1], end]
    decoded = parse_codes(raw, first, last, begin, end)
    assert decoded == list(range(63)) + [255]
    for invalid in (raw[:-1], raw[:-2] + [end], [begin, *raw], raw[:12] + [0, *raw[13:]]):
        try:
            parse_codes(invalid, first, last, begin, end)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid token sequence accepted")
    return {"start": begin, "end": end, "code_first": first, "code_last": last,
            "count": len(identifiers), "invalid_controls_rejected": 4}


def batch(processor, pictures):
    message = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                    {"type": "text", "text": "Generate a color table from the two pictures."}]}]
    prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], images=pictures, padding=True, return_tensors="pt")


@torch.inference_mode()
def run(assets):
    import transformers

    source = assets / "upstream"
    weights = assets / "checkpoint"
    paths = sorted([*source.glob("model/*.py"), *weights.glob("*.json")])
    identities = {path.relative_to(assets).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in paths}
    expected_source = {
        "upstream/model/config_acetone.py": "cc9ae9c69c4a926b15d20ef2a0c706f39ac02fef2dbe4888792f8d16088f794c",
        "upstream/model/modeling_acetone.py": "fb9432327c283b84399dbba33f177acfd80629bbbfc6a2da5c85de24bbb5ec90",
    }
    if any(identities.get(key) != value for key, value in expected_source.items()):
        raise ValueError("Unexpected upstream Python source")
    sys.path.insert(0, str(source.resolve()))
    from model.config_acetone import AceToneConfig
    from model.modeling_acetone import AceToneVLM

    raw = load_json(weights / "config.json")
    index = load_json(weights / "model.safetensors.index.json")
    assert index["metadata"]["total_size"] == WEIGHTS_BYTES
    with torch.device("meta"):
        full = AceToneVLM(AceToneConfig(**raw))
    mapped = parameter_map(full, index["weight_map"])
    assert full.model.language_model.embed_tokens.weight.shape == full.lm_head.weight.shape
    assert tuple(full.lm_head.weight.shape) == (151936, 2048)

    processor = Qwen2_5_VLProcessor.from_pretrained(weights, local_files_only=True, use_fast=True)
    torch.set_num_threads(2)
    torch.manual_seed(351967)
    tiny = AceToneVLM(tiny_config(AceToneConfig, raw, processor.tokenizer)).eval()
    tiny.tie_weights()
    tokens = tokenizer_check(processor.tokenizer, tiny)
    first = Image.fromarray(np.tile(np.array([23, 128, 239], dtype=np.uint8), (56, 56, 1)))
    second = Image.fromarray(np.tile(np.array([192, 98, 46], dtype=np.uint8), (56, 56, 1)))
    normal, swapped = [batch(processor, pictures) for pictures in ([first, second], [second, first])]
    torch.testing.assert_close(normal["input_ids"], swapped["input_ids"], atol=0, rtol=0)
    assert normal["image_grid_thw"].shape == (2, 3)
    expected_slots = int(normal["image_grid_thw"].prod(dim=1).sum()) // raw["vision_config"]["spatial_merge_size"] ** 2
    assert int((normal["input_ids"] == raw["image_token_id"]).sum()) == expected_slots
    logits = tiny(**normal).logits[:, -1]
    reverse = tiny(**swapped).logits[:, -1]
    delta = float((logits - reverse).abs().max())
    assert torch.isfinite(logits).all() and delta > 1e-6
    generated = tiny.generate(**normal, max_new_tokens=3, min_new_tokens=3, do_sample=False, use_cache=True)
    appended = generated[0, normal["input_ids"].size(1):].tolist()
    assert len(appended) == 3 and all(0 <= item < raw["vocab_size"] for item in appended)
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "attention": tiny.config._attn_implementation, "weights_bytes": WEIGHTS_BYTES,
            "parameter_names_verified": len(mapped), "head_shape": list(full.lm_head.weight.shape),
            "assets": identities, "tokenizer": tokens, "image_grid": normal["image_grid_thw"].tolist(),
            "image_token_slots": expected_slots, "input_swap_logit_delta": delta,
            "random_generated_ids": appended}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    data = run(args.assets)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
