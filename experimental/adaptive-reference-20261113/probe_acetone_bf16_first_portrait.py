"""Run the released AceTone BF16 weights on the frozen Gros cold/warm pair."""
from argparse import ArgumentParser
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from accelerate import dispatch_model, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import numpy as np
from PIL import Image
from safetensors import safe_open
import torch
from transformers import Qwen2_5_VLProcessor, Qwen2VLImageProcessor

from probe_acetone_architecture_preflight import load_json, parameter_map, parse_codes
from probe_acetone_vq_film_b import PARENT_SHA, load_model as load_codec, sample
from probe_arri_shared_look import quantize
from probe_gros_film_b_cross import LIGHTS, score
from probe_gros_portrait_light_transfer import masks
from probe_target_mirror_chroma import BASE, STRENGTH, checked_rgb, digest, save
from spatial_render import finish_spatial


GPU_BUDGET = 5200 * 1024**2
SHARDS = {
    "model-00001-of-00002.safetensors": (4997750760, "60d937b53d5a26e8f1c5f9639992834ef1f2dbb785fdf530446566a76f2783ec"),
    "model-00002-of-00002.safetensors": (3133917248, "71d87dd119d1ca9b4f7842ce6b6b99edcbe5085a99fb9ab1c77d7135e9ae5104"),
}
PROMPT = ("The first image is an un-touched raw image, and the second is toned with stylish LUTs. "
          "These two images may have the same source or not, and your task is to mimic the toning method. "
          "You are a professional color grader. Please generate the 64-bit LUT in 'Global toning: <SoT>...<EoT>'.")


def checked_inputs(assets):
    preflight = load_json(assets / "preflight-verify.json")
    for name, expected in preflight["assets"].items():
        if digest(assets / name) != expected:
            raise ValueError(f"Configuration changed: {name}")
    checkpoint = assets / "checkpoint"
    for name, (size, expected) in SHARDS.items():
        path = checkpoint / name
        with path.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != size or fingerprint != expected:
            raise ValueError(f"Checkpoint shard changed: {name}")
    return checkpoint


def placement(model):
    sizes = {name: tensor.numel() * 2 for name, tensor in model.state_dict().items()}
    layers = model.model.language_model.layers
    prefixes = [f"model.language_model.layers.{i}" for i in range(len(layers))]
    totals = {prefix: sum(value for name, value in sizes.items() if name.startswith(prefix + "."))
              for prefix in prefixes}
    placement = {
        "model.visual": 0, "model.language_model.embed_tokens": 0, "model.language_model.norm": 0,
        "model.language_model.rotary_emb": 0, "lm_head": 0, **dict.fromkeys(prefixes, 0),
    }
    resident = sum(sizes.values())
    for prefix in reversed(prefixes):
        if resident <= GPU_BUDGET:
            break
        placement[prefix] = "disk"
        resident -= totals[prefix]
    if resident > GPU_BUDGET:
        raise ValueError("Cannot fit persistent GPU allocation under budget")
    return placement, resident


def load_weights(checkpoint, assets):
    sys.path.insert(0, str((assets / "upstream").resolve()))
    from model.config_acetone import AceToneConfig
    from model.modeling_acetone import AceToneVLM

    config = AceToneConfig(**load_json(checkpoint / "config.json"), attn_implementation="sdpa")
    # Both published tensors are loaded independently, preserving either value exactly.
    config.tie_word_embeddings = False
    config.text_config.tie_word_embeddings = False
    with init_empty_weights():
        model = AceToneVLM(config)
    registry = load_json(checkpoint / "model.safetensors.index.json")["weight_map"]
    names = parameter_map(model, registry)
    devices, resident = placement(model)
    offload = {}
    with ExitStack() as stack:
        stores = {name: stack.enter_context(safe_open(checkpoint / name, framework="pt", device="cpu"))
                  for name in SHARDS}
        actual_keys = {name: filename for filename, archive in stores.items() for name in archive.keys()}
        if actual_keys != registry:
            raise ValueError("Checkpoint names differ from published index")
        state = model.state_dict()
        for new, old in names.items():
            filename = registry[old]
            handler = stores[filename]
            metadata = handler.get_slice(old)
            if tuple(metadata.get_shape()) != tuple(state[new].shape) or metadata.get_dtype() != "BF16":
                raise ValueError(f"Unexpected checkpoint tensor: {old}")
            layer = ".".join(new.split(".")[:4])
            if devices.get(layer) == "disk":
                set_module_tensor_to_device(model, new, "meta", dtype=torch.bfloat16)
                offload[new] = {"safetensors_file": str((checkpoint / filename).resolve()),
                                "weight_name": old, "dtype": "bfloat16"}
            else:
                set_module_tensor_to_device(model, new, 0, value=handler.get_tensor(old), dtype=torch.bfloat16)
    model.eval()
    dispatch_model(model, devices, main_device=0, offload_index=offload)
    assert not (set(offload) & {key for key, value in model.state_dict().items() if value.device.type == "cuda"})
    return model, {"mapped_keys": len(names), "disk_tensor_count": len(offload), "persistent_gpu_bytes": resident,
                   "disk_layers": [name for name, device in devices.items() if device == "disk"]}


def load_processor(checkpoint):
    processor = Qwen2_5_VLProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    processor.image_processor = Qwen2VLImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    return processor


def inference(model, processor, codec, target, reference):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                            {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[Image.fromarray(target), Image.fromarray(reference)],
                       padding=True, return_tensors="pt").to("cuda:0")
    before = perf_counter()
    sequence = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True,
                              pad_token_id=processor.tokenizer.pad_token_id)
    torch.cuda.synchronize()
    generated = sequence[0, inputs.input_ids.shape[1]:].tolist()
    first, last = model.vq_range
    begin, end = [processor.tokenizer.convert_tokens_to_ids(token) for token in ("<SoT>", "<EoT>")]
    stats = {"generation_seconds": perf_counter() - before, "generated_ids": generated,
             "text": processor.tokenizer.decode(generated, skip_special_tokens=False),
             "image_grid": inputs.image_grid_thw.tolist()}
    try:
        codes = parse_codes(generated, first, last, begin, end)
    except ValueError as exc:
        return None, stats | {"valid": False, "reason": str(exc)}
    table = codec.decode_indices(torch.tensor(codes).reshape(1, 4, 4, 4))
    array = table[0].permute(1, 2, 3, 0).numpy()[..., ::-1]
    rendered = finish_spatial(target, quantize(sample(array, target)), STRENGTH)
    return rendered, stats | {"valid": True, "codes": codes}


@torch.inference_mode()
def run(assets, codec, out):
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info(0)[0] < 6 * 1024**3:
        raise ValueError("At least 6 GiB free GPU memory is required")
    checkpoint = checked_inputs(assets)
    frozen = BASE / "gros-film-b-cross-v2"
    parent = frozen / "results.json"
    assert digest(parent) == PARENT_SHA
    report = load_json(parent)
    zones = load_json(Path(__file__).with_name("gros-portrait-light-regions.json"))
    source_hashes = report["inputs"]["01-cold"]["hashes"]
    target, styled = [checked_rgb(frozen / f"01-cold-{key}.png", source_hashes[key]) for key in ("off", "on")]
    out.mkdir()
    start = perf_counter()
    model, configuration = load_weights(checkpoint, assets)
    torch.cuda.synchronize()
    configuration["load_seconds"] = perf_counter() - start
    print(json.dumps(configuration), flush=True)
    processor = load_processor(checkpoint)
    codec_model = load_codec(codec)
    predictions = {}
    outcomes = {}
    for light in LIGHTS:
        predictions[light] = {}
        for state in ("off", "on"):
            label = f"{light}-{state}"
            panel = f"02-{light}"
            guide = checked_rgb(frozen / f"{panel}-{state}.png", report["inputs"][panel]["hashes"][state])
            result, stats = inference(model, processor, codec_model, target, guide)
            if result is not None:
                predictions[light][state] = result
                stats["output_sha256"] = save(out / f"{label}.png", result)
            outcomes[label] = stats
            (out / f"{label}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
            print(label, stats["valid"], len(stats["generated_ids"]), flush=True)
    expected = finish_spatial(target, styled, STRENGTH)
    regions = masks(zones["01"]["cold"], zones["screen"], target.shape[:2])
    complete = all(state["valid"] for state in outcomes.values())
    scores = score(target, expected, predictions, regions) if complete else {}
    final = {"setup": configuration, "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
             "peak_reserved_bytes": torch.cuda.max_memory_reserved(0), "cases": outcomes, "scores": scores,
             "baseline": report["cases"]["01-from-02"]["scores"], "passed": complete and all(
                row["passed"] for row in scores.values())}
    (out / "results.json").write_text(json.dumps(final, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("assets", "codec", "out"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
