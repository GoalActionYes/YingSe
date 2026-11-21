"""Render the checkpoint-verified VCG on one frozen Gros portrait comparison."""

from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter
from safetensors import safe_open
import torch
from accelerate.utils import set_module_tensor_to_device

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, finish_spatial, save
from probe_vcg_model_loading import compare, create_models
from probe_vcg_preprocess import FILE_SIZE, upstream_preprocess


SIZE = 512
STEPS = 25
SEED = 48
EXPECTED_SHA = '788ee27da6d2459f867238488055d32ebb79f50c1988c3362efc91c25240221d'
MODEL_FP = torch.float16


def activate(weights, model, component, aliases, dtype):
    prefix = component + '.'
    known = set(model.state_dict())
    for full in weights.keys():
        if full.startswith(prefix):
            old = full.removeprefix(prefix)
            local = aliases.get(old, old)
            if local not in known:
                raise ValueError(f'Unexpected model tensor: {full}')
            value = weights.get_tensor(full)
            set_module_tensor_to_device(model, local, 'cuda', value=value, dtype=dtype)
            known.remove(local)
    if known:
        raise ValueError(f'Missing tensors in {component}: {sorted(known)[:4]}')
    for name, value in tuple(model.named_buffers()):
        if value.is_meta:
            if component == 'clip_vision' and name == 'vision_model.embeddings.position_ids':
                position = torch.arange(50, device='cuda').expand((1, -1))
                set_module_tensor_to_device(model, name, 'cuda', value=position)
            else:
                raise ValueError(f'Unmaterialized buffer: {component}.{name}')
    model.eval()


def unload(model):
    model.to('meta')
    torch.cuda.empty_cache()


def identity_lut():
    values = np.linspace(0., 1., 16, dtype=np.float32)
    b, g, r = np.meshgrid(values, values, values, indexing='ij')
    return np.stack((r, g, b), axis=-1).reshape(64, 64, 3)


@torch.inference_mode()
def rendered(weights, models, aliases, processor, scheduler, hook, preprocess, target, reference, skip):
    reference = np.asarray(Image.fromarray(reference).resize((SIZE, SIZE)))
    native, reduced = preprocess([target], reference, SIZE, skip)
    pair = np.stack((reference, reduced[0]))

    clip = models['clip_vision']
    activate(weights, clip, 'clip_vision', aliases['clip_vision'], torch.float32)
    clip_rgb = processor(images=[Image.fromarray(image) for image in pair], return_tensors='pt').pixel_values
    pooled = clip.vision_model(pixel_values=clip_rgb.to('cuda')).pooler_output
    pooled = pooled.unsqueeze(1).to(MODEL_FP)
    unload(clip)

    vae = models['vae']
    activate(weights, vae, 'vae', aliases['vae'], MODEL_FP)
    vae_in = torch.from_numpy(pair).to(device='cuda', dtype=MODEL_FP).permute(0, 3, 1, 2) / 127.5 - 1
    vae_latent = torch.cat([vae.encode(frame[None]).latent_dist.mean * .18215 for frame in vae_in])
    unload(vae)

    ref = models['referencenet']
    diffusion = models['l_diffuser']
    writer, reader = hook
    activate(weights, ref, 'referencenet', aliases['referencenet'], MODEL_FP)
    ref(vae_latent, torch.zeros((), device='cuda', dtype=torch.long), encoder_hidden_states=pooled,
        return_dict=False)
    reader.update(writer, dtype=MODEL_FP)
    writer.clear()
    unload(ref)

    activate(weights, diffusion, 'l_diffuser', aliases['l_diffuser'], MODEL_FP)
    ident = identity_lut()
    id_tensor = torch.from_numpy(ident).permute(2, 0, 1)[None].to(device='cuda', dtype=MODEL_FP)
    generator = torch.Generator('cuda').manual_seed(SEED)
    latent = torch.randn((1, 3, 64, 64), device='cuda', generator=generator)
    scheduler.set_timesteps(STEPS, device='cuda')
    for timestep in scheduler.timesteps:
        condition = scheduler.scale_model_input(latent, timestep).to(MODEL_FP)
        prediction = diffusion(
            torch.cat((condition, id_tensor), dim=1), timestep,
            encoder_hidden_states=pooled[:1] - pooled[1:], return_dict=False,
        )[0]
        latent = scheduler.step(prediction.float(), timestep, latent).prev_sample
    if not torch.isfinite(latent).all():
        raise ValueError('Diffusion produced a nonfinite LUT')
    reader.clear()
    unload(diffusion)

    lut = np.clip(latent[0].permute(1, 2, 0).cpu().numpy() + ident, 0., 1.)
    transform = ImageFilter.Color3DLUT(16, lut.ravel().tolist())
    grade = np.array(Image.fromarray(native[0]).filter(transform))
    return finish_spatial(target, grade, STRENGTH), lut


def prepare_models(source, author, checkpoint):
    torch.set_num_threads(4)
    if checkpoint.stat().st_size != FILE_SIZE:
        raise ValueError('Incomplete checkpoint')
    if digest_file(checkpoint) != EXPECTED_SHA:
        raise ValueError('Checkpoint hash mismatch')
    models = create_models(source, author)
    from diffusers import DDIMScheduler
    from models.ReferenceNet_attention_diff import ReferenceNetAttention
    from transformers import CLIPImageProcessor

    raw = (source / 'header.bin').read_bytes()
    index = json.loads(raw[8:8 + int.from_bytes(raw[:8], 'little')])
    mappings = {name: compare(model, name, index) for name, model in models.items()}
    if any(row[key] for row in mappings.values() for key in ('missing', 'unexpected', 'wrong_shape')):
        raise ValueError('Checkpoint does not match the pinned graph')
    aliases = {name: row['aliases'] for name, row in mappings.items()}
    hook = (ReferenceNetAttention(models['referencenet'], mode='write', fusion_blocks='full'),
            ReferenceNetAttention(models['l_diffuser'], mode='read', fusion_blocks='full'))
    processor = CLIPImageProcessor()
    scheduler = DDIMScheduler(beta_start=.00085, beta_end=.012, beta_schedule='linear',
                             steps_offset=1, clip_sample=False)
    official = upstream_preprocess(source / 'util.py')
    return models, aliases, processor, scheduler, hook, official


def run(source, author, checkpoint, out):
    models, aliases, processor, scheduler, hook, official = prepare_models(source, author, checkpoint)

    prior = BASE / 'gros-film-b-cross-v2'
    manifest = checked_json(prior / 'results.json', BASELINE_SHA)
    groups = checked_json(Path(__file__).with_name('gros-portrait-light-regions.json'), REGIONS_SHA)
    direction = '01-from-02'
    case = manifest['cases'][direction]
    target = checked_rgb(prior / '01-cold-off.png', manifest['inputs']['01-cold']['hashes']['off'])
    truth = checked_rgb(prior / f'{direction}-expected.png', case['expected_sha'])
    zones = masks(groups['01']['cold'], groups['screen'], target.shape[:2])
    control = np.asarray(Image.fromarray(target).filter(ImageFilter.Color3DLUT(16, identity_lut().ravel().tolist())))
    np.testing.assert_array_equal(control, target)

    out.mkdir()
    result = {'checkpoint_sha256': EXPECTED_SHA, 'seed': SEED, 'steps': STEPS,
              'baseline_13': case['scores']['13'], 'conditions': {}}
    with safe_open(checkpoint, framework='pt', device='cpu') as weights:
        for arm, skip in [('ncc', True), ('default', False)]:
            frames = {}
            frame_hashes = {}
            for light in LIGHTS:
                frames[light] = {}
                for state in ('off', 'on'):
                    name = f'02-{light}'
                    ref = checked_rgb(prior / f'{name}-{state}.png', manifest['inputs'][name]['hashes'][state])
                    start = time.monotonic()
                    torch.cuda.reset_peak_memory_stats()
                    grade, lut = rendered(weights, models, aliases, processor, scheduler, hook, official, target, ref, skip)
                    frames[light][state] = grade
                    stem = f'{arm}-{light}-{state}'
                    lut_file = out / f'{stem}.npy'
                    np.save(lut_file, lut, allow_pickle=False)
                    frame_hashes[stem] = {
                        'lut': digest_file(lut_file), 'image': save(out / f'{stem}.png', grade),
                        'seconds': time.monotonic() - start,
                        'cuda_peak_allocated': torch.cuda.max_memory_allocated(),
                    }
                    print(stem, frame_hashes[stem], flush=True)
            result['conditions'][arm] = {'hashes': frame_hashes, 'scores': score(target, truth, frames, zones)}
            (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest_file(path):
    from hashlib import file_digest
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for name in ('source', 'author', 'checkpoint', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
