"""Apply the frozen ColorFM-L parameter heads before target spatial pooling."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from probe_colorfm_l import (
    ColorFMProbe, PARENT_SHA, PARENT_REPORT, PSBENCH_SHA, WEIGHT_SHA,
    BASE, PARENT, ROOT, authored, background_controls, checked_json, decision, digest, local_controls,
)


def image_tensor(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).cuda().float().permute(2, 0, 1)[None] / 255


def kernels(generator, descriptors):
    return tuple(layer(descriptors).reshape(len(descriptors), a, b) for layer, a, b in (
        (generator.layer1, 4, 16), (generator.layer2, 16, 16),
        (generator.layer3, 16, 16), (generator.layer4, 16, 3),
    ))


def interpolate_weights(indices, height, width, size):
    nodes = torch.arange(size, device=indices.device)
    y = (((indices // width + .5) / height) * size - .5).clamp(0, size - 1)
    x = (((indices % width + .5) / width) * size - .5).clamp(0, size - 1)
    wy = (1 - (y - nodes[:, None]).abs()).clamp_min(0)
    wx = (1 - (x - nodes[:, None]).abs()).clamp_min(0)
    return (wy[:, None] * wx[None, :]).flatten(0, 1)


class LocalColorFM(ColorFMProbe):
    def flow(self, pixels, time_code, matrices):
        values = torch.cat((pixels, torch.full_like(pixels[..., :1], time_code)), dim=-1)
        for matrix in matrices[:-1]:
            values = self.model.act(torch.bmm(values, matrix))
        return torch.bmm(values, matrices[-1])

    @torch.inference_mode()
    def local_transfer(self, target, reference, pool=False):
        model = self.model
        source, look = image_tensor(target), image_tensor(reference)
        features = [model.encoder.forward_features(F.interpolate(
            frame, size=(256, 256), mode='bilinear', align_corners=False))['x_norm_patchtokens']
            for frame in (source, look)]
        subject, style = features
        subject_shift = model.cross_attn(subject, style)
        style_shift = model.cross_attn(style, subject)
        generator = model.param_gen
        spatial = generator.fuse(torch.cat((subject, subject_shift), dim=-1).transpose(1, 2).reshape(1, 768, 16, 16))
        assert spatial.shape == (1, 384, 4, 4)
        if pool:
            spatial = generator.gap(spatial)
        size = spatial.shape[-1]
        transforms = kernels(generator, spatial.flatten(2).transpose(1, 2)[0])
        reverse = generator(style, style_shift)
        pixels = source[0].permute(1, 2, 0).reshape(-1, 3)
        height, width = target.shape[:2]
        results = []
        for offset in range(0, len(pixels), 32768):
            block = pixels[offset:offset + 32768]
            indices = torch.arange(offset, offset + len(block), device=block.device)
            weight = interpolate_weights(indices, height, width, size)
            change = self.flow(block[None].expand(size * size, -1, -1), 0., transforms)
            mid = block + (change * weight[..., None]).sum(dim=0)
            results.append(mid - self.flow(mid[None], 1., reverse)[0])
        graded = torch.cat(results)
        assert bool(torch.isfinite(graded).all())
        output = ((pixels + (graded - pixels) * .8).clamp(0, 1) * 255).round().byte()
        return output.reshape(target.shape).cpu().numpy()

    def measure(self, target, reference, baseline):
        if np.array_equal(target, reference):
            return target.copy(), dict(reason='same_image')
        return self.local_transfer(target, reference), dict(reason='target_spatial_features')


def run(upstream, out):
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261118)
    docs = ROOT / 'docs/research'
    parent = checked_json(PARENT / 'results.json', PARENT_SHA)
    toggle = checked_json(docs / '2026-11-18-local-edit-toggle-evidence.json', PARENT_REPORT)
    pair = docs / '2026-11-18-psbench-action-pair-evidence.json'
    assert digest(pair) == PSBENCH_SHA and toggle['parent_sha256'] == PARENT_SHA and parent['strength'] == .8
    probe = LocalColorFM(upstream)
    target = np.empty((64, 96, 3), np.uint8)
    target[:, :48] = (66, 82, 109)
    target[:, 48:] = (188, 101, 92)
    target[12:24, 12:24] = target[40:52, 68:80] = (126, 126, 126)
    ref = np.roll(target, 1, axis=-1)
    native = probe.transfer(target, ref)
    pooled = probe.local_transfer(target, ref, pool=True)
    replay_error = int(np.abs(native.astype(np.int16)-pooled).max())
    assert replay_error <= 1
    position = ((17, 17), (45, 73))
    assert np.array_equal(native[position[0]], native[position[1]])
    spatial = probe.local_transfer(target, ref)
    same_color = [spatial[index].tolist() for index in position]
    assert np.array_equal(probe.measure(target, target, 255-target)[0], target)
    out.mkdir()
    background = background_controls(out, parent, probe.measure)
    local = local_controls(probe, out, toggle, {row['pair']: row for row in parent['cases']})
    real = authored(out, BASE / 'psbench-shared-local-v1', pair, probe.measure)
    if time.monotonic() - start >= 300:
        raise TimeoutError('ColorFM-L local run exceeded budget')
    source = Path(__file__)
    report = dict(background=background, background_gate=decision(background), local=local, real=real,
                  replay_max_code_error=replay_error, same_input_rgb_outputs=same_color, same_image=True,
                  weight_sha256=WEIGHT_SHA, parent_sha256=PARENT_SHA, source_sha256=digest(source),
                  protocol_sha256=digest(source.with_name('colorfm-local-pooling-protocol.md')))
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(background=report['background_gate'], real={k: v['gate'] for k,v in real.items()},
                          replay_max_code_error=replay_error, same_input_rgb_outputs=same_color)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()))
