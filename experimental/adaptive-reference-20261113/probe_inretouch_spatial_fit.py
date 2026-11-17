"""Run the frozen upstream compact spatial network on one registered reference."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from background_intervention import target_regions
from independent_coco import digest
from reference_color_app.media import decode_image
from spatial_render import finish_spatial

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/research/2026-11-18-inretouch-public-pilot-evidence.json'
QUADRATIC = ROOT / 'docs/research/2026-11-18-inretouch-reference-original-evidence.json'
MODEL_SHA256 = '2ce7b908dd8e48fa037ac337c81b4ac234b783fab4a5b4097c37a90b3fa1e44d'
TARGET = 'a0007-IMG_2480'


def input_tensor(rgb, device):
    h, w = rgb.shape[:2]
    vertical, horizontal = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    pixels = torch.from_numpy(rgb.copy()).float().permute(2, 0, 1) / 255
    return torch.cat([torch.stack([vertical, horizontal]), pixels]).to(device)


def run(source, upstream, out):
    assert digest(upstream) == MODEL_SHA256
    spec = importlib.util.spec_from_file_location('inretouch_frozen_network', upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evidence = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    prior = json.loads(QUADRATIC.read_text(encoding='utf-8'))

    def read(entry):
        path = source / entry.removeprefix('example_data/').replace('/', '-')
        assert digest(path) == evidence['inputs'][entry]
        return decode_image(path)

    out.mkdir(exist_ok=False)
    before = read('example_data/Example1_Before.jpg')
    after = read('example_data/Example1_After.jpg')
    target = read(f'example_data/Inference_Data/{TARGET}.jpg')
    assert before.shape == after.shape
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    model = module.CNNDWSplitSiren().to(device)
    optimizers = [torch.optim.AdamW(layer.parameters(), lr=.01, weight_decay=decay)
                  for layer, decay in zip((model.model_p, model.model_s, model.model_m), (.1, .0001, .001))]
    schedules = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=.0001)
                 for optimizer in optimizers]
    x = input_tensor(before, device)
    y = input_tensor(after, device)[2:]
    ncols = before.shape[1] - 11
    windows = (before.shape[0] - 11) * ncols
    offsets = torch.arange(12, device=device)
    started = time.monotonic()
    train_loss = {}
    for step in range(1, 1001):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        idx = (torch.rand(484, device=device) * windows).long().clamp(max=windows-1)
        rows = (idx // ncols)[:, None, None] + offsets[None, :, None]
        cols = (idx % ncols)[:, None, None] + offsets[None, None, :]
        samples = x[:, rows, cols].permute(1, 0, 2, 3).contiguous()
        expected = y[:, rows, cols].permute(1, 0, 2, 3)
        loss = torch.nn.functional.l1_loss(model(samples), expected)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .01)
        for optimizer, schedule in zip(optimizers, schedules):
            optimizer.step()
            schedule.step()
        if step == 1 or step % 100 == 0:
            train_loss[step] = float(loss.detach())
            assert np.isfinite(train_loss[step]) and time.monotonic()-started < 90
            print(step, train_loss[step], flush=True)

    model.eval()
    with torch.inference_mode():
        rgb8 = lambda val: (model(val[None])[0].clamp(0, 1)*255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        recon = rgb8(x)
        full = rgb8(input_tensor(target, device))
    output = finish_spatial(target, full, evidence['strength'])
    path = out / 'target.png'
    Image.fromarray(output).save(path)
    Image.fromarray(recon).save(out/'reference-reconstruction.png')
    prediction_sha = digest(path)
    (out/'prediction.sha256').write_text(prediction_sha, encoding='ascii')
    truth = read(f'example_data/GT/{TARGET}.jpg')
    assert truth.shape == target.shape
    truth80 = finish_spatial(target, truth, evidence['strength']).astype(np.int16)
    regions = target_regions(target, np.zeros(target.shape[:2], bool))
    metrics = {}
    for label in ('whole', 'neutral_proxy'):
        mask = regions[label]
        metrics[label] = dict(candidate=float(np.abs(output[mask]-truth80[mask]).mean()),
                              baseline=evidence['cases'][TARGET][label]['outputs_mae']['Example1'],
                              quadratic=prior['cases'][TARGET][label]['output_errors']['Example1'])
    whole, neutral = metrics['whole'], metrics['neutral_proxy']
    passed = (whole['candidate'] <= min(whole['baseline'], whole['quadratic'])-.25
              and neutral['candidate'] <= min(neutral['baseline'], neutral['quadratic'])+.25)
    protocol = Path(__file__).with_name('inretouch-spatial-fit-protocol.md')
    report = dict(network_sha256=MODEL_SHA256, script_sha256=digest(__file__), protocol_sha256=digest(protocol),
                  baseline_evidence_sha256=digest(EVIDENCE),
                  prior_fit_evidence_sha256=digest(QUADRATIC), output_sha256=prediction_sha,
                  reference_reconstruction_sha256=digest(out/'reference-reconstruction.png'),
                  reference_in_sample_mae=float(np.abs(recon.astype(np.int16)-after).mean()),
                  metrics=metrics, passed=passed, losses=train_loss, elapsed_seconds=time.monotonic()-started)
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.source, args.upstream, args.out)
