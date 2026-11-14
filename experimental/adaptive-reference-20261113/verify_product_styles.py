"""Fixed controlled product diagnostics; no training, selection or acceptance by average."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from independent_coco import ROOT, digest
from reference_feature_assay import OPERATIONS, transform
from multiref_response import response_metrics
from tone_response import tone_response
from spatial_render import SpatialEngine
from spatial_store import SpatialStore
from aligned_spatial import PreparedAligned
from spatial_jobs import grade_video_frame
from temporal_grid import TemporalGrid
from reference_color_app.pipeline import analyze, algorithm_fingerprint
from reference_color_app.lut import apply_mapping
from reference_color_app.media import ffmpeg_path


def metrics(output, truth, source):
    error = output.astype(float) - truth.astype(float)
    brightness = source.astype(float) @ [.299, .587, .114]
    low = brightness <= np.median(brightness)
    gradient = []
    for axis in (0, 1):
        gradient.append(np.abs(np.diff(output.astype(float), axis=axis) -
                               np.diff(truth.astype(float), axis=axis)).mean())
    return dict(rgb_mae=float(np.abs(error).mean()),
                shadow_half_mae=float(np.abs(error[low]).mean()),
                light_half_mae=float(np.abs(error[~low]).mean()) if (~low).any() else None,
                gradient_mae=float(np.mean(gradient)),
                endpoint_fraction=float(((output == 0) | (output == 255)).mean()),
                reference_response=response_metrics(source, output, source, truth),
                tone_response=tone_response(source, output, source, truth))


def save(rgb, path):
    Image.fromarray(rgb).save(path)
    return dict(path=path.name, sha256=digest(path))


def sheet(rows, directory, filename):
    # Scientific comparison contact sheet; originals stay in their own PNGs.
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    canvas = Image.new('RGB', (4 * 256, len(rows) * 294), '#202225')
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        y = row_index * 294
        for column, name in enumerate(('source', 'truth', 'global', 'adaptive')):
            with Image.open(directory / row['files'][name]['path']) as image:
                canvas.paste(image.resize((256, 256)), (column * 256, y + 38))
            label = row['operation'] if name == 'truth' else name
            if name in row['metrics']:
                label += ' MAE %.2f' % row['metrics'][name]['rgb_mae']
            draw.text((column * 256 + 5, y + 10), label, font=font, fill='white')
    canvas.save(directory / filename)


def encode(frames, path):
    height, width = frames[0].shape[:2]
    result = subprocess.run([str(ffmpeg_path()), '-v', 'error', '-y', '-f', 'rawvideo',
                             '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '30',
                             '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18',
                             '-pix_fmt', 'yuv420p', str(path)],
                            input=np.stack(frames).tobytes(), capture_output=True,
                            timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', 'replace'))
    return dict(path=path.name, sha256=digest(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    if not args.run or Path(args.run).name != args.run or args.run in ('.', '..'):
        raise ValueError('Expected a single directory name')
    torch.set_num_threads(4)
    out = ROOT / 'outputs/adaptive-reference-20261113/product-styles' / args.run
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / 'frozen-source'
    frozen.mkdir()
    for name in ('verify_product_styles.py', 'product-style-acceptance-protocol.md',
                 'reference_feature_assay.py', 'spatial_render.py', 'spatial_store.py','aligned_spatial.py',
                 'spatial_jobs.py', 'temporal_grid.py', 'tone_response.py', 'multiref_response.py'):
        (frozen / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    data = ROOT / 'outputs/adaptive-reference-20261113/tst100k/reserved-multiref-v1'
    manifest = json.loads((data / 'results.json').read_text(encoding='utf-8'))
    selected = manifest['cases'][:6]
    engine = SpatialEngine('multiref-v1')
    store = SpatialStore(out / 'data', engine)
    report = dict(finished=False, training=False, scope='Already-consumed images; same-content synthetic references; public short video; not semantic or human-preference acceptance',
                  checkpoint_sha256=engine.checkpoint_sha, global_algorithm_fingerprint=algorithm_fingerprint(),
                  source_sha256={p.name: digest(p) for p in frozen.iterdir()},
                  selection_manifest_sha256=digest(data / 'results.json'),
                  selected_indices=[row['index'] for row in selected], images=[], videos=[])
    (out / 'frozen-config.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for case in selected:
        original = data / case['outputs']['input']['path']
        if digest(original) != case['outputs']['input']['sha256']:
            raise ValueError('Frozen source changed')
        directory = out / str(case['index'])
        directory.mkdir()
        with Image.open(original) as image:
            source = np.array(image.convert('RGB'))
        target = directory / 'source.png'
        source_file = save(source, target)
        target_record = store.add_media(target)
        rows = []
        for operation in OPERATIONS:
            truth = transform(source, operation)
            ref_path = directory / (operation + '-reference.png')
            reference_file = save(truth, ref_path)
            reference_record = store.add_media(ref_path)
            start = time.perf_counter()
            record = store.analyze(target_record['id'], reference_record['id'])
            adaptive = store.render(record['id'], source, 1)
            adaptive_seconds = time.perf_counter() - start
            start = time.perf_counter()
            mapping = analyze(target, ref_path)
            global_output = apply_mapping(source, mapping, 1)
            global_seconds = time.perf_counter() - start
            row = dict(index=case['index'], operation=operation,
                       seconds=dict(adaptive=adaptive_seconds, global_pipeline=global_seconds),
                       exact_identity=record['diagnostics']['exact_identity'],
                       global_aligned_fit=bool(mapping.diagnostics.get('aligned_reference', {}).get('accepted', False)),
                       files=dict(source=source_file, truth=reference_file,
                                  adaptive=save(adaptive, directory / (operation + '-adaptive.png')),
                                  global_=save(global_output, directory / (operation + '-global.png'))),
                       metrics=dict(adaptive=metrics(adaptive, truth, source), global_=metrics(global_output, truth, source)))
            row['files']['global'] = row['files'].pop('global_')
            row['metrics']['global'] = row['metrics'].pop('global_')
            rows.append(row)
        sheet(rows, directory, 'all-operations.png')
        report['images'].extend(rows)
        print(json.dumps(dict(image=case['index'], completed=len(report['images']))), flush=True)
    # Actual video grading function and temporal settings; one fixed reference per sequence.
    video = ROOT / 'outputs/skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4'
    decoded = subprocess.run([str(ffmpeg_path()), '-v', 'error', '-i', str(video), '-an',
                              '-vf', 'fps=30,scale=640:360', '-f', 'rawvideo', '-pix_fmt',
                              'rgb24', 'pipe:1'], capture_output=True, timeout=30,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    if decoded.returncode or len(decoded.stdout) != 90 * 640 * 360 * 3:
        raise ValueError('Unexpected controlled source video')
    frames = np.frombuffer(decoded.stdout, np.uint8).reshape(90, 360, 640, 3)
    video_dir = out / 'video'
    video_dir.mkdir()
    first = video_dir / 'first.png'
    save(frames[0], first)
    for operation in ('warm', 'cool', 'saturation_low'):
        reference = transform(frames[0], operation)
        ref_path = video_dir / (operation + '-reference.png')
        save(reference, ref_path)
        mapping = analyze(first, ref_path)
        first_record=store.add_media(first);ref_record=store.add_media(ref_path)
        product_record=store.analyze(first_record['id'],ref_record['id'])
        selected_prepared=store.get_prepared(product_record['id'])
        fixed=selected_prepared if isinstance(selected_prepared,PreparedAligned) else None
        temporal = TemporalGrid(relative_cut=True)
        outputs = dict(adaptive=[], global_=[], truth=[])
        rows = []
        for index, frame in enumerate(frames):
            truth = transform(frame, operation)
            start = time.perf_counter()
            adaptive = grade_video_frame(engine, frame, reference, 1, None, temporal,fixed)
            seconds = time.perf_counter() - start
            baseline = apply_mapping(frame, mapping, 1)
            outputs['adaptive'].append(adaptive)
            outputs['global_'].append(baseline)
            outputs['truth'].append(truth)
            rows.append(dict(frame=index, adaptive_seconds=seconds,
                             adaptive_mae=float(np.abs(adaptive.astype(float)-truth).mean()),
                             global_mae=float(np.abs(baseline.astype(float)-truth).mean())))
        summary = {}
        for name in ('adaptive', 'global_'):
            residual = np.stack(outputs[name]).astype(np.float32) - np.stack(outputs['truth']).astype(np.float32)
            errors = np.abs(residual).mean((1, 2, 3))
            steps = np.abs(np.diff(residual, axis=0)).mean((1, 2, 3))
            summary[name.rstrip('_')] = dict(mean_rgb_mae=float(errors.mean()),
                                            temporal_residual_step_mean=float(steps.mean()),
                                            temporal_residual_step_p95=float(np.quantile(steps, .95)))
        files = {name.rstrip('_'): encode(images, video_dir / (operation + '-' + name.rstrip('_') + '.mp4')) for name, images in outputs.items()}
        report['videos'].append(dict(operation=operation, frames=90, fps=30, source_sha256=digest(video),
                                     certified_fixed_mapping=fixed is not None,
                                     reference_sha256=digest(ref_path), temporal=temporal.settings() if fixed is None else None,
                                     rows=rows, summary=summary, files=files))
        print(json.dumps(dict(video=operation, summary=summary)), flush=True)
    report['image_summary'] = {operation: {name: float(np.mean([row['metrics'][name]['rgb_mae'] for row in report['images'] if row['operation']==operation]))
                                          for name in ('adaptive', 'global')} for operation in OPERATIONS}
    report['finished'] = True
    (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['image_summary']), flush=True)


if __name__ == '__main__':
    main()
