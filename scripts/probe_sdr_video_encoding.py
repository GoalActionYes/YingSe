"""Check equivalent SDR encodings through the still and full-video paths."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import subprocess
import threading
from urllib.error import HTTPError

import numpy as np
from PIL import Image, ImageDraw

from probe_reference_icc_video import export_arm, frame_hashes, png, request, sha
from reference_color_app.media import ffmpeg_path
from reference_color_app.server import make_server
from spatial_workbench import make_spatial_workbench

VARIANTS = {
    '709-limited': ('bt709', 'tv', 'bt709'),
    '709-full': ('bt709', 'pc', 'bt709'),
    '601-limited': ('bt601', 'tv', 'smpte170m'),
    'wrong-label': ('bt601', 'tv', 'bt709'),
}
SIZE = (320, 180)


def ffmpeg(*args, raw=None):
    return subprocess.run([str(ffmpeg_path()), '-v', 'error', '-nostdin', *map(str, args)],
                          input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def sample(path):
    with Image.open(path) as image:
        return image.convert('RGB').resize(SIZE, Image.Resampling.BILINEAR)


def mae(a, b):
    return float(np.abs(a.astype(np.int16) - b).mean())


def run_workbench(server, output, videos, original, reference):
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    scores, grades = {}, {}
    try:
        for name, path in videos.items():
            target = request(url, '/api/media', path.read_bytes(), f'{name}.mkv')
            frame = request(url, '/api/frame', {'media_id': target['id'], 'time_seconds': 0})
            with Image.open(BytesIO(base64.b64decode(frame['image'].split(',', 1)[1]))) as still:
                scores[name] = {'decode_vs_original': mae(np.asarray(still), original)}
            export = export_arm(url, target, reference, name, output, 0)
            video = output / f'{name}.mp4'
            raw = ffmpeg('-i', video, '-an', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1')
            grades[name] = np.frombuffer(raw, np.uint8).reshape(-1, SIZE[1], SIZE[0], 3)
            hashes, timeline = frame_hashes(video)
            if len(grades[name]) != 8 or len(timeline[1]) != 8:
                raise ValueError('Unexpected export length')
            scores[name].update(export_frames=8, export_frames_sha256=sha(hashes.encode()),
                                preview_sha256=export['preview_sha256'])
        base = grades['709-limited']
        for name, frames in grades.items():
            scores[name]['grade_vs_709_limited'] = mae(frames, base)
        passes = all(row['decode_vs_original'] <= 2 and row['grade_vs_709_limited'] <= 4
                     for name, row in scores.items() if name != 'wrong-label')
        negative = scores['wrong-label']
        return {'cases': scores, 'passed': passes and negative['decode_vs_original'] > 2
                and negative['grade_vs_709_limited'] > 2}
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def run(source, reference, output, stress_colors=False):
    output.mkdir()
    original = sample(source)
    if stress_colors:
        draw = ImageDraw.Draw(original)
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0))
        for index, color in enumerate(colors):
            for row, box_color in enumerate((color, tuple(channel // 2 for channel in color))):
                start, stop = SIZE[0] * index // 6, SIZE[0] * (index + 1) // 6
                draw.rectangle((start, 60 + row * 60, stop - 1, 119 + row * 60), fill=box_color)
    videos, planes = {}, {}
    for name, (matrix, color_range, label) in VARIANTS.items():
        path = output / f'{name}.mkv'
        samples = ffmpeg('-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '320x180',
               '-i', 'pipe:0', '-vf', f'scale=in_range=pc:out_range={color_range}:out_color_matrix={matrix}',
               '-pix_fmt', 'yuv444p', '-f', 'rawvideo', 'pipe:1', raw=original.tobytes())
        ffmpeg('-f', 'rawvideo', '-pixel_format', 'yuv444p', '-video_size', '320x180', '-framerate', '30',
               '-i', 'pipe:0', '-vf', f'setparams=range={color_range}:colorspace={label}',
               '-c:v', 'ffv1', '-pix_fmt', 'yuv444p', '-colorspace', label, '-color_range', color_range, '-color_primaries', 'bt709',
               '-color_trc', 'bt709', '-fflags', '+bitexact', path, raw=samples * 8)
        retained = ffmpeg('-i', path, '-c:v', 'rawvideo', '-f', 'rawvideo', 'pipe:1')
        if retained != samples * 8:
            raise ValueError('Encoding changed the fixture YUV planes')
        planes[name] = sha(retained)
        videos[name] = path
    if planes['601-limited'] != planes['wrong-label'] or planes['601-limited'] == planes['709-limited']:
        raise ValueError('Wrong-label fixture must change only the decoder label')
    results = {}
    for name, factory in (('stable', make_server), ('spatial', lambda root: make_spatial_workbench(root, enhanced=True))):
        folder = output / name
        folder.mkdir()
        try:
            results[name] = run_workbench(factory(folder / 'store'), folder, videos, np.asarray(original), png(sample(reference)))
        except (HTTPError, RuntimeError) as exc:
            results[name] = {'passed': False, 'error': str(exc)}
    original.save(output / 'working-rgb.png')
    report = {'source_sha256': sha(source.read_bytes()), 'reference_sha256': sha(reference.read_bytes()),
              'working_rgb_sha256': sha(original.tobytes()), 'stress_colors': stress_colors,
              'input_yuv_sha256': planes,
              'input_sha256': {name: sha(path.read_bytes()) for name, path in videos.items()}, 'results': results}
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results))
    if not all(row['passed'] for row in results.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'reference', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--stress-colors', action='store_true')
    run(**vars(parser.parse_args()))
