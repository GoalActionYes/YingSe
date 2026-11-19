"""Validate embedded RGB ICC references on the ordinary video HTTP workflow."""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import threading
import time
from urllib.request import Request, urlopen

from PIL import Image, ImageCms

from reference_color_app.media import ffmpeg_path
from reference_color_app.server import make_server


def sha(data):
    return hashlib.sha256(data).hexdigest()


def png(image):
    memory = BytesIO()
    image.save(memory, 'PNG')
    return memory.getvalue()


def references(path):
    with Image.open(path) as reference:
        profile = reference.info.get('icc_profile')
        if reference.mode != 'RGB' or not profile:
            raise ValueError('Expected a tagged RGB reference')
        source = ImageCms.ImageCmsProfile(BytesIO(profile))
        managed = ImageCms.profileToProfile(reference, source, ImageCms.createProfile('sRGB'),
                     renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode='RGB')
        # Rebuild from RGB bytes to ensure the two controls contain no source tag.
        return {'tagged': path.read_bytes(),
                'srgb': png(Image.frombytes('RGB', managed.size, managed.tobytes())),
                'untagged': png(Image.frombytes('RGB', reference.size, reference.tobytes()))}


def request(url, route, data=None, filename=None):
    headers = {}
    if isinstance(data, dict):
        data = json.dumps(data).encode()
        headers['Content-Type'] = 'application/json'
    if filename is not None:
        headers = {'Content-Type': 'application/octet-stream', 'X-Filename': filename}
    with urlopen(Request(url + route, data=data, headers=headers), timeout=90) as response:
        content = response.read()
        return json.loads(content) if response.headers.get_content_type() == 'application/json' else content


def export_arm(url, target, raw, label, out, analyze_seconds):
    reference = request(url, '/api/media', raw, label + '.png')
    start = time.perf_counter()
    mapping = request(url, '/api/analyze', dict(target_id=target['id'], reference_id=reference['id'],
                                               time_seconds=analyze_seconds))
    analysis = time.perf_counter() - start
    adjustments = {'basic': {'brightness': 12}}
    params = dict(mapping_id=mapping['mapping_id'], strength=.8, adjustments=adjustments)
    start = time.perf_counter()
    preview = request(url, '/api/preview', params)
    elapsed_preview = time.perf_counter() - start
    start = time.perf_counter()
    job = request(url, '/api/export', dict(params, media_id=target['id'], kind='mp4'))
    while True:
        state = request(url, '/api/jobs/' + job['id'])
        if state['status'] == 'done':
            break
        if state['status'] not in ('queued', 'running') or time.perf_counter() - start > 180:
            raise RuntimeError(state.get('error') or state['status'])
        time.sleep(.05)
    export = time.perf_counter() - start
    path = out / f'{label}.mp4'
    path.write_bytes(request(url, state['result_url']))
    return dict(preview_sha256=sha(base64.b64decode(preview['after'].split(',',1)[1])),
                analysis_seconds=analysis, preview_seconds=elapsed_preview, export_seconds=export)


def frame_hashes(path):
    args = [str(ffmpeg_path()), '-hide_banner', '-loglevel', 'error', '-i', str(path), '-map', '0:v:0',
            '-pix_fmt', 'rgb24', '-fps_mode', 'passthrough', '-f', 'framemd5', 'pipe:1']
    result = subprocess.run(args, check=True, capture_output=True).stdout.decode('ascii')
    timeline = [line.split(',')[:4] for line in result.splitlines() if not line.startswith('#')]
    time_base = next(line for line in result.splitlines() if line.startswith('#tb '))
    return result, (time_base, timeline)


def run(video, reference, out, seconds):
    out.mkdir()
    server = make_server(out / 'store')
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        target = request(url, '/api/media', video.read_bytes(), 'target.mp4')
        arms = {label: export_arm(url, target, raw, label, out, seconds)
                for label, raw in references(reference).items()}
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    times = {}
    _, input_timing = frame_hashes(video)
    for label, row in arms.items():
        start = time.perf_counter()
        hashes, times[label] = frame_hashes(out / f'{label}.mp4')
        row.update(post_export_check_seconds=time.perf_counter() - start, frame_count=len(times[label][1]),
                   framemd5_sha256=sha(hashes.encode()))
        (out / f'{label}.framemd5').write_text(hashes, encoding='ascii', newline='\n')
    output = dict(video_sha256=sha(video.read_bytes()), reference_sha256=sha(reference.read_bytes()),
                  analyze_seconds=seconds, results=arms,
                  matching_video_frames=arms['tagged']['framemd5_sha256'] == arms['srgb']['framemd5_sha256'],
                  matching_preview=arms['tagged']['preview_sha256'] == arms['srgb']['preview_sha256'],
                  untagged_differs=arms['tagged']['framemd5_sha256'] != arms['untagged']['framemd5_sha256'],
                  same_timing=input_timing == times['srgb'] == times['tagged'] == times['untagged'])
    (out / 'results.json').write_text(json.dumps(output, indent=2) + '\n',encoding='utf-8')
    if not all(output[key] for key in ('matching_video_frames', 'matching_preview', 'untagged_differs', 'same_timing')):
        raise SystemExit('Video ICC parity check did not pass')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--time', type=float, default=3.)
    args = parser.parse_args()
    run(args.video, args.reference, args.out, args.time)
