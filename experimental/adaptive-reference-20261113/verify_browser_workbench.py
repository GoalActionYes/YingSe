"""Own an isolated server and a fresh headless browser; never attach to user sessions."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

import numpy as np
from PIL import Image

from independent_coco import ROOT, digest
from spatial_workbench import make_spatial_workbench
from reference_color_app.media import ffmpeg_path, probe_media


def run_owned_browser(command, timeout=240):
    """Contain Node and its descendants. Node waits for stdin before launching Edge."""
    if os.name != 'nt':
        raise RuntimeError('This verifier currently requires Windows process-tree containment')
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                    ('flags', wintypes.DWORD), ('min_working_set', ctypes.c_size_t),
                    ('max_working_set', ctypes.c_size_t), ('active_processes', wintypes.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                    ('scheduling', wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [('basic', BasicLimits), ('io', IoCounters),
                    ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                    ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    process = None
    try:
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding='utf-8', creationflags=subprocess.CREATE_NO_WINDOW)
        # The test runner cannot launch a browser until containment succeeds.
        if not kernel.AssignProcessToJobObject(job, wintypes.HANDLE(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            stdout, stderr = process.communicate('go\n', timeout=timeout)
        except subprocess.TimeoutExpired as error:
            kernel.CloseHandle(job)
            job = None
            stdout, stderr = process.communicate(timeout=10)
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from error
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        if job:
            kernel.CloseHandle(job)
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--fixtures', required=True, type=Path)
    parser.add_argument('--browser-executable', required=True, type=Path)
    parser.add_argument('--playwright-module', required=True, type=Path)
    parser.add_argument('--require-aligned', action='store_true')
    args = parser.parse_args()
    if not args.run or Path(args.run).name != args.run or args.run in ('.', '..'):
        raise ValueError('Run must be a single directory name')
    fixtures = args.fixtures.resolve()
    for name in ('target.png', 'reference.png', 'input-audio.mp4'):
        if not (fixtures / name).is_file():
            raise FileNotFoundError(fixtures / name)
    for path in (args.browser_executable, args.playwright_module):
        if not path.is_file():
            raise FileNotFoundError(path)
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Existing Node executable required; this verifier installs nothing')
    out = ROOT / 'outputs/adaptive-reference-20261113/spatial-workbench' / args.run
    out.mkdir(parents=True, exist_ok=False)
    server = make_spatial_workbench(out / 'data', enhanced=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    report = dict(checkpoint_sha256=server.store.engine.checkpoint_sha,
                  fixtures={name: digest(fixtures / name) for name in
                            ('target.png', 'reference.png', 'input-audio.mp4')})
    try:
        command = [
            node, str(Path(__file__).with_suffix('.cjs')),
            f'http://127.0.0.1:{server.server_port}', str(fixtures), str(out),
            str(args.browser_executable.resolve()), str(args.playwright_module.resolve()),
            'aligned' if args.require_aligned else 'ordinary',
        ]
        try:
            result = run_owned_browser(command)
        except subprocess.TimeoutExpired as error:
            (out / 'browser.stdout.log').write_text(error.output or '', encoding='utf-8')
            (out / 'browser.stderr.log').write_text(error.stderr or '', encoding='utf-8')
            report['failure'] = 'Browser timeout; owned process job closed'
            raise
        (out / 'browser.stdout.log').write_text(result.stdout, encoding='utf-8')
        (out / 'browser.stderr.log').write_text(result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Browser verification failed; inspect {out}/browser.stderr.log')
        report['browser'] = json.loads((out / 'browser-results.json').read_text(encoding='utf-8'))
        png_settings = report['browser']['png_export']['settings']
        if args.require_aligned:
            for kind in ('png_export','mp4_export'):
                mapping=server.store.get_mapping_record(report['browser'][kind]['settings']['mapping_id'])
                assert mapping['diagnostics']['certified_aligned']['accepted'], kind
            report['certified_aligned_photo_and_video']=True
        source, _ = server.store.decode(server.store.get_media(png_settings['media_id']))
        expected = server.store.render(png_settings['mapping_id'], source,
                                       png_settings['strength'], png_settings['adjustments'])
        with Image.open(out / 'browser-export.png') as image:
            actual = np.array(image.convert('RGB'))
        assert actual.shape == (1080, 1920, 3) and np.array_equal(actual, expected)
        report['browser_png_verified'] = dict(dimensions=[1920, 1080], exact_renderer_match=True,
                                             sha256=digest(out / 'browser-export.png'))
        video_path = out / 'browser-export.mp4'
        metadata = probe_media(video_path)
        assert metadata['width'] == 640 and metadata['height'] == 360
        assert metadata['has_audio'] and abs(metadata['duration_seconds'] - 3) < .1
        decoded_video = subprocess.run([str(ffmpeg_path()), '-v', 'error', '-i', str(video_path),
                                        '-an', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                                       capture_output=True, timeout=30,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert decoded_video.returncode == 0 and len(decoded_video.stdout) == 90 * 640 * 360 * 3
        report['browser_mp4_verified'] = dict(dimensions=[640, 360], frames=90, has_audio=True,
                                             duration_seconds=metadata['duration_seconds'],
                                             sha256=digest(video_path))
        report['passed'] = report['browser']['passed']
        report['scope'] = 'Fresh headless browser with isolated app data; not visible desktop or subjective color acceptance'
        report['verifier_sha256'] = {p.name: digest(p) for p in
                                    (Path(__file__), Path(__file__).with_suffix('.cjs'))}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
