"""PROBE COPY: regional pixel processor, not shipped protection pipeline."""
"""Bounded, cancellable 30 fps RGB export with per-frame skin segmentation.

This is an experimental spatial layer, independent of the recovered LUT math.
The caller owns atomic publication of output_path after this function returns.
"""
from contextlib import ExitStack
from pathlib import Path
import re
import subprocess
import threading
import time

import numpy as np

from reference_color_app import media
from reference_color_app.lut import apply_mapping, _strength

FPS = 30
STALL_TIMEOUT_SECONDS = 180


class ProtectedVideoCancelled(Exception):
    """The export was cancelled; its intermediate file must not be published."""


def _read_frame(stream, size):
    """Pipes may return short reads; EOF is valid only on a frame boundary."""
    result = bytearray()
    while len(result) < size:
        block = stream.read(size - len(result))
        if not block:
            if result:
                raise ValueError('视频解码返回不完整的 RGB 帧')
            return None
        result.extend(block)
    return result


def _write_frame(stream, rgb):
    remaining = memoryview(np.ascontiguousarray(rgb)).cast('B')
    while remaining:
        written = stream.write(remaining)
        if not written:
            raise ValueError('视频编码管道提前关闭')
        remaining = remaining[written:]


def _log_tail(path, limit=3000):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(max(0, Path(path).stat().st_size - limit))
            return stream.read(limit).decode('utf-8', 'replace')
    except OSError:
        return ''


def _metadata(log, formats):
    """The same descriptor/header rules as media.probe_media, without spawning.

    The format map is obtained through this export's registered FFmpeg process,
    so precision detection cannot create an untracked media helper process.
    """
    streams = [line.strip() for line in log.splitlines() if re.search(r'Stream #.*Video:', line)]
    duration = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', log)
    if not streams or not duration:
        raise ValueError('无法读取有限时长的视频信息')
    seconds = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])
    dims = re.search(r'(?<!\w)([1-9]\d{0,5})x([1-9]\d{0,5})(?!\w)', streams[0])
    pixel = re.search(r'Video:\s*[^,]+,\s*(\w+)(?=[(,\s])', streams[0])
    pixel_format = pixel[1] if pixel else None
    depths = formats.get(pixel_format)
    if not dims or seconds <= 0:
        raise ValueError('视频尺寸或时长无效')
    if re.search(r'smpte2084|arib-std-b67|bt2020', streams[0], re.I):
        raise ValueError('暂不支持 HDR 或 BT.2020 视频，请先转换为普通 SDR 视频后导入')
    if not depths or any(depth not in (8, 10) for depth in depths):
        raise ValueError('无法确认或不支持此视频位深；目前支持8位或10位 SDR 视频')
    converted = 10 in depths
    return dict(kind='video', width=int(dims[1]), height=int(dims[2]),
        duration_seconds=seconds, has_audio=bool(re.search(r'Stream #.*Audio:', log)),
        pixel_format=pixel_format, component_bit_depths=list(depths),
        working_precision_bits=8, precision_conversion=converted,
        precision_note='10位 SDR 视频转为8位处理；导出为8位' if converted else '8位 SDR 视频处理',
        video_stream=streams[0], hdr_unsupported=False,
        color_management='FFmpeg normal SDR RGB decoding; native/profile equivalence unvalidated')


def _preflight(source_path, base, job_dir, launch, stack):
    """All preflight subprocesses are owned by the same cancellation registry."""
    formats_path = job_dir / 'skin-pixel-formats.txt'
    formats_log = stack.enter_context((job_dir / 'skin-pixel-formats.log').open('wb'))
    formats_out = stack.enter_context(formats_path.open('wb'))
    process = launch(base + ['-pix_fmts'], stdin=subprocess.DEVNULL, stdout=formats_out, stderr=formats_log)
    if process.wait(timeout=STALL_TIMEOUT_SECONDS):
        raise ValueError('无法读取视频解码器的像素精度列表')
    formats_out.flush()
    formats = {}
    for line in _log_tail(formats_path, 131072).splitlines():
        match = re.fullmatch(r'\s*[IOHPB.]{5}\s+(\w+)\s+(\d+)\s+\d+\s+(\d+(?:-\d+)*)\s*', line)
        if match:
            depths = tuple(map(int, match[3].split('-')))
            if len(depths) == int(match[2]):
                formats[match[1]] = depths
    preflight_path = job_dir / 'skin-preflight.log'
    preflight_log = stack.enter_context(preflight_path.open('wb'))
    process = launch(base + ['-xerror', '-err_detect', 'explode', '-i', str(source_path),
        '-map', '0:v:0', '-an', '-sn', '-dn', '-vf', 'showinfo,format=rgb24',
        '-frames:v', '1', '-f', 'null', '-'],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=preflight_log)
    if process.wait(timeout=STALL_TIMEOUT_SECONDS):
        raise ValueError('视频首帧预检失败：' + _log_tail(preflight_path))
    preflight_log.flush()
    log = _log_tail(preflight_path, 262144)
    metadata = _metadata(log, formats)
    dimensions = re.findall(r'\bn:\s*\d+\s+pts:.*?\bs:(\d+)x(\d+)', log)
    if not dimensions:
        raise ValueError('无法确认视频旋转后的显示尺寸')
    metadata['width'], metadata['height'] = map(int, dimensions[0])
    metadata['dimension_convention'] = 'FFmpeg auto-rotated first frame showinfo dimensions'
    return metadata, formats


def _stop(process):
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        raise ValueError('视频子进程未能退出')


def export_protected_video(source_path, output_path, mapping, strength, job_dir,
                           is_cancelled, on_progress, on_processes):
    """Write an unpublished MP4, verifying complete decoding before returning.

    on_processes receives all currently owned children, then [] on every exit.
    A watchdog kills children on cancellation even if a pipe read/write blocks.
    At most one decoded and one protected full-resolution RGB frame are queued.
    """
    strength = float(_strength(strength))
    if strength <= 0:
        raise ValueError('零强度应使用普通视频导出路径')
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    job_dir = Path(job_dir).resolve()
    if source_path == output_path:
        raise ValueError('视频导出路径不能覆盖输入文件')
    job_dir.mkdir(parents=True, exist_ok=True)
    if is_cancelled():
        on_processes([])
        raise ProtectedVideoCancelled('已取消视频导出')
    session = None
    children = []
    child_lock = threading.Lock()
    finished = threading.Event()
    state = dict(cancelled=False, timed_out=False, heartbeat=time.monotonic())
    started = time.monotonic()

    def check():
        if state['cancelled'] or is_cancelled():
            raise ProtectedVideoCancelled('已取消视频导出')
        if state['timed_out']:
            raise ValueError('视频处理长时间无响应，已停止导出')

    def register(process):
        with child_lock:
            children.append(process)
            snapshot = list(children)
        on_processes(snapshot)
        check()
        return process

    def watch():
        while not finished.wait(.1):
            if is_cancelled():
                state['cancelled'] = True
            if time.monotonic() - state['heartbeat'] > STALL_TIMEOUT_SECONDS:
                state['timed_out'] = True
            if state['cancelled'] or state['timed_out']:
                with child_lock:
                    snapshot = list(children)
                for process in snapshot:
                    if process.poll() is None:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass

    monitor = threading.Thread(target=watch, name='protected-video-watchdog', daemon=True)
    monitor.start()
    binary = str(media.ffmpeg_path())
    base = [binary, '-hide_banner', '-nostdin', '-nostats', '-y']
    creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    decode_log = job_dir / 'skin-decode.log'
    encode_log = job_dir / 'skin-encode.log'
    verify_log = job_dir / 'skin-verify.log'
    verify_progress = job_dir / 'skin-verify-progress.txt'
    frames = detected_frames = 0
    last_skin = {}
    success = False
    def launch(args, **kwargs):
        check()
        return register(subprocess.Popen(args, bufsize=0, shell=False,
            creationflags=creationflags, **kwargs))
    try:
        on_progress(0.0)
        with ExitStack() as stack:
            metadata, formats = _preflight(source_path, base, job_dir, launch, stack)
            check()
            height, width = metadata['height'], metadata['width']
            duration = float(metadata['duration_seconds'])
            from __main__ import RegionalSession as SkinSession  # Ordinary export never imports it.
            session = SkinSession()
            dlog = stack.enter_context(decode_log.open('wb'))
            elog = stack.enter_context(encode_log.open('wb'))
            # fps samples the input timeline. Output rawvideo has a declared
            # 30 fps clock; no claim of preserving VFR source timestamps.
            decoder = register(subprocess.Popen(base + [
                '-xerror', '-err_detect', 'explode', '-i', str(source_path),
                '-map', '0:v:0', '-an', '-sn', '-dn', '-vf',
                'fps=fps=30:start_time=0:round=near,format=rgb24',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=dlog,
                bufsize=0, shell=False, creationflags=creationflags))
            encoder = register(subprocess.Popen(base + [
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-video_size', f'{width}x{height}',
                '-framerate', str(FPS), '-i', 'pipe:0', '-i', str(source_path),
                '-map', '0:v:0', '-map', '1:a:0?', '-sn', '-dn',
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
                '-c:v', 'libx264', '-crf', '18', '-c:a', 'aac',
                '-colorspace', 'bt709', '-color_primaries', 'bt709',
                '-color_trc', 'bt709', '-color_range', 'tv',
                '-movflags', '+faststart', str(output_path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=elog,
                bufsize=0, shell=False, creationflags=creationflags))
            while True:
                check()
                raw = _read_frame(decoder.stdout, width * height * 3)
                if raw is None:
                    break
                original = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                graded = original  # regional processor applies the two experimental maps
                protected, diagnostics = session.process(original, graded, temporal=True)
                if protected.dtype != np.uint8 or protected.shape != original.shape:
                    raise ValueError('肤色保护返回了无效的视频帧')
                check()
                _write_frame(encoder.stdin, protected)
                frames += 1
                last_skin = diagnostics
                detected_frames += int(bool(diagnostics.get('detected', diagnostics.get('skin_detected', False))))
                state['heartbeat'] = time.monotonic()
                on_progress(min(.97, frames / FPS / duration * .97))
            check()
            decoder.stdout.close()
            encoder.stdin.close()
            decode_code = decoder.wait(timeout=STALL_TIMEOUT_SECONDS)
            encode_code = encoder.wait(timeout=STALL_TIMEOUT_SECONDS)
            check()
            decode_details = _log_tail(decode_log, 65536)
            # Matroska can report premature EOF while returning zero. A
            # decodable prefix must not be presented as a complete export.
            if decode_code or 'File ended prematurely' in decode_details:
                raise ValueError('视频完整解码失败：' + decode_details[-3000:])
            if encode_code:
                raise ValueError('保护视频编码失败：' + _log_tail(encode_log))
            if not frames:
                raise ValueError('视频没有可处理的完整画面')
            # Successful encoding alone is insufficient. Decode every output
            # frame and check the exact count, including the last frame.
            state['heartbeat'] = time.monotonic()
            vlog = stack.enter_context(verify_log.open('wb'))
            verifier = register(subprocess.Popen(base + [
                '-xerror', '-err_detect', 'explode', '-i', str(output_path),
                '-map', '0:v:0', '-an', '-sn', '-dn', '-fps_mode', 'passthrough',
                '-progress', str(verify_progress), '-f', 'null', '-'],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=vlog,
                shell=False, creationflags=creationflags))
            code = verifier.wait(timeout=STALL_TIMEOUT_SECONDS)
            check()
            progress = _log_tail(verify_progress, 65536)
            counts = re.findall(r'^frame=(\d+)\s*$', progress, re.M)
            verified_frames = int(counts[-1]) if counts else 0
            if code or verified_frames != frames or 'progress=end' not in progress:
                raise ValueError('保护视频完整性验证失败：' + _log_tail(verify_log))
            vlog.flush()
            output_metadata = _metadata(_log_tail(verify_log, 262144), formats)
            check()
            on_progress(1.0)
            success = True
            return dict(codec='libx264', crf=18, audio='first optional audio track; AAC; no shortest truncation',
                fps=FPS, frames=frames, verified_frames=verified_frames,
                video_duration_seconds=frames / FPS, input_duration_seconds=duration,
                output_duration_seconds=output_metadata['duration_seconds'],
                output_metadata=output_metadata, source_display_width=width,
                source_display_height=height, width=width + width % 2, height=height + height % 2,
                padding='right/bottom black padding to even dimensions',
                frame_rate_conversion='input timeline sampled to fixed 30fps; VFR timestamps not preserved',
                working_precision_bits=8, input_component_bit_depths=metadata.get('component_bit_depths'),
                precision_conversion=metadata.get('precision_conversion', False),
                color='SDR BT709 limited YUV420P; RGB8 working frames',
                encoding_loss='Lossy H264/AAC; separate from mapping fidelity',
                skin=dict(enabled=True, per_frame_original_rgb=True, temporal=True,
                          detected_frames=detected_frames, last_frame=last_skin),
                elapsed_seconds=time.monotonic() - started)
    except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as exc:
        check()
        raise ValueError('保护视频管道失败：' + str(exc) + '\n' + _log_tail(encode_log)) from exc
    except Exception:
        # Terminating a decoder can also produce a short-frame ValueError.
        # Cancellation takes precedence over the resulting transport error.
        check()
        raise
    finally:
        finished.set()
        cleanup_error = None
        for process in children:
            try:
                _stop(process)
            except Exception as exc:
                cleanup_error = exc
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        monitor.join(timeout=2)
        close = getattr(session, 'close', None)
        try:
            if close:
                close()
        finally:
            on_processes([])
            if not success:
                output_path.unlink(missing_ok=True)
        if cleanup_error:
            raise cleanup_error
