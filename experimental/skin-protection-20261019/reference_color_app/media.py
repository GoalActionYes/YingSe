"""Full display-resolution SDR inputs using Pillow and packaged FFmpeg.

8/10-bit SDR video is decoded to RGB8 before feature analysis or mapping.
10-bit import is a compatibility conversion, not native decode equivalence.

No profile-managed/native decode equivalence is claimed. FFmpeg frame PTS is
reported in the demuxer's timeline; it is not a promise of native frame selection.
"""
from io import BytesIO
from functools import lru_cache
import math
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
from PIL import Image, ImageOps


class UnsupportedImageError(ValueError):
    """Input pixel precision is outside the supported ordinary 8-bit domain."""


def _validate_image_mode(im):
    # Pillow's public RGB/RGBA mode describes the *decoded* buffer: its PNG
    # decoder can silently reduce RGB;16B/RGBA;16B into that 8-bit buffer.
    # Inspect the still-unloaded decoder plan before EXIF access or conversion.
    raw_modes=[]
    for tile in getattr(im,'tile',()):
        args=tile.args if hasattr(tile,'args') else tile[3]
        if isinstance(args,str):raw_modes.append(args)
        elif isinstance(args,tuple):raw_modes.extend(x for x in args if isinstance(x,str))
    tags=getattr(im,'tag_v2',{})
    bits=tags.get(258,())  # TIFF BitsPerSample, including multichannel images.
    bits=(bits,) if isinstance(bits,int) else bits
    sample_formats=tags.get(339,())  # TIFF SampleFormat=3 is IEEE floating point.
    sample_formats=(sample_formats,) if isinstance(sample_formats,int) else sample_formats
    if (im.mode in ('I','F') or im.mode.startswith('I;16') or
        any(re.search(r';(?:16|32|64)',mode) for mode in raw_modes) or
        any(int(value)>8 for value in bits) or 3 in sample_formats):
        raise UnsupportedImageError('High-bit-depth/float images are unsupported; use an 8-bit SDR image')


def ffmpeg_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    for base in (root/'.deps/imageio_ffmpeg/binaries',
                 root/'dist/build-20260927-194527/YingSe-0.2.2-Windows-x64/.deps/imageio_ffmpeg/binaries'):
        try:
            for candidate in sorted(base.glob('*.exe')):
                if 'ffmpeg' in candidate.name.lower() and candidate.is_file():
                    return candidate
        except OSError:
            continue
    raise FileNotFoundError('Packaged FFmpeg executable is unavailable')


def _run(args):
    # stderr goes to disk, never an unbounded in-memory capture. Preserve both
    # header metadata and final errors/showinfo within a bounded diagnostic read.
    with tempfile.TemporaryFile() as log:
        try:
            result = subprocess.run([str(ffmpeg_path()),'-hide_banner','-nostdin',*map(str,args)],
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=log,
                shell=False,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=120)
        except (OSError,subprocess.TimeoutExpired) as exc:
            raise ValueError('FFmpeg media processing failed or timed out') from exc
        size=log.tell();log.seek(0);head=log.read(131072)
        if size>131072:
            log.seek(max(131072,size-131072));head+=log.read(131072)
    return result.returncode,result.stdout,head.decode('utf-8',errors='replace')


@lru_cache(maxsize=4)
def _pixel_depths(binary, size, modified_ns):
    # Query this decoder's descriptors, not total packed bits per pixel or a
    # guessed format suffix. Unknown/hardware formats have no proven precision.
    code, output, _ = _run(['-pix_fmts'])
    if code:return {}
    result={}
    for line in output.decode('ascii',errors='replace').splitlines():
        match=re.fullmatch(r'\s*[IOHPB.]{5}\s+(\w+)\s+(\d+)\s+\d+\s+(\d+(?:-\d+)*)\s*',line)
        if match:
            depths=tuple(map(int,match[3].split('-')))
            if len(depths)==int(match[2]):result[match[1]]=depths
    return result


def _video_precision(stream):
    binary=ffmpeg_path();stat=binary.stat()
    formats=_pixel_depths(str(binary),stat.st_size,stat.st_mtime_ns)
    match=re.search(r'Video:\s*[^,]+,\s*(\w+)(?=[(,\s])',stream)
    name=match[1] if match else None
    return name,list(formats[name]) if name in formats else None


def decode_image(path) -> np.ndarray:
    try:
        with Image.open(path) as im:
            _validate_image_mode(im)
            return np.array(ImageOps.exif_transpose(im).convert('RGB'),dtype=np.uint8)
    except UnsupportedImageError:raise
    except (OSError,ValueError,Image.DecompressionBombError) as exc:
        raise ValueError('Invalid or unsupported image input') from exc


def probe_media(path) -> dict:
    path=Path(path)
    if not path.is_file():raise ValueError('Media input does not exist')
    try:
        with Image.open(path) as im:
            _validate_image_mode(im)
            width,height=im.size
            orientation=im.getexif().get(274,1)
            if orientation in (5,6,7,8):width,height=height,width
            # getexif can load PNG state; verify in a fresh decoder instance.
            with Image.open(path) as check:check.verify()
            return dict(kind='image',width=width,height=height,duration_seconds=None,
                        has_audio=False,exif_orientation=orientation,
                        color_management='Pillow RGB conversion; ICC color management unvalidated')
    except UnsupportedImageError:raise
    except (OSError,ValueError,Image.DecompressionBombError):pass
    _,_,log=_run(['-i',path])
    streams=[line.strip() for line in log.splitlines() if re.search(r'Stream #.*Video:',line)]
    duration=re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)',log)
    if not streams or not duration:raise ValueError('Invalid video or unavailable finite duration')
    seconds=int(duration[1])*3600+int(duration[2])*60+float(duration[3])
    dims=re.search(r'(?<!\w)([1-9]\d{0,5})x([1-9]\d{0,5})(?!\w)',streams[0])
    if not dims or seconds<=0:raise ValueError('Invalid video dimensions or duration')
    pixel_format,component_depths=_video_precision(streams[0])
    hdr_unsupported=bool(re.search(r'smpte2084|arib-std-b67|bt2020',streams[0],re.I))
    converts_to_8 = bool(not hdr_unsupported and component_depths and 10 in component_depths
                         and all(depth in (8, 10) for depth in component_depths))
    precision_note=('暂不支持 HDR / BT.2020 视频' if hdr_unsupported else
                    '10位 SDR 视频转为8位处理；导出为8位' if converts_to_8 else
                    '8位 SDR 视频处理' if component_depths and all(depth==8 for depth in component_depths) else
                    '无法确认或暂不支持此视频位深')
    return dict(kind='video',width=int(dims[1]),height=int(dims[2]),
        pixel_format=pixel_format,component_bit_depths=component_depths,
        working_precision_bits=8,precision_conversion=converts_to_8,
        precision_note=precision_note,
        duration_seconds=seconds,has_audio=bool(re.search(r'Stream #.*Audio:',log)),
        video_stream=streams[0],hdr_unsupported=hdr_unsupported,
        metadata_text=log,dimension_convention='coded stream dimensions; decoded PNG gives oriented display dimensions',
        color_management='FFmpeg normal SDR RGB decoding; native/profile equivalence unvalidated')


def decode_frame(path,time_seconds):
    if isinstance(time_seconds,(bool,str)):
        raise ValueError('Frame time must be a finite number')
    try:t=float(time_seconds)
    except (ValueError,TypeError,OverflowError) as exc:raise ValueError('Invalid frame time') from exc
    if not math.isfinite(t) or t<0:raise ValueError('Frame time must be finite and nonnegative')
    metadata=probe_media(path)
    if metadata['kind']!='video':raise ValueError('Expected a video input')
    if t>=metadata['duration_seconds']:raise ValueError('Frame time must be less than duration')
    if metadata['hdr_unsupported']:raise ValueError('暂不支持 HDR 或 BT.2020 视频，请先转换为普通 SDR 视频后导入')
    depths=metadata['component_bit_depths']
    if not depths:
        raise ValueError('无法确认视频位深（'+str(metadata.get('pixel_format') or '未知像素格式')+'）；目前支持8位或10位 SDR 视频')
    if any(depth not in (8,10) for depth in depths):
        raise ValueError('暂不支持此视频位深（'+ '/'.join(map(str,depths)) +'位）；目前支持8位或10位 SDR 视频')
    code,png,log=_run(['-ss',format(t,'.17g'),'-copyts','-i',Path(path),'-map','0:v:0',
        '-an','-sn','-dn','-vf','showinfo','-frames:v','1','-f','image2pipe','-c:v','png','-pix_fmt','rgb24','pipe:1'])
    used_tail=False
    if not code and not png:
        # A valid time can fall after the last frame's PTS but before duration.
        # Include keyframe preroll so a long-lived low-FPS final frame is kept.
        code,png,log=_run(['-sseof','-1','-noaccurate_seek','-copyts','-i',Path(path),'-map','0:v:0',
            '-an','-sn','-dn','-vf','showinfo,reverse','-frames:v','1','-f','image2pipe',
            '-c:v','png','-pix_fmt','rgb24','pipe:1'])
        used_tail=True
    if code or not png:raise ValueError('Failed to decode requested video frame: '+log[-1000:])
    try:
        with Image.open(BytesIO(png)) as im:rgb=np.array(im.convert('RGB'),dtype=np.uint8)
    except (OSError,ValueError) as exc:raise ValueError('FFmpeg returned an invalid frame') from exc
    pts=re.findall(r'\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([-+\d.eE]+)',log)
    if not pts:raise ValueError('Decoded frame timestamp unavailable')
    metadata.update(width=rgb.shape[1],height=rgb.shape[0],requested_time_seconds=t,
        actual_frame_pts_seconds=float(pts[-1] if used_tail else pts[0]),
        extraction_convention='FFmpeg accurate input seek with copyts; first frame at/after requested time, or final frame when seek passes final PTS; showinfo PTS in input timeline',
        final_frame_fallback=used_tail)
    return rgb,metadata
