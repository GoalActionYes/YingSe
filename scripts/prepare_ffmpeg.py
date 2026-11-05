"""Prepare the app's local FFmpeg path from the installed Windows wheel."""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skin', action='store_true')
    args = parser.parse_args()
    import imageio_ffmpeg

    root = Path(__file__).resolve().parents[1]
    if args.skin:
        root = root / 'experimental/skin-protection-20261019'
    source = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if not source.is_relative_to(Path(imageio_ffmpeg.__file__).resolve().parent):
        raise SystemExit('Expected FFmpeg from installed imageio-ffmpeg; remove IMAGEIO_FFMPEG_EXE override')
    if source.suffix.lower() != '.exe':
        raise SystemExit('This preparation command targets Windows x64')
    subprocess.run([str(source), '-version'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    destination = root / '.deps/imageio_ffmpeg/binaries' / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
        temporary = destination.with_suffix('.preparing')
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    print('FFmpeg ready:', destination)


if __name__ == '__main__':
    main()
