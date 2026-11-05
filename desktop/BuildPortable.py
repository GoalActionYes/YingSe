"""Build an allowlisted portable folder. User uploads, outputs and profiles never enter it."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, zipfile

ROOT = Path(__file__).resolve().parent.parent
STAMP = time.strftime('%Y%m%d-%H%M%S')
OUT = ROOT/'dist'/f'build-{STAMP}'
PACKAGE = OUT/'YingSe-0.2.2-Windows-x64'
PACKAGE.mkdir(parents=True,exist_ok=False)
RUNTIME = PACKAGE/'runtime'
RUNTIME.mkdir()
PYTHON = Path(sys.executable).parent

def copy(source, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)

def ignored(directory,names):
    return {n for n in names if n in {'__pycache__','test','tests','idlelib','tkinter','turtledemo','ensurepip','site-packages','direct_url.json','RECORD','REQUESTED','INSTALLER'}}

for name in ['python.exe','pythonw.exe','python3.dll','python312.dll','vcruntime140.dll','vcruntime140_1.dll','LICENSE.txt']:
    copy(PYTHON/name,RUNTIME/name)
shutil.copytree(PYTHON/'DLLs',RUNTIME/'DLLs',ignore=ignored)
shutil.copytree(PYTHON/'Lib',RUNTIME/'Lib',ignore=ignored)
sites=RUNTIME/'Lib'/'site-packages'
sites.mkdir()
for pattern in ['numpy','numpy.libs','numpy-*.dist-info','PIL','pillow-*.dist-info']:
    for src in (PYTHON/'Lib'/'site-packages').glob(pattern):
        shutil.copytree(src,sites/src.name,ignore=ignored)
(RUNTIME/'python312._pth').write_text('.\nLib\nDLLs\nLib/site-packages\nimport site\n',encoding='ascii')
for name in ['imageio_ffmpeg','imageio_ffmpeg-0.6.0.dist-info']:
    shutil.copytree(ROOT/'.deps'/name,PACKAGE/'.deps'/name,ignore=ignored)
for name in ['server.py','grading.py','app.js','index.html','styles.css']:
    copy(ROOT/'prototype'/name,PACKAGE/'prototype'/name)
for name in ['source.png','reference-v2.png','source.mp4']:
    copy(ROOT/'prototype'/'demo'/name,PACKAGE/'prototype'/'demo'/name)
for name in ['portable_boot.py','PortableLauncher.cs','yingse.ico','BuildPortable.py']:
    copy(ROOT/'desktop'/name,PACKAGE/'desktop'/name)
copy(ROOT/'desktop'/'体验版说明.txt',PACKAGE/'先读我.txt')
copy(ROOT/'desktop'/'体验反馈模板.txt',PACKAGE/'体验反馈模板.txt')
compiler=Path(os.environ['WINDIR'])/'Microsoft.NET'/'Framework64'/'v4.0.30319'/'csc.exe'
subprocess.run([str(compiler),'/nologo','/target:winexe','/optimize+','/reference:System.Windows.Forms.dll',
                '/win32icon:'+str(PACKAGE/'desktop'/'yingse.ico'),'/out:'+str(PACKAGE/'映色.exe'),
                str(PACKAGE/'desktop'/'PortableLauncher.cs')],check=True)
# A separate ASCII launcher alias supports a reliable stop command on Chinese and English systems.
copy(PACKAGE/'映色.exe',PACKAGE/'YingSe.exe')
(PACKAGE/'Stop-Background.cmd').write_bytes(b'@echo off\r\n"%~dp0YingSe.exe" --stop\r\n')
notices=PACKAGE/'THIRD-PARTY-NOTICES'
notices.mkdir()
copy(PYTHON/'LICENSE.txt',notices/'Python-LICENSE.txt')
ffmpeg=next((PACKAGE/'.deps'/'imageio_ffmpeg'/'binaries').glob('*.exe'))
license_result=subprocess.run([str(ffmpeg),'-L'],capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
(notices/'FFmpeg-license.txt').write_bytes(license_result.stdout+license_result.stderr)
(notices/'README.txt').write_text('''Bundled dependencies and upstream source locations

Python 3.12.14: https://www.python.org/downloads/source/
NumPy 2.3.5: https://github.com/numpy/numpy/tree/v2.3.5
Pillow 12.3.0: https://github.com/python-pillow/Pillow
imageio-ffmpeg 0.6.0: https://github.com/imageio/imageio-ffmpeg/tree/v0.6.0
FFmpeg 7.1 binary is supplied by the unmodified imageio-ffmpeg Windows wheel.
FFmpeg source: https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz
Binary build vendor/build information: https://www.gyan.dev/ffmpeg/builds/
FFmpeg includes additional libraries; see FFmpeg-license.txt and -buildconf output.

Python license is copied here. NumPy/Pillow license files are preserved inside
runtime/Lib/site-packages/*.dist-info/licenses (or LICENSE files).
imageio-ffmpeg license is in .deps/imageio_ffmpeg-0.6.0.dist-info/LICENSE.
The binary's license is distinct from the Python wrapper's BSD license.
This package includes this project's program source. No user media is included.
''',encoding='utf8')
conf=subprocess.run([str(ffmpeg),'-buildconf'],capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
(notices/'FFmpeg-buildconf.txt').write_bytes(conf.stdout+conf.stderr)
manifest=[]
for file in sorted(PACKAGE.rglob('*')):
    if file.is_file():
        manifest.append({'path':file.relative_to(PACKAGE).as_posix(),'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
(PACKAGE/'package-manifest.json').write_text(json.dumps({'version':'0.2.2-portable','files':manifest},ensure_ascii=False,indent=2),encoding='utf8')
archive=OUT/'YingSe-0.2.2-Windows-x64.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for file in sorted(PACKAGE.rglob('*')):
        if file.is_file(): z.write(file,file.relative_to(OUT))
record={'package':str(PACKAGE),'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'build_time':STAMP}
(ROOT/'dist'/'latest-build.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
