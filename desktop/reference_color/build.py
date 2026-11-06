"""Build a fresh allowlisted Windows release from existing cached dependencies."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reference_color_app.media import ffmpeg_path

def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
def ignored(directory,names):
    return set(names)&{'__pycache__','test','tests','idlelib','tkinter','turtledemo','ensurepip','site-packages','direct_url.json','RECORD','REQUESTED','INSTALLER'}
def main():
    stamp=time.strftime('%Y%m%d-%H%M%S')
    package=Path(os.environ.get('YINGSE_RELEASE_ROOT',str(ROOT/'dist')))/('reference-color-'+stamp)
    package.mkdir(parents=True,exist_ok=False)
    runtime=package/'runtime'; runtime.mkdir()
    python=Path(sys.executable).parent
    for name in ('python.exe','pythonw.exe','python3.dll','python312.dll','vcruntime140.dll','vcruntime140_1.dll','LICENSE.txt'):
        copy(python/name,runtime/name)
    for name in ('DLLs','Lib'): shutil.copytree(python/name,runtime/name,ignore=ignored)
    sites=runtime/'Lib/site-packages'; sites.mkdir()
    for pattern in ('numpy','numpy.libs','numpy-*.dist-info','PIL','pillow-*.dist-info'):
        for source in (python/'Lib/site-packages').glob(pattern): shutil.copytree(source,sites/source.name,ignore=ignored)
    (runtime/'python312._pth').write_text('.\nLib\nDLLs\nLib/site-packages\nimport site\n',encoding='ascii')
    appfiles=('__init__.py','contracts.py','features.py','jobs.py','lut.py','matrices.py','media.py','parameters.py','pipeline.py','calibration.py','aligned_reference.py','preview.py','adjustments.py','server.py','storage.py','curve_tables.json','web/index.html','web/app.js','web/adjustments.js','web/styles.css','demo/source.png','demo/reference.png','demo/source-motion.mp4','demo/README.md')
    for name in appfiles: copy(ROOT/'reference_color_app'/name,package/'reference_color_app'/name)
    for name in ('boot.py','Launcher.cs','build.py','verify.py','README.md'): copy(Path(__file__).parent/name,package/'desktop/reference_color'/name)
    copy(Path(__file__).parent/'README.md',package/'README.md')
    copy(ROOT/'desktop/yingse.ico',package/'desktop/reference_color/yingse.ico')
    ff=ffmpeg_path(); target=package/'.deps/imageio_ffmpeg/binaries'/ff.name; copy(ff,target)
    notices=package/'THIRD-PARTY-NOTICES'; notices.mkdir()
    copy(python/'LICENSE.txt',notices/'Python-LICENSE.txt')
    for source in (ROOT/'.deps/imageio_ffmpeg-0.6.0.dist-info').glob('*LICENSE*'): copy(source,notices/('imageio-ffmpeg-'+source.name))
    for flag,name in (('-L','FFmpeg-license.txt'),('-buildconf','FFmpeg-buildconf.txt')):
        result=subprocess.run([str(ff),flag],capture_output=True,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)); (notices/name).write_bytes(result.stdout+result.stderr)
    (notices/'README.txt').write_text('Python 3.12: https://www.python.org/downloads/source/\nNumPy: https://github.com/numpy/numpy\nPillow: https://github.com/python-pillow/Pillow\nNumPy/Pillow version metadata and licenses are preserved in runtime/Lib/site-packages/*.dist-info.\nFFmpeg 7.1 binary from existing imageio-ffmpeg 0.6.0 cache; wrapper is not required at runtime.\nFFmpeg source: https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz\nBuild vendor: https://www.gyan.dev/ffmpeg/builds/\nSee license/buildconf files for binary license and configured external libraries.\n',encoding='utf8')
    compiler=Path(os.environ['WINDIR'])/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    subprocess.run([str(compiler),'/nologo','/target:winexe','/optimize+','/reference:System.Windows.Forms.dll','/win32icon:'+str(package/'desktop/reference_color/yingse.ico'),'/out:'+str(package/'映色参考调色.exe'),str(package/'desktop/reference_color/Launcher.cs')],check=True)
    copy(package/'映色参考调色.exe',package/'ReferenceColor.exe')
    files=[dict(path=p.relative_to(package).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(package.rglob('*')) if p.is_file()]
    (package/'package-manifest.json').write_text(json.dumps(dict(version='1.1.1-smooth-preview',files=files),ensure_ascii=False,indent=2),encoding='utf8')
    archive=package.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as output:
        for p in sorted(package.rglob('*')):
            if p.is_file(): output.write(p,Path(package.name)/p.relative_to(package))
    record=dict(package=str(package),archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size)
    archive.with_suffix('.sha256').write_text(record['sha256']+'  '+archive.name+'\n',encoding='ascii')
    (ROOT/'dist/latest-reference-color.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(record,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
