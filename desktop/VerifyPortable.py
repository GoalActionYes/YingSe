"""Verify a fresh ZIP extraction with the package's own Python, including service shutdown."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, time, urllib.error, urllib.request, zipfile

ROOT=Path(__file__).resolve().parent.parent
build=json.loads((ROOT/'dist/latest-build.json').read_text(encoding='utf8'))
testdir=ROOT/'dist'/('组员 解压验收 '+time.strftime('%H%M%S'))
checks=[]
def check(name,ok,**details):
    checks.append(dict(name=name,passed=bool(ok),**details))
    print(name,ok,flush=True)
    assert ok,name

archive=Path(build['archive'])
with zipfile.ZipFile(archive) as z:
    bad=[n for n in z.namelist() if any(x in n.split('/') for x in ('data','outputs','profile','.codex'))]
    check('archive_excludes_user_media_and_profiles',not bad)
    z.extractall(testdir)
package=testdir/'YingSe-0.2.2-Windows-x64'
manifest=json.loads((package/'package-manifest.json').read_text(encoding='utf8'))
check('all_manifest_checksums_match',all(hashlib.sha256((package/f['path']).read_bytes()).hexdigest()==f['sha256'] for f in manifest['files']))
env=os.environ.copy();env.update(PYTHONHOME='Z:/missing-python',PYTHONPATH='Z:/missing-packages',IMAGEIO_FFMPEG_EXE='Z:/missing-ffmpeg.exe')
started=time.perf_counter()
proc=subprocess.run([str(package/'映色.exe'),'--check'],env=env,timeout=70)
check('isolated_launcher_starts',proc.returncode==0,seconds=round(time.perf_counter()-started,3))
audit=json.loads((package/'desktop/runtime-check.json').read_text(encoding='utf8'))
check('all_runtime_imports_inside_package',all(Path(audit[k]).is_relative_to(package) for k in ('python','numpy','pillow','ffmpeg')) and all(Path(p).is_relative_to(package) for p in audit['sys_path']))
shutil.copy2(ROOT/'prototype/verify_http.py',package/'prototype/verify_http.py')
env['GRADE_TEST_URL']=audit['url']
result=subprocess.run([str(package/'runtime/python.exe'),'-I',str(package/'prototype/verify_http.py')],env=env,timeout=60)
check('nine_http_integration_checks',result.returncode==0)
# Exercise unmodified inherited Windows environment too (including possible Path/PATH aliases).
stage=Path(build['package'])
result=subprocess.run([str(stage/'映色.exe'),'--check'],timeout=70)
check('second_copy_launches_with_inherited_environment',result.returncode==0)
url,token,*_=(stage/'desktop/session.txt').read_text().splitlines()
check('each_copy_uses_distinct_free_port',url!=audit['url'])
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(path,body=None,tok=None):
    data=b'' if body is None else json.dumps(body).encode()
    headers={'Content-Type':'application/json'}
    if tok:headers['X-Yingse-Token']=tok
    return opener.open(urllib.request.Request(url+path,data=data,headers=headers),timeout=10)

try:
    post('/desktop/stop',tok='wrong');ok=False
except urllib.error.HTTPError as error:ok=error.code==403
check('stop_requires_correct_instance_token',ok)
body=dict(palette_version='cinematic-v2',source='demo-video',weights=[0,0,0,1,0,0,0,0,0],strength=1,adjustments={})
with post('/api/export',body) as response: job=json.load(response)
try:
    post('/desktop/stop',tok=token);ok=False
except urllib.error.HTTPError as error:ok=error.code==409
check('stop_refuses_to_interrupt_export',ok)
for _ in range(100):
    with opener.open(url+'/api/job/'+job['id'],timeout=5) as response:status=json.load(response)
    if status['status']!='running':break
    time.sleep(.1)
check('export_finishes_after_refused_stop',status['status']=='done')
with post('/desktop/stop',tok=token) as response:check('stop_succeeds_when_idle',response.status==200)
for _ in range(30):
    if not (stage/'desktop/session.txt').exists():break
    time.sleep(.1)
check('stopped_session_file_removed',not (stage/'desktop/session.txt').exists())
record=dict(package=str(package),archive=str(archive),url=audit['url'],sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),checks=checks,
            scope='Fresh relocation with Chinese/spaces, malicious Python environment variables ignored; tested on this Windows 11 PC, not a separate physical machine')
(ROOT/'dist/latest-test.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
(ROOT/'dist/portable-verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
