"""Focused local acceptance checks; does not restart the user's main server."""
import json, os, pathlib, subprocess, sys, time, urllib.error, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8765'
checks = []

def call(path, body=None, base=BASE):
    req = urllib.request.Request(base+path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

def check(name, passed, **details):
    checks.append(dict(name=name, passed=bool(passed), **details))
    print(name, passed, details, flush=True)
    assert passed, name

body = dict(palette_version='cinematic-v2', source='demo-source', reference='demo-reference',
            weights=[0,0,0,1,0,0,0,0,0], strength=1, adjustments={})
info = call('/api/info')
check('main_server_ready', info['palette_version']=='cinematic-v2' and bool(info['ffmpeg']))
for name, change in [
    ('missing_source', {'source':'missing-acceptance-asset'}),
    ('strength_above_range', {'strength':1.1}),
    ('negative_weight', {'weights':[-1,0,0,1,0,0,0,0,0]}),
    ('wrong_weight_count', {'weights':[1]}),
    ('exposure_above_range', {'adjustments':{'exposure':4}}),
    ('unknown_adjustment', {'adjustments':{'unknown':10}}),
]:
    try:
        call('/api/preview', body | change)
        valid = False
    except urllib.error.HTTPError as error:
        detail = json.load(error)
        valid = error.code==400 and bool(detail.get('error'))
    check(name+'_rejected', valid)

timings=[]
for i in range(6):
    start=time.perf_counter()
    preview=call('/api/preview', body | {'strength':i/5})
    timings.append(round(time.perf_counter()-start,3))
check('repeated_preview_recovers_after_errors', preview['width']==960, seconds=timings)

# Use a separate process/port so an active user's uploads and jobs survive.
env=os.environ.copy(); env['GRADE_PORT']='8766'
proc=subprocess.Popen([sys.executable,str(ROOT/'server.py')], env=env,
                      stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
try:
    start=time.perf_counter()
    while time.perf_counter()-start < 15:
        if proc.poll() is not None:
            raise RuntimeError(proc.stderr.read().decode('utf8','replace'))
        try:
            fresh=call('/api/info',base='http://127.0.0.1:8766')
            break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(.15)
    else:
        raise TimeoutError('isolated cold start failed')
    check('isolated_cold_start', fresh['pid']==proc.pid and fresh['palette_version']=='cinematic-v2',
          seconds=round(time.perf_counter()-start,3))
    preview=call('/api/preview',body,base='http://127.0.0.1:8766')
    check('cold_start_preview',preview['width']==960)
finally:
    proc.terminate()
    proc.wait(timeout=10)

(ROOT/'usability-verification.json').write_text(json.dumps({
    'created_local':time.strftime('%Y-%m-%d %H:%M:%S'),
    'palette_version':'cinematic-v2','all_passed':all(c['passed'] for c in checks),
    'checks':checks,'scope':'Local machine, synthetic preview, isolated Python server cold start. Launcher desktop opening and real long-video performance not validated.'
},ensure_ascii=False,indent=2),encoding='utf8')
