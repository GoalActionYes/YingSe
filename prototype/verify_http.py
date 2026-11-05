"""Run against the local V0.2 server after starting server.py."""
import urllib.request, urllib.error, json, pathlib, time, base64, os
ROOT=pathlib.Path(__file__).resolve().parent
BASE=os.environ.get('GRADE_TEST_URL','http://127.0.0.1:8765')
checks=[]
def post(path,data,ctype='application/json'):
    if isinstance(data,dict):data=json.dumps(data).encode()
    request=urllib.request.Request(BASE+path,data=data,headers={'Content-Type':ctype})
    with urllib.request.urlopen(request,timeout=40) as r:return r.read()
def check(name,ok):
    checks.append({'name':name,'passed':bool(ok)})
    assert ok,name
    print('PASS '+name,flush=True)
def await_job(jid):
    for _ in range(150):
        with urllib.request.urlopen(BASE+'/api/job/'+jid,timeout=10) as r:job=json.load(r)
        if job['status']!='running':return job
        time.sleep(.1)
    raise TimeoutError('video job did not finish')

photo=ROOT/'data'/'36c37fca589a477fa7c06f50d783a549.jpg'
if not photo.exists():photo=ROOT/'demo'/'source.png'
u=json.loads(post('/api/upload?name='+photo.name,photo.read_bytes(),'application/octet-stream'))
b={'palette_version':'cinematic-v2','source':u['id'],'reference':'demo-reference',
   'weights':[0,0,0,1,0,0,0,0,0],'strength':1,'adjustments':{}}
styles=json.loads(post('/api/styles',b))['styles']
check('nine_original_and_style_thumbnails',len(styles)==9 and styles[1]['name']=='黑金')
b['adjustments']={'exposure':.3,'contrast':15,'saturation':-10,'temperature':20}
png=post('/api/image',b);p=json.loads(post('/api/preview',b))
check('preview_equals_image_download',base64.b64decode(p['after'].split(',')[1])==png)
check('cube_export',b'cinematic-v2' in post('/api/cube',b))
rec=json.loads(post('/api/recommend',b))
check('recommendation_factors_original_into_strength',0<=rec['strength']<=1 and (rec['weights'][0]==0 or rec['strength']==0))
old=b.copy();old.pop('palette_version')
try:post('/api/preview',old);valid=False
except urllib.error.HTTPError as e:valid=e.code==400
check('old_palette_rejected',valid)
b['source']='demo-video';j=json.loads(post('/api/export',b));job=await_job(j['id'])
check('video_with_style_and_basic_adjustments',job['status']=='done')
with urllib.request.urlopen(BASE+job['report']) as r:report=json.load(r)
check('video_report_captures_controls',report['adjustments']==b['adjustments'] and report['palette_version']=='cinematic-v2')
with urllib.request.urlopen(BASE+job['cube']) as r:cube=r.read()
# Windows text files use CRLF; compare all content lines, not platform newline bytes.
check('video_cube_equals_preview_cube',cube.splitlines()==post('/api/cube',b).splitlines())
j=json.loads(post('/api/export',b));post('/api/cancel',{'id':j['id']});job=await_job(j['id'])
check('cancel_video',job['status']=='cancelled')
(ROOT/'http-verification.json').write_text(json.dumps({'palette_version':'cinematic-v2','passed':True,'checks':checks},ensure_ascii=False,indent=2),encoding='utf8')
