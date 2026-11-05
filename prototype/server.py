"""Local-only prototype. Run with Python, numpy, Pillow, imageio-ffmpeg."""
import sys, os, json, uuid, base64, subprocess, threading, time, mimetypes
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'.deps'))
import grading as g

DATA=ROOT/'data'; OUT=ROOT/'outputs'; DEMO=ROOT/'demo'
for p in (DATA, OUT, DEMO): p.mkdir(exist_ok=True)
g.create_demo(DEMO)
g.create_style_reference(DEMO)
UPLOADS={}; JOBS={}; LOCK=threading.Lock()
PORT=int(os.environ.get('GRADE_PORT','8765'))

def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def run_ff(args, **kwargs):
    return subprocess.run([ffmpeg(), '-hide_banner', '-nostdin', *args], capture_output=True,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0), **kwargs)

def asset(key):
    if key=='demo-source': return DEMO/'source.png'
    if key=='demo-reference': return DEMO/'reference-v2.png'
    if key=='demo-video' and (DEMO/'source.mp4').exists(): return DEMO/'source.mp4'
    if key not in UPLOADS: raise ValueError('请重新选择素材，服务重启后上传记录会重置')
    return UPLOADS[key]

def source_image(key):
    p=asset(key)
    if p.suffix.lower() in ('.mp4','.mov','.mkv','.webm','.avi'):
        target=DATA/(p.stem+'_frame.png')
        if not target.exists():
            result=run_ff(['-y','-i',str(p),'-frames:v','1','-vf','scale=960:-2',str(target)],timeout=60)
            if result.returncode: raise ValueError('视频首帧提取失败：'+result.stderr.decode('utf8','replace')[-700:])
        return g.load_image(target)
    return g.load_image(p)

def export_worker(jid, source, weights, strength, adjustments=None):
    job=JOBS[jid]; start=time.perf_counter()
    try:
        work=OUT/jid; work.mkdir()
        lut=work/'grade.cube'; lut.write_text(g.cube_text(g.make_lut(weights,strength,adjustments=adjustments)),encoding='ascii')
        output=work/'graded.mp4'
        args=[ffmpeg(),'-hide_banner','-nostdin','-y','-i',str(source),'-map','0:v:0','-map','0:a:0?',
              '-vf','lut3d=file=grade.cube:interp=trilinear,pad=ceil(iw/2)*2:ceil(ih/2)*2',
              '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-fps_mode','passthrough',
              '-c:a','aac','-b:a','192k','-movflags','+faststart','-progress','pipe:1','-nostats',str(output)]
        with (work/'ffmpeg.log').open('w',encoding='utf8') as log:
            proc=subprocess.Popen(args,cwd=work,stdout=subprocess.PIPE,stderr=log,text=True,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            job['process']=proc
            if job.get('cancelled'): proc.terminate()
            for line in proc.stdout:
                if line.startswith('out_time='): job['processed_time']=line.strip().split('=',1)[1]
            code=proc.wait()
        if job.get('cancelled'):
            job.update(status='cancelled',message='已取消；未完成文件不会作为成果提供。')
        elif code:
            raise RuntimeError((work/'ffmpeg.log').read_text(encoding='utf8')[-1000:])
        else:
            elapsed=round(time.perf_counter()-start,3)
            report={'method':'fixed mixed LUT statistical baseline; no neural model; no skin protection',
                    'source_name':source.name,'weights':g.normalized(weights).tolist(),'strength':strength,
                    'palette_version':g.PALETTE_VERSION,'style_labels':g.LABELS,'adjustments':g.controls(adjustments),
                    'elapsed_seconds':elapsed,'lut_size':33,'audio':'first audio track re-encoded to AAC if present',
                    'scope':'SDR test material; HDR/Log and full color-managed workflows not validated'}
            (work/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
            job.update(status='done',elapsed_seconds=elapsed,url=f'/outputs/{jid}/graded.mp4',
                       report=f'/outputs/{jid}/report.json',cube=f'/outputs/{jid}/grade.cube')
    except Exception as e: job.update(status='error',message=str(e))
    finally: job.pop('process',None)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def send(self,data,status=200,ctype='application/json; charset=utf-8'):
        if isinstance(data,(dict,list)): data=json.dumps(data,ensure_ascii=False).encode('utf8')
        self.send_response(status); self.send_header('Content-Type',ctype)
        self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        path=urlparse(self.path).path
        try:
            if path=='/api/info':
                try: version=run_ff(['-version'],timeout=10).stdout.decode('utf8','replace').splitlines()[0]
                except Exception: version=None
                return self.send({'labels':g.LABELS,'style_ids':g.STYLE_IDS,'descriptions':g.DESCRIPTIONS,
                                  'palette_version':g.PALETTE_VERSION,'pid':os.getpid(),
                                  'ffmpeg':version,'demo_video':(DEMO/'source.mp4').exists()})
            if path.startswith('/api/job/'):
                job=JOBS.get(path.rsplit('/',1)[1])
                if not job: return self.send({'error':'任务不存在'},404)
                return self.send({k:v for k,v in job.items() if k!='process'})
            allowed={'/':ROOT/'index.html','/index.html':ROOT/'index.html',
                     '/app.js':ROOT/'app.js','/styles.css':ROOT/'styles.css'}
            file=allowed.get(path)
            if file is None and (path.startswith('/demo/') or path.startswith('/outputs/')):
                file=(ROOT/path.lstrip('/')).resolve()
                if not (file.is_relative_to(DEMO) or file.is_relative_to(OUT)): raise ValueError('非法路径')
            if file is None or not file.is_file(): return self.send({'error':'文件不存在'},404)
            return self.send(file.read_bytes(),ctype=mimetypes.guess_type(str(file))[0] or 'application/octet-stream')
        except Exception as e: self.send({'error':str(e)},400)
    def do_POST(self):
        try:
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}'):
                return self.send({'error':'仅接受本机页面请求'},403)
            length=int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 300*1024*1024: raise ValueError('文件或请求需在300MB以内')
            path=urlparse(self.path).path
            if path=='/api/upload':
                name=parse_qs(urlparse(self.path).query).get('name',['file'])[0]
                ext=Path(name).suffix.lower()
                if ext not in ('.png','.jpg','.jpeg','.webp','.bmp','.mp4','.mov','.mkv','.webm','.avi'): raise ValueError('不支持该文件格式')
                key=uuid.uuid4().hex; p=DATA/(key+ext)
                with p.open('wb') as f:
                    left=length
                    while left:
                        block=self.rfile.read(min(left,1024*1024))
                        if not block: raise ValueError('上传中断')
                        f.write(block); left-=len(block)
                UPLOADS[key]=p; return self.send({'id':key,'name':Path(name).name})
            if length > 65536: raise ValueError('参数请求过大')
            body=json.loads(self.rfile.read(length))
            if path != '/api/cancel' and body.get('palette_version') != g.PALETTE_VERSION:
                raise ValueError('风格库已升级，请刷新页面；旧版基础预设权重不可直接用于新版风格库。')
            if path=='/api/recommend':
                weights,metrics=g.recommend(source_image(body['source']),source_image(body['reference']))
                weights,strength=g.split_strength(weights)
                return self.send({'weights':weights,'strength':strength,'metrics':metrics})
            if path=='/api/styles':
                from PIL import Image
                im=source_image(body['source'])
                thumbnail=Image.fromarray((im*255).astype('uint8'))
                thumbnail.thumbnail((240,160))
                import numpy as np
                small=np.asarray(thumbnail).astype(np.float32)/255
                cards=[]
                for i, name in enumerate(g.LABELS):
                    w=[0]*9;w[i]=1
                    out=g.apply_lut(small,g.make_lut(w))
                    cards.append({'id':g.STYLE_IDS[i],'name':name,'description':g.DESCRIPTIONS[i],
                                  'image':'data:image/png;base64,'+base64.b64encode(g.png_bytes(out)).decode('ascii')})
                return self.send({'styles':cards})
            if path=='/api/preview':
                im=source_image(body['source']); lut=g.make_lut(body['weights'],body.get('strength',1),adjustments=body.get('adjustments'))
                result=g.apply_lut(im,lut)
                enc=lambda a:'data:image/png;base64,'+base64.b64encode(g.png_bytes(a)).decode('ascii')
                return self.send({'before':enc(im),'after':enc(result),'width':im.shape[1],'height':im.shape[0]})
            if path=='/api/cube':
                return self.send(g.cube_text(g.make_lut(body['weights'],body.get('strength',1),adjustments=body.get('adjustments'))).encode('ascii'),ctype='text/plain')
            if path=='/api/image':
                im=source_image(body['source'])
                return self.send(g.png_bytes(g.apply_lut(im,g.make_lut(body['weights'],body.get('strength',1),adjustments=body.get('adjustments')))),ctype='image/png')
            if path=='/api/export':
                src=asset(body['source'])
                if src.suffix.lower() not in ('.mp4','.mov','.mkv','.webm','.avi'): raise ValueError('请先载入视频')
                weights=g.normalized(body['weights']).tolist(); strength=float(body.get('strength',1))
                adjustments=g.controls(body.get('adjustments'));g.make_lut(weights,strength,adjustments=adjustments)
                ffmpeg()
                with LOCK:
                    if any(j['status']=='running' for j in JOBS.values()): raise ValueError('已有视频正在导出，请等待或取消')
                    jid=uuid.uuid4().hex; JOBS[jid]={'status':'running','processed_time':'00:00:00'}
                threading.Thread(target=export_worker,args=(jid,src,weights,strength,adjustments),daemon=True).start()
                return self.send({'id':jid})
            if path=='/api/cancel':
                job=JOBS.get(body['id'])
                if job and job['status']=='running':
                    job['cancelled']=True
                    if job.get('process'): job['process'].terminate()
                return self.send({'ok':True})
            return self.send({'error':'接口不存在'},404)
        except Exception as e: self.send({'error':str(e)},400)

if __name__=='__main__':
    print(f'Local prototype: http://127.0.0.1:{PORT}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
