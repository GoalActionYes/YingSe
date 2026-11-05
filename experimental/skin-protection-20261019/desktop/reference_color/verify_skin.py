"""Verify clean portable extraction, native launcher and actual offline protection."""
from pathlib import Path
import argparse, base64, hashlib, io, json, subprocess, time, urllib.request
from urllib.error import HTTPError
from PIL import Image
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('package',type=Path);args=parser.parse_args()
    root=args.package.resolve();exe=root/'ReferenceColor.exe'
    assert not (root/'data').exists(),'Use a fresh owned extraction for verification'
    manifest=json.loads((root/'package-manifest.json').read_text('utf8'))
    for row in manifest['files']:
        assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def launch(flag):subprocess.run([str(exe),flag],cwd=root.parent,check=True,timeout=60)
    launch('--check')
    ready=json.loads((root/'data/session.json').read_text('utf8'));url=ready['url']
    launch('--check');assert ready==json.loads((root/'data/session.json').read_text('utf8'))
    def api(path,value=None,headers=None):
        values={'Content-Type':'application/json'} if headers is None else headers
        data=json.dumps(value).encode() if isinstance(value,dict) else value
        with opener.open(urllib.request.Request(url+path,data=data,headers=values),timeout=180) as r:
            raw=r.read();return json.loads(raw) if r.headers.get_content_type()=='application/json' else raw
    def upload(path):
        return api('/api/media',path.read_bytes(),{'Content-Type':'application/octet-stream','X-Filename':path.name})
    def analyze(target,reference):
        return api('/api/analyze',dict(target_id=target['id'],reference_id=reference['id'],time_seconds=0))['mapping_id']
    results=[]
    def export(target,mapping,kind,flag):
        j=api('/api/export',dict(mapping_id=mapping,media_id=target['id'],strength=.8,kind=kind,skin_protection=flag))
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            j=api('/api/jobs/'+j['id'])
            if j['status'] in ('done','error','cancelled'):break
            time.sleep(.1)
        assert j['status']=='done',j
        raw=api(j['result_url']);assert len(raw)>100
        output=root/'data'/('verified-'+('skin' if flag else 'base')+'.'+kind);output.write_bytes(raw)
        results.append(dict(kind=kind,skin_protection=flag,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),settings=j['settings']))
        return raw,j
    try:
        audit=json.loads((root/'data/runtime-check.json').read_text('utf8'))
        for k in ('python','numpy','pillow','ffmpeg'):assert Path(audit[k]).resolve().is_relative_to(root)
        source=upload(root/'reference_color_app/demo/skin-target.png')
        ref=upload(root/'reference_color_app/demo/skin-reference.png');mapping=analyze(source,ref)
        preview=api('/api/preview',dict(mapping_id=mapping,strength=.8,skin_protection=True))
        assert preview['skin']['enabled'] and preview['skin']['skin_detected'],preview['skin']
        protected,j=export(source,mapping,'png',True)
        before,_=export(source,mapping,'png',False)
        def rgb(raw):return np.array(Image.open(io.BytesIO(raw)))
        assert np.any(rgb(before)!=rgb(protected)), 'Protection did not change the human sample'
        np.testing.assert_array_equal(rgb(protected),rgb(base64.b64decode(preview['after'].split(',')[1])))
        export(source,mapping,'cube',False)
        try:api('/api/export',dict(mapping_id=mapping,media_id=source['id'],strength=.8,kind='cube',skin_protection=True))
        except HTTPError as exc:assert exc.code==400
        else:raise AssertionError('Spatial protection silently lost in CUBE')
        api('/api/projects',dict(mapping_id=mapping,strength=.8,output_settings={'kind':'png','skin_protection':True}))
        assert api('/api/projects/latest')['output_settings']['skin_protection'] is True
        ff=audit['ffmpeg'];video_path=root/'data/verification-input.mp4'
        subprocess.run([ff,'-hide_banner','-loglevel','error','-nostdin','-y','-loop','1','-framerate','12',
            '-i',str(root/'reference_color_app/demo/skin-target.png'),'-f','lavfi','-i','sine=frequency=440:duration=0.5',
            '-t','1','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(video_path)],check=True,timeout=30)
        video=upload(video_path);vm=analyze(video,ref)
        _,j=export(video,vm,'mp4',True)
        assert j['settings']['frames']==30 and j['settings']['verified_frames']==30,j
        assert j['settings']['skin']['detected_frames']>0,j
        subprocess.run([ff,'-v','error','-xerror','-i',str(root/'data/verified-skin.mp4'),'-f','null','-'],check=True,timeout=30)
        export(video,vm,'mp4',False)
        report=dict(package=str(root),model=preview['skin'],exports=results,manifest_files=len(manifest['files']),native_launcher=True,offline_model=True,
                    coverage='Public human photo + static 1s clip with short audio; real motion verified separately; no broad accuracy claim')
        (root/'data/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
        print(json.dumps({'verified':True,'manifest_files':len(manifest['files']),'exports':len(results),'skin_frames':j['settings']['frames']},indent=2))
    finally:launch('--stop')


if __name__=='__main__':main()
