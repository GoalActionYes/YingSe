"""Real HTTP smoke using a portable exe, runnable outside the source tree."""
from pathlib import Path
import argparse, hashlib, json, socket, subprocess, time, urllib.request

def main():
    parser=argparse.ArgumentParser();parser.add_argument('package',type=Path);args=parser.parse_args()
    root=args.package.resolve();exe=root/'ReferenceColor.exe'
    def launch(flag,expected=0):
        result=subprocess.run([str(exe),flag],cwd=root.parent,timeout=60)
        assert result.returncode==expected,(flag,result.returncode)
    def api(path,value=None):
        data=None if value is None else json.dumps(value).encode()
        req=urllib.request.Request(url+path,data=data,headers={'Content-Type':'application/json'})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=120) as response:
            raw=response.read()
            return json.loads(raw) if response.headers.get_content_type()=='application/json' else raw
    manifest=json.loads((root/'package-manifest.json').read_text(encoding='utf8'))
    for entry in manifest['files']:
        assert hashlib.sha256((root/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
    assert not (root/'data').exists(),'Use a fresh owned verification copy'
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1',0));occupied.listen()
        launch('--check')
        ready=json.loads((root/'data/session.json').read_text(encoding='utf8'));url=ready['url']
        assert int(url.rsplit(':',1)[1]) != occupied.getsockname()[1]
    launch('--check');assert json.loads((root/'data/session.json').read_text(encoding='utf8'))==ready
    audit=json.loads((root/'data/runtime-check.json').read_text(encoding='utf8'))
    for key in ('python','numpy','pillow','ffmpeg'): assert Path(audit[key]).resolve().is_relative_to(root)
    results=[]
    for mode,kinds in (('image',('png','cube')),('video',('mp4',))):
        demo=api('/api/demo',{'kind':mode});target=demo['target']['id']
        mapping=api('/api/analyze',dict(target_id=target,reference_id=demo['reference']['id'],time_seconds=0))['mapping_id']
        for kind in kinds:
            job=api('/api/export',dict(mapping_id=mapping,media_id=target,strength=.8,kind=kind))
            for _ in range(600):
                job=api('/api/jobs/'+job['id'])
                if job['status'] in ('done','error','cancelled'): break
                time.sleep(.1)
            assert job['status']=='done',job
            result=api(job['result_url']); assert len(result)>100
            output=root/'data'/('verified.'+kind);output.write_bytes(result)
            if kind=='png':
                from PIL import Image
                with Image.open(output) as image:
                    image.load(); assert image.size==(960,540),image.size
            if kind=='mp4':
                assert b'ftyp' in result[:32]
                decoded=subprocess.run([audit['ffmpeg'],'-v','error','-i',str(output),'-frames:v','1','-f','image2pipe','-c:v','png','pipe:1'],capture_output=True,timeout=30)
                assert decoded.returncode==0 and decoded.stdout.startswith(b'\x89PNG'),decoded.stderr
            if kind=='cube': assert b'LUT_3D_SIZE' in result
            results.append(dict(kind=kind,bytes=len(result),sha256=hashlib.sha256(result).hexdigest()))
    launch('--stop');assert not (root/'data/session.json').exists()
    # A stale ready pointing at a real app but with incorrect identity is rejected.
    launch('--check'); fresh=json.loads((root/'data/session.json').read_text(encoding='utf8'));url=fresh['url']
    launch('--stop')
    (root/'data/session.json').write_text(json.dumps(ready),encoding='utf8')
    launch('--check');assert json.loads((root/'data/session.json').read_text(encoding='utf8'))['token']!=ready['token']
    launch('--stop')
    ff=Path(audit['ffmpeg']);hidden=ff.with_suffix('.verification-missing')
    ff.rename(hidden)
    try: launch('--check',1)
    finally: hidden.rename(ff)
    launch('--check');launch('--stop')
    report=dict(package=str(root),runtime=audit,exports=results,reuse=True,restart=True,occupied_port=True,stale_ready=True,missing_ffmpeg=True)
    (root/'data/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
