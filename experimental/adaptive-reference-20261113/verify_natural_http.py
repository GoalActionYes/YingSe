"""Real owned 4K source: preview latency and native short export with audio."""
import base64
import argparse
import io
import json
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import quote
from urllib.request import Request,urlopen
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from spatial_workbench import make_spatial_workbench
from reference_color_app.media import ffmpeg_path,probe_media


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='http-v2');parser.add_argument('--multiref',action='store_true');args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Run must be a single directory name')
    out=ROOT/'outputs/adaptive-reference-20261113/natural-video'/args.run;out.mkdir(exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['verify_natural_http.py','spatial_workbench.py','spatial_store.py','spatial_jobs.py','spatial_render.py','temporal_grid.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    source=ROOT/'outputs/视频兼容副本/P1106511-8bit-SDR.mp4';clip=out/'native-short.mp4'
    process=subprocess.run([str(ffmpeg_path()),'-v','error','-y','-i',str(source),'-t','2','-map','0','-c','copy',str(clip)],capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    if process.returncode:raise ValueError(process.stderr.decode('utf-8','replace'))
    server=make_spatial_workbench(out/'data',enhanced=args.multiref);worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start();url=f'http://127.0.0.1:{server.server_port}';report=dict(variant=server.variant,checkpoint_sha256=server.store.engine.checkpoint_sha)
    def api(path,data=None):
        request=Request(url+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=180) as response:
            raw=response.read();return json.loads(raw) if 'application/json' in response.headers.get('Content-Type','') else raw
    def upload(path):
        request=Request(url+'/api/media',data=path.read_bytes(),headers={'Content-Type':'application/octet-stream','X-Filename':quote(path.name)})
        with urlopen(request,timeout=180) as response:return json.load(response)
    try:
        dataset=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-v2';records=json.loads((dataset/'results.json').read_text());item=next(r for r in records['cases'] if r['index']==67849)['triplet']
        reference=upload(dataset/'srgb'/(item['reference']+'.png'));target=upload(source);start=time.perf_counter()
        mapping=api('/api/analyze',dict(target_id=target['id'],reference_id=reference['id'],time_seconds=0));report['full_source_analysis_seconds']=time.perf_counter()-start
        def preview(strength):
            start=time.perf_counter();value=api('/api/preview',dict(mapping_id=mapping['mapping_id'],strength=strength,interactive=True));return time.perf_counter()-start,value
        times=[]
        for strength in [0.,1.,.5,.2,.8]:
            seconds,result=preview(strength);times.append(seconds)
        report['preview_before_export_seconds']=times;print(json.dumps(report),flush=True)
        short=upload(clip);sm=api('/api/analyze',dict(target_id=short['id'],reference_id=reference['id']))
        start=time.perf_counter();job=api('/api/export',dict(mapping_id=sm['mapping_id'],media_id=short['id'],strength=1,kind='mp4'));job_id=job['id'];deadline=time.monotonic()+180
        while job_id not in server.manager.processes:
            if time.monotonic()>deadline:raise TimeoutError('Native export decoder did not start')
            time.sleep(.02)
        in_export=[]
        for strength in [.3,.6,.9]:
            seconds,value=preview(strength);in_export.append(seconds)
        while job['status'] in ['queued','running']:
            if time.monotonic()>deadline:raise TimeoutError('Native export did not finish')
            time.sleep(.1);job=api('/api/jobs/'+job_id)
        if job['status']!='done':raise ValueError(job)
        path=out/'native-export.mp4';path.write_bytes(api(job['result_url']));meta=probe_media(path)
        if (meta['width'],meta['height'])!=(2160,3840) or not meta['has_audio']:raise ValueError('Native output dimensions/audio changed')
        decoded=subprocess.run([str(ffmpeg_path()),'-v','error','-i',str(path),'-map','0:v:0','-an','-fps_mode','passthrough','-f','framemd5','pipe:1'],capture_output=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
        if decoded.returncode:raise ValueError('Native output could not decode')
        frame_count=sum(bool(line) and not line.startswith(b'#') for line in decoded.stdout.splitlines());(out/'decoded-frames.md5').write_bytes(decoded.stdout)
        input_frames=subprocess.run([str(ffmpeg_path()),'-v','error','-i',str(clip),'-map','0:v:0','-an','-vf','fps=30','-fps_mode','passthrough','-f','framemd5','pipe:1'],capture_output=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
        if input_frames.returncode:raise ValueError('Cannot count source CFR frames')
        expected_frames=sum(bool(line) and not line.startswith(b'#') for line in input_frames.stdout.splitlines())
        if frame_count!=expected_frames or frame_count!=job['settings']['frames']:raise ValueError('Export frame count mismatch')
        if args.multiref and job['settings']['temporal']!=dict(alpha=.2,cut_threshold=.16):raise ValueError('Temporal export not enabled')
        report.update(source_sha256=digest(source),clip_metadata=probe_media(clip),output_metadata=meta,output_sha256=digest(path),
            output_decoded_frames=frame_count,input_cfr_frames=expected_frames,preview_during_export_seconds=in_export,export_wall_seconds=time.perf_counter()-start,
            source_code_sha256={p.name:digest(p) for p in frozen.iterdir()},
            scope='Single natural 4K short export and full-source latency; no browser/AV perceptual acceptance',passed=True)
    finally:
        server.shutdown();server.server_close();worker.join(timeout=5)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['clip_metadata','output_metadata']}),flush=True)


if __name__=='__main__':main()
