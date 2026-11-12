"""Real HTTP image/video analysis and export check against the running research UI."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys
import subprocess
import time
from urllib.request import Request,urlopen
import numpy as np
from PIL import Image
from canonical_model import ROOT
sys.path.insert(0,str(ROOT))
from reference_color_app import media


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);args=parser.parse_args()
    url=args.url.rstrip('/')
    def raw(path,data=None,headers=None):
        request=Request(url+path,data=data,headers=headers or {},method='POST' if data is not None else 'GET')
        with urlopen(request,timeout=180) as response:return response.read()
    def api(path,data=None):
        return json.loads(raw(path,None if data is None else json.dumps(data).encode(),{'Content-Type':'application/json'}))
    def upload(path):
        return json.loads(raw('/api/media',path.read_bytes(),{'Content-Type':'application/octet-stream','X-Filename':path.name}))
    def pixels(image):return np.array(Image.open(io.BytesIO(base64.b64decode(image.split(',',1)[1]))).convert('RGB'))
    def export(mapping,source,kind,adjustments=None):
        job=api('/api/export',dict(mapping_id=mapping,media_id=source,strength=1,kind=kind,adjustments=adjustments))
        deadline=time.monotonic()+180
        while job['status'] not in ('done','error','cancelled'):
            assert time.monotonic()<deadline,job
            time.sleep(.1);job=api('/api/jobs/'+job['id'])
        assert job['status']=='done',job
        return raw(job['result_url']),json.loads(raw(job['report_url']))
    out=ROOT/'outputs/adaptive-reference-20261113/workbench-verification';out.mkdir(parents=True,exist_ok=True)
    state=api('/api/engine-state');assert state['state']=='ready',state
    data=ROOT/'outputs/adaptive-reference-20261113/cross-scene'
    target=upload(data/'target-0.png');reference=upload(data/'reference-1-warm.png')
    start=time.perf_counter()
    model=api('/api/analyze',dict(target_id=target['id'],reference_id=reference['id'],engine='model'))
    analyze_seconds=time.perf_counter()-start
    baseline=api('/api/analyze',dict(target_id=target['id'],reference_id=reference['id'],engine='baseline'))
    assert model['mapping_id']!=baseline['mapping_id']
    zero=api('/api/preview',dict(mapping_id=model['mapping_id'],strength=0,interactive=True))
    assert zero['before']==zero['after']
    preview=api('/api/preview',dict(mapping_id=model['mapping_id'],strength=1))
    direct=np.array(Image.open(ROOT/'outputs/adaptive-reference-20261113/canonical-model/0-1-warm.png').convert('RGB'))
    error=np.abs(pixels(preview['after']).astype(int)-direct)
    assert np.quantile(error,.95)<=2 and error.max()<=8,(error.mean(),error.max())
    adjustments=dict(basic=dict(temperature=20))
    manual=api('/api/preview',dict(mapping_id=model['mapping_id'],strength=1,adjustments=adjustments))
    assert manual.get('preview_renderer')=='gpu',manual.get('preview_renderer')
    preview_times=[]
    for value in (.25,.5,.75):
        start=time.perf_counter()
        api('/api/preview',dict(mapping_id=model['mapping_id'],strength=value,interactive=True,adjustments=adjustments))
        preview_times.append(time.perf_counter()-start)
    png,report=export(model['mapping_id'],target['id'],'png',adjustments)
    exported=np.array(Image.open(io.BytesIO(png)))
    export_error=np.abs(exported.astype(int)-pixels(manual['after']).astype(int))
    assert export_error.max()<=1
    (out/'manual-result.png').write_bytes(png)
    cube,cubereport=export(model['mapping_id'],target['id'],'cube',adjustments)
    assert b'LUT_3D_SIZE 64' in cube
    (out/'manual-result.cube').write_bytes(cube)
    api('/api/projects',dict(mapping_id=model['mapping_id'],strength=1,
                            output_settings=dict(kind='png',adjustments=adjustments)))
    restored=api('/api/projects/latest');assert restored['engine']=='model'
    identity=api('/api/analyze',dict(target_id=target['id'],reference_id=target['id'],engine='model'))
    identitypreview=api('/api/preview',dict(mapping_id=identity['mapping_id'],strength=1))
    assert identitypreview['before']==identitypreview['after']
    video=upload(ROOT/'reference_color_app/demo/source-motion.mp4')
    videomap=api('/api/analyze',dict(target_id=video['id'],reference_id=reference['id'],engine='model'))
    mp4,videoreport=export(videomap['mapping_id'],video['id'],'mp4',adjustments)
    videopath=out/'manual-video.mp4';videopath.write_bytes(mp4)
    frame,metadata=media.decode_frame(videopath,0)
    assert frame.size>0
    def frame_count(path):
        result=subprocess.run([str(media.ffmpeg_path()),'-hide_banner','-nostdin','-v','error','-i',str(path),
                               '-map','0:v:0','-progress','pipe:1','-f','null','-'],
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert result.returncode==0,result.stderr[-2000:]
        return int([line.split('=',1)[1] for line in result.stdout.splitlines() if line.startswith('frame=')][-1])
    input_count=frame_count(ROOT/'reference_color_app/demo/source-motion.mp4')
    output_count=frame_count(videopath)
    assert input_count==output_count and output_count>0
    summary=dict(real_model_http=True,model_state=state,model_analyze_seconds=analyze_seconds,
                 model_analysis_cache_hit=model.get('cache_hit'),
                 compiled_vs_direct_mae=float(error.mean()),compiled_vs_direct_max=int(error.max()),
                 compiled_vs_direct_p95=float(np.quantile(error,.95)),zero_strength_exact=True,
                 image_identity_exact=True,png_vs_full_preview_max=int(export_error.max()),cube_size64=True,
                 preview_renderer=manual.get('preview_renderer','unknown'),
                 http_interactive_preview_seconds=preview_times,
                 restored_engine=restored['engine'],video_output=metadata,
                 entire_video_decoded=True,input_frames=input_count,output_frames=output_count,
                 scope='One real photo pair and one synthetic short video; not general quality or full realtime playback')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
