"""Actual HTTP workflows in an isolated owned server/data directory."""
import base64
import argparse
import io
import json
from pathlib import Path
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request,urlopen
import numpy as np
from PIL import Image
from independent_coco import ROOT,digest
from spatial_workbench import make_spatial_workbench
from reference_color_app.adjustments import normalize,transform
from reference_color_app.media import ffmpeg_path,probe_media
from temporal_grid import TemporalGrid


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='verification-v4');parser.add_argument('--multiref',action='store_true');args=parser.parse_args()
    if not args.run or Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Run must be a single directory name')
    out=ROOT/'outputs/adaptive-reference-20261113/spatial-workbench'/args.run;out.mkdir(parents=True,exist_ok=False)
    frozen=out/'frozen-source';frozen.mkdir()
    for name in ['spatial_workbench.py','spatial_store.py','spatial_jobs.py','spatial_render.py','spatial_build.py','temporal_grid.py','verify_spatial_workbench.py']:(frozen/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    server=make_spatial_workbench(out/'data',enhanced=args.multiref);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url=f'http://127.0.0.1:{server.server_port}';report={}
    def api(path,data=None):
        req=Request(url+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=120) as response:
            raw=response.read();return json.loads(raw) if 'application/json' in response.headers.get('Content-Type','') else raw
    def upload(path):
        req=Request(url+'/api/media',data=path.read_bytes(),headers={'Content-Type':'application/octet-stream','X-Filename':quote(path.name)})
        with urlopen(req,timeout=120) as response:return json.load(response)
    def decode(uri):return np.array(Image.open(io.BytesIO(base64.b64decode(uri.split(',',1)[1]))).convert('RGB'))
    def expect400(path,data):
        try:api(path,data)
        except HTTPError as exc:
            if exc.code!=400:raise
            return json.loads(exc.read())
        raise AssertionError('Invalid request was accepted')
    def export(mapping,target,strength,kind,adjustments=None):
        job=api('/api/export',dict(mapping_id=mapping,media_id=target,strength=strength,kind=kind,adjustments=adjustments))
        deadline=time.monotonic()+90
        while job['status'] in ['queued','running']:
            if time.monotonic()>deadline:raise TimeoutError('Owned export did not finish')
            time.sleep(.05);job=api('/api/jobs/'+job['id'])
        if job['status']!='done':raise ValueError(job)
        return job,api(job['result_url'])
    try:
        health=api('/api/health');assert health['variant']==('multiref-temporal-v1' if args.multiref else 'app8')
        report['variant']=health['variant'];report['checkpoint_sha256']=server.store.engine.checkpoint_sha
        html=api('/').decode();assert 'option value="cube"' not in html and '课程研究' in html
        data=ROOT/'outputs/adaptive-reference-20261113/tst100k/reserved-v2';records=json.loads((data/'results.json').read_text())
        item=next(r for r in records['cases'] if r['index']==67849)['triplet']
        target_path=out/'target.png';ref_path=out/'reference.png'
        with Image.open(data/'srgb'/(item['content']+'.png')) as im:im.convert('RGB').resize((1920,1080),Image.Resampling.BILINEAR).save(target_path)
        with Image.open(data/'srgb'/(item['reference']+'.png')) as im:im.convert('RGB').save(ref_path)
        target,reference=upload(target_path),upload(ref_path);start=time.perf_counter()
        for mode,color in [('RGBA',(20,80,160,140)),('L',93)]:
            picture=out/(mode+'.png');image=Image.new(mode,(31,19),color);image.save(picture);uploaded=upload(picture)
            result=api('/api/frame',dict(media_id=uploaded['id']))
            assert np.array_equal(decode(result['image']),np.array(image.convert('RGB')))
        report['rgba_gray_http']=True
        start=time.perf_counter()
        mapping=api('/api/analyze',dict(target_id=target['id'],reference_id=reference['id']));mid=mapping['mapping_id'];report['analysis_seconds']=time.perf_counter()-start
        previews={};timings=[]
        for strength in [0.,.5,1.,.3,.6,.9]:
            start=time.perf_counter();result=api('/api/preview',dict(mapping_id=mid,strength=strength,interactive=True,include_basis=args.multiref and strength==0));timings.append(time.perf_counter()-start);previews[strength]=decode(result['after'])
            if strength==0:before=decode(result['before'])
            if args.multiref and strength==0:
                basis=decode(result['basis']['full']);assert basis.shape==before.shape
        assert np.array_equal(previews[0.],before)
        if args.multiref:
            assert np.array_equal(basis,previews[1.])
            report['verified_basis_matches_full_preview']=True
        expected=np.rint((before.astype(float)+previews[1.].astype(float))*.5).astype(np.uint8)
        assert np.abs(expected.astype(int)-previews[.5].astype(int)).max()<=1
        report['preview_seconds']=timings
        settings=normalize();settings['basic']['brightness']=20
        changed=api('/api/preview',dict(mapping_id=mid,strength=1,adjustments=settings,interactive=True));assert not np.array_equal(decode(changed['after']),previews[1.])
        if args.multiref:
            changed_basis=api('/api/preview',dict(mapping_id=mid,strength=1,adjustments=settings,interactive=True,include_basis=True))
            assert changed_basis['after']==changed['after'] and np.array_equal(decode(changed_basis['basis']['full']),basis)
            report['basis_is_independent_of_manual_adjustments']=True
        sample=api('/api/curve-sample',dict(mapping_id=mid,strength=.5,x=.5,y=.5,adjustments=settings));assert sample['pixel']==[960,540]
        api('/api/projects',dict(mapping_id=mid,strength=.5,output_settings=dict(kind='png',clone_enabled=True,adjustments=settings)))
        restored=api('/api/projects/latest');assert restored['mapping_id']==mid and restored['strength']==.5
        job,raw=export(mid,target['id'],.5,'png',settings);(out/'export.png').write_bytes(raw)
        exported=np.array(Image.open(io.BytesIO(raw)).convert('RGB'));source,meta=server.store.decode(server.store.get_media(target['id']))
        oracle=server.store.render(mid,source,.5,settings);assert exported.shape==(1080,1920,3) and np.array_equal(exported,oracle)
        report['png']=dict(dimensions=[1920,1080],matches_server_renderer=True,sha256=digest(out/'export.png'))
        report['cube_rejection']=expect400('/api/export',dict(mapping_id=mid,media_id=target['id'],strength=1,kind='cube'))
        video=upload(ROOT/'outputs/skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4')
        vm=api('/api/analyze',dict(target_id=video['id'],reference_id=reference['id'],time_seconds=1))
        start=time.perf_counter();vj,vraw=export(vm['mapping_id'],video['id'],1,'mp4');video_path=out/'video.mp4';video_path.write_bytes(vraw)
        metadata=probe_media(video_path);assert metadata['width']==640 and metadata['height']==360 and abs(metadata['duration_seconds']-3)<.1
        decoded=subprocess.run([str(ffmpeg_path()),'-v','error','-i',str(video_path),'-an','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
        assert decoded.returncode==0 and len(decoded.stdout)//(640*360*3)==90
        report['video']=dict(wall_seconds=time.perf_counter()-start,output_frames=90,fps=30,dimensions=[640,360],duration=metadata['duration_seconds'],sha256=digest(video_path))
        if args.multiref:
            assert vj['settings']['temporal']==dict(alpha=.2,cut_threshold=.16)
            decoded_source=subprocess.run([str(ffmpeg_path()),'-v','error','-i',str(ROOT/'outputs/skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4'),'-vf','fps=30','-an','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
            assert decoded_source.returncode==0 and len(decoded_source.stdout)==90*640*360*3
            frames=np.frombuffer(decoded_source.stdout,np.uint8).reshape(90,360,640,3)
            exported_frames=np.frombuffer(decoded.stdout,np.uint8).reshape(90,360,640,3)
            ref,_=server.store.decode(server.store.get_media(reference['id']));temporal=TemporalGrid();checks=[]
            for index,frame in enumerate(frames):
                prepared=temporal.smooth(server.store.engine.prepare(frame,ref),frame)
                if index in [0,30,60,89]:
                    expected=prepared.apply(frame);error=float(np.abs(expected.astype(float)-exported_frames[index].astype(float)).mean())
                    assert error<5,('Temporal export diverged from sequential oracle',index,error)
                    checks.append(dict(frame=index,lossy_encoded_mae=error))
                    if index==30:
                        static=server.store.render(vm['mapping_id'],frame,1)
                        report['video_single_frame_preview_vs_temporal_mae']=float(np.abs(static.astype(float)-expected.astype(float)).mean())
            report['video']['temporal_oracle_samples']=checks
        audio_path=out/'input-audio.mp4'
        tone=subprocess.run([str(ffmpeg_path()),'-v','error','-y','-i',str(ROOT/'outputs/skin-protection-validation-20261019/video-agent/official-person-motion-3s.mp4'),'-f','lavfi','-i','sine=frequency=440:duration=3','-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac',str(audio_path)],capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
        assert tone.returncode==0 and probe_media(audio_path)['has_audio']
        audio=upload(audio_path);am=api('/api/analyze',dict(target_id=audio['id'],reference_id=reference['id']))
        video_settings=normalize();video_settings['basic']['brightness']=12;video_settings['basic']['contrast']=10
        aj,bytes_audio=export(am['mapping_id'],audio['id'],.5,'mp4',video_settings);audio_output=out/'audio-export.mp4';audio_output.write_bytes(bytes_audio)
        audio_meta=probe_media(audio_output);assert audio_meta['has_audio'] and abs(audio_meta['duration_seconds']-3)<.1
        report['audio_export']=dict(has_audio=True,duration=audio_meta['duration_seconds'],sha256=digest(audio_output))
        if args.multiref:
            decoded_audio=subprocess.run([str(ffmpeg_path()),'-v','error','-i',str(audio_output),'-an','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
            assert decoded_audio.returncode==0 and len(decoded_audio.stdout)==90*640*360*3
            actual=np.frombuffer(decoded_audio.stdout,np.uint8).reshape(90,360,640,3);temporal=TemporalGrid();checks=[]
            for index,frame in enumerate(frames):
                prepared=temporal.smooth(server.store.engine.prepare(frame,ref),frame)
                if index in [0,30,89]:
                    full=prepared.apply(frame,1)
                    mixed=frame.astype(np.float32)/255*.5+full.astype(np.float32)/255*.5
                    expected=np.rint(transform(mixed,video_settings)*255).astype(np.uint8)
                    error=float(np.abs(expected.astype(float)-actual[index].astype(float)).mean())
                    assert error<5,('Half strength and manual settings diverged',index,error)
                    checks.append(dict(frame=index,lossy_encoded_mae=error))
            report['audio_export']['half_strength_manual_oracle_samples']=checks
        cancel=api('/api/export',dict(mapping_id=am['mapping_id'],media_id=audio['id'],strength=1,kind='mp4'))
        deadline=time.monotonic()+10
        while cancel['id'] not in server.manager.processes:
            if time.monotonic()>deadline:raise TimeoutError('No owned live decoder to cancel')
            time.sleep(.01)
        assert server.manager.processes[cancel['id']].poll() is None
        try:api('/api/stop',dict(token=server.token))
        except HTTPError as exc:
            assert exc.code==409
        else:raise AssertionError('Service stopped with an active export')
        report['active_export_stop_rejected']=True
        cancelled=api('/api/jobs/'+cancel['id']+'/cancel',{});assert cancelled['status']=='cancelled'
        while cancel['id'] in server.manager.processes:
            if time.monotonic()>deadline:raise TimeoutError('Owned cancellation did not clean up')
            time.sleep(.02)
        directory=server.store.root/'jobs'/cancel['id'];assert not (directory/'result.mp4').exists() and not (directory/'partial.mp4').exists()
        report['cancel_running_export']=True
        report['source_sha256']={p.name:digest(p) for p in frozen.iterdir()}
        report['passed']=True;report['scope']='Actual HTTP workflows including audio and running cancellation; no visual browser acceptance'
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)
        (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
