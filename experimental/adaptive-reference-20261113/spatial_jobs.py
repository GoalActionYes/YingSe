"""Owned asynchronous PNG/video export for the spatial academic workbench."""
import os
import subprocess
import numpy as np
from PIL import Image
from reference_color_app.jobs import JobManager
from reference_color_app import media
from temporal_grid import TemporalGrid
from spatial_render import finish_spatial


def grade_video_frame(engine,rgb,reference,strength,adjustments,temporal=None):
    if strength==0 or np.array_equal(rgb,reference):
        graded=rgb
        if temporal is not None:temporal.reset()
    else:
        prepared=engine.prepare(rgb,reference)
        if temporal is not None:prepared=temporal.smooth(prepared,rgb)
        graded=prepared.apply(rgb,1)
    return finish_spatial(rgb,graded,strength,adjustments)


class SpatialJobs(JobManager):
    def __init__(self,store,temporal=False):super().__init__(store);self.encoders={};self.temporal_enabled=temporal

    def temporal_state(self):
        return TemporalGrid(relative_cut=True) if self.temporal_enabled else None

    def start_export(self,mapping_id,media_id,strength,kind,adjustments=None):
        if kind=='cube':raise ValueError('局部调色无法完整导出为普通CUBE，请选择PNG或MP4')
        self.store.get_prepared(mapping_id)
        return super().start_export(mapping_id,media_id,strength,kind,adjustments)

    def get_job(self,job_id):
        # Progress is maintained by the Python frame loop, not FFmpeg out_time.
        with self.lock:
            import copy
            return copy.deepcopy(self.jobs[job_id])

    def cancel_job(self,job_id):
        result=super().cancel_job(job_id)
        with self.lock:
            process=self.encoders.get(job_id)
            if process is not None and process.poll() is None:process.terminate()
        return result

    def _run(self,identifier,placeholder,source):
        directory=self.store.root/'jobs'/identifier;decoder=encoder=None;log=None
        try:
            with self.lock:
                job=self.jobs[identifier]
                if job['status']=='cancelled':return
                job.update(status='running',duration_seconds=source['metadata'].get('duration_seconds'));self._persist(job)
            kind=job['kind'];partial=directory/('partial.'+kind)
            if kind=='png':
                rgb,meta=self.store.decode(source)
                output=self.store.render(job['mapping_id'],rgb,job['strength'],job['adjustments'])
                Image.fromarray(output).save(partial,format='PNG')
                settings=dict(width=rgb.shape[1],height=rgb.shape[0],format='RGB8 PNG',color='prepared sRGB',lossless=True)
            else:
                first,_=self.store.decode(source,0);height,width=first.shape[:2]
                reference=self.store.get_media(self.store.get_mapping_record(job['mapping_id'])['reference_id']);ref,_=self.store.decode(reference)
                duration=source['metadata'].get('duration_seconds',0)
                log=(directory/'ffmpeg.log').open('wb');flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
                decoder_args=[str(media.ffmpeg_path()),'-v','error','-nostdin','-i',source['path'],'-an','-vf',f'fps=30,scale={width}:{height}','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
                encoder_args=[str(media.ffmpeg_path()),'-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','30','-i','pipe:0','-i',source['path'],'-map','0:v:0','-map','1:a:0?','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p','-c:v','libx264','-crf','18','-c:a','aac','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-color_range','tv','-movflags','+faststart',str(partial)]
                with self.lock:
                    if job['status']=='cancelled':return
                    decoder=subprocess.Popen(decoder_args,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=log,creationflags=flags)
                    self.processes[identifier]=decoder
                    encoder=subprocess.Popen(encoder_args,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=flags)
                    self.encoders[identifier]=encoder
                count=0;frame_size=width*height*3;temporal=self.temporal_state()
                while True:
                    with self.lock:
                        if job['status']=='cancelled':return
                    raw=bytearray()
                    while len(raw)<frame_size:
                        chunk=decoder.stdout.read(frame_size-len(raw))
                        if not chunk:break
                        raw.extend(chunk)
                    if not raw:break
                    if len(raw)!=frame_size:raise ValueError('视频解码帧不完整')
                    rgb=np.frombuffer(raw,np.uint8).reshape(height,width,3)
                    with self.store.render_lock:
                        output=grade_video_frame(self.store.engine,rgb,ref,job['strength'],job['adjustments'],temporal)
                    encoder.stdin.write(output.tobytes());count+=1
                    with self.lock:
                        job['progress']=min(.99,count/max(1,duration*30))
                        if count%15==0:self._persist(job)
                encoder.stdin.close();encoder.stdin=None
                if decoder.wait(timeout=30) or encoder.wait(timeout=120):raise ValueError('视频编解码失败，请查看导出记录')
                if count==0:raise ValueError('视频未解码出任何帧')
                settings=dict(width=width,height=height,frames=count,fps=30,frame_rate_policy='Resampled to constant 30fps',
                    algorithm='Recompute spatial grid per frame from constant reference',audio='First optional source audio track, AAC',
                    temporal=temporal.settings() if temporal is not None else None,
                    encoding='Lossy H264 CRF18/YUV420p, SDR BT709 limited',output_metadata=media.probe_media(partial))
            with self.lock:
                if job['status']=='cancelled':return
                os.replace(partial,directory/('result.'+kind));job.update(status='done',settings=settings,progress=1.);self._persist(job)
        except Exception as exc:
            with self.lock:
                if self.jobs[identifier]['status']!='cancelled':self.jobs[identifier].update(status='error',error=str(exc));self._persist(self.jobs[identifier])
        finally:
            for process in [decoder,encoder]:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                    for stream in [process.stdin,process.stdout]:
                        if stream is not None:
                            try:stream.close()
                            except OSError:pass
            if log is not None:log.close()
            with self.lock:self.processes.pop(identifier,None);self.encoders.pop(identifier,None)
            for name in ['partial.png','partial.mp4']:(directory/name).unlink(missing_ok=True)
