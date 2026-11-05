"""Asynchronous export with cancellation and atomic result publication."""
import copy, json, os, subprocess, threading, uuid
from PIL import Image
from . import media
from .storage import _json
from .lut import apply_mapping, cube_text, _strength
from .adjustments import normalize, compose
from .pipeline import layer_sha256

class JobManager:
    def __init__(self,store):
        self.store=store; self.lock=threading.RLock(); self.jobs={}; self.processes={}
        for report in (store.root/'jobs').glob('*/report.json'):
            try:
                job=json.loads(report.read_text(encoding='utf-8'))
                if job['kind'] not in ('png','mp4','cube'): continue
                if report.parent.name!=job['id'] or len(job['id'])!=32 or any(c not in '0123456789abcdef' for c in job['id']): continue
                if job['status'] in ('queued','running'): job.update(status='error',error='Export interrupted by application shutdown')
                if job['status']=='done' and not (report.parent/('result.'+job['kind'])).is_file(): job.update(status='error',error='Completed output missing')
                self.jobs[job['id']]=job; _json(report,job)
            except (OSError,ValueError,KeyError,TypeError): continue
    def _persist(self,job):
        directory=self.store.root/'jobs'/job['id']; directory.mkdir(exist_ok=True)
        _json(directory/'report.json',job)
    def active_jobs(self):
        with self.lock: return [copy.deepcopy(j) for j in self.jobs.values() if j['status'] in ('queued','running')]
    def has_active_jobs(self): return bool(self.active_jobs())
    def start_export(self,mapping_id,media_id,strength,kind,adjustments=None):
        if kind not in ('png','mp4','cube'): raise ValueError('Unknown export kind')
        strength=float(_strength(strength)); record=self.store.get_mapping_record(mapping_id)
        if record['target_id']!=media_id: raise ValueError('Mapping belongs to a different source ID')
        mapping=self.store.get_mapping(mapping_id); source=self.store.get_media(media_id)
        if kind=='png' and source['kind']!='image' or kind=='mp4' and source['kind']!='video': raise ValueError('Export kind does not match media')
        adjustments=normalize(adjustments)
        identifier=uuid.uuid4().hex
        with self.lock:
            self.jobs[identifier]=dict(id=identifier,status='queued',kind=kind,mapping_id=mapping_id,media_id=media_id,strength=strength,adjustments=adjustments,error=None,settings={},progress=0.0,diagnostics=record['diagnostics'])
            self._persist(self.jobs[identifier])
        threading.Thread(target=self._run,args=(identifier,mapping,source),daemon=True).start(); return identifier
    def get_job(self,job_id):
        with self.lock:
            job=self.jobs[job_id]
            if job['status']=='running' and job['kind']=='mp4':
                path=self.store.root/'jobs'/job_id/'progress.txt'
                try:
                    with path.open('rb') as stream:
                        stream.seek(max(0,path.stat().st_size-65536)); lines=stream.read(65536).decode('ascii','ignore').splitlines()
                    times=[int(line.split('=',1)[1]) for line in lines if line.startswith('out_time_us=') and line.split('=',1)[1].lstrip('-').isdigit()]
                    if times: job['progress']=min(.999,max(0,times[-1]/1e6/job['duration_seconds']))
                except OSError: pass
            return copy.deepcopy(job)
    def cancel_job(self,job_id):
        with self.lock:
            job=self.jobs[job_id]
            if job['status'] in ('queued','running'):
                job['status']='cancelled'
                process=self.processes.get(job_id)
                if process is not None and process.poll() is None:
                    try: process.terminate()
                    except ProcessLookupError: pass
                self._persist(job)
            return copy.deepcopy(job)
    def result_file(self,job_id,artifact='result'):
        if artifact not in ('result','report'): raise ValueError('Invalid artifact')
        with self.lock:
            job=self.jobs[job_id]
            if job['status']!='done': raise ValueError('Result is unavailable')
            return self.store.root/'jobs'/job_id/('report.json' if artifact=='report' else 'result.'+job['kind'])
    def _run(self,identifier,mapping,source):
        directory=self.store.root/'jobs'/identifier
        try:
            with self.lock:
                job=self.jobs[identifier]
                if job['status']=='cancelled': return
                job['status']='running'; job['duration_seconds']=source['metadata'].get('duration_seconds'); self._persist(job)
            mapping,effective_strength=compose(mapping,job['strength'],job['adjustments'])
            with self.lock:
                job.update(diagnostics=mapping.diagnostics,manual_algorithm_version=1,
                           effective_mapping_sha256=layer_sha256(mapping.layers),effective_strength=effective_strength)
                self._persist(job)
            directory.mkdir(exist_ok=True); kind=job['kind']; partial=directory/('partial.'+kind)
            if kind=='png':
                rgb=media.decode_image(source['path']); Image.fromarray(apply_mapping(rgb,mapping,effective_strength)).save(partial,format='PNG')
                settings=dict(width=rgb.shape[1],height=rgb.shape[0],format='RGB PNG',lossless=True)
            elif kind=='cube':
                partial.write_text(cube_text(mapping,effective_strength),encoding='ascii'); settings=dict(size=64,combined_final_strength=True)
            else:
                job['settings']=dict(codec='libx264',crf=18,audio='first optional audio track; AAC',fps='source timestamps; passthrough',padding='ceil(width/2)*2, ceil(height/2)*2',color='SDR BT709 limited YUV420P; explicit RGB conversion',encoding_loss='Lossy H264/AAC; separate from mapping fidelity',input_component_bit_depths=source['metadata'].get('component_bit_depths'),working_precision_bits=8,precision_conversion=source['metadata'].get('precision_conversion',False))
                with self.lock: self._persist(job)
                (directory/'grade.cube').write_text(cube_text(mapping,effective_strength),encoding='ascii')
                filters='format=rgb24,lut3d=file=grade.cube:interp=trilinear,pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p'
                args=[str(media.ffmpeg_path()),'-hide_banner','-nostdin','-y','-progress','progress.txt','-i',source['path'],'-map','0:v:0','-map','0:a:0?','-vf',filters,'-fps_mode','passthrough','-c:v','libx264','-crf','18','-c:a','aac','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709','-color_range','tv','-movflags','+faststart',str(partial)]
                with (directory/'ffmpeg.log').open('wb') as log:
                    with self.lock:
                        if job['status']=='cancelled': return
                        process=subprocess.Popen(args,cwd=directory,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=log,shell=False,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)); self.processes[identifier]=process
                    try: code=process.wait(timeout=3600)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(); raise ValueError('FFmpeg timed out')
                if code: raise ValueError('FFmpeg export failed: '+(directory/'ffmpeg.log').read_bytes()[-4000:].decode('utf-8','replace'))
                _,output_metadata=media.decode_frame(partial,0.0)
                settings=dict(job['settings'],output_metadata=output_metadata)
            with self.lock:
                if job['status']=='cancelled': return
                os.replace(partial,directory/('result.'+kind)); job.update(status='done',settings=settings,progress=1.0); self._persist(job)
        except Exception as exc:
            with self.lock:
                if self.jobs[identifier]['status']!='cancelled': self.jobs[identifier].update(status='error',error=str(exc)); self._persist(self.jobs[identifier])
        finally:
            with self.lock: self.processes.pop(identifier,None)
            # Delete only specifically named owned intermediate artifacts.
            if directory.parent==self.store.root/'jobs' and directory.name==identifier:
                for name in ('partial.png','partial.mp4','partial.cube','grade.cube'): (directory/name).unlink(missing_ok=True)
