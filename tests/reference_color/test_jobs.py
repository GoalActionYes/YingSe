import json, subprocess, tempfile, threading, time, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from reference_color_app import jobs, pipeline, media
from reference_color_app.contracts import Mapping
from reference_color_app.storage import Store
from reference_color_app.jobs import JobManager
from reference_color_app.lut import apply_mapping

def identity():
 axis=np.linspace(0,1,64,dtype=np.float32)
 cube=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
 return Mapping(np.stack([cube,cube]),dict(parameter_buffer=[1,None],nonfinite_diagnostic_policy='null preserves index'))

class JobTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
  self.p=self.root/'a.png'; Image.fromarray(np.random.default_rng(9).integers(0,256,(9,13,3),dtype=np.uint8)).save(self.p)
  self.s=Store(self.root/'store'); self.a=self.s.add_media(self.p); self.b=self.s.add_media(self.p)
 def analyze(self,source=None,t=0):
  with patch.object(pipeline,'analyze',return_value=identity()): return self.s.analyze((source or self.a)['id'],self.b['id'],t)
 def wait(self,m,j):
  deadline=time.monotonic()+30
  while time.monotonic()<deadline:
   state=m.get_job(j)
   if state['status'] in ('done','cancelled','error'): return state
   time.sleep(.01)
  self.fail('job timeout')
 def test_import_owned_and_duplicate_binding(self):
  self.p.unlink(); self.assertNotEqual(self.a['id'],self.b['id']); self.assertTrue(Path(self.s.get_media(self.a['id'])['path']).exists())
  a=self.analyze(); b=self.analyze(self.b)
  self.assertEqual(a['cache_key'],b['cache_key']); self.assertNotEqual(a['id'],b['id']); self.assertEqual(b['target_id'],self.b['id'])
  with self.assertRaises(ValueError): JobManager(self.s).start_export(a['id'],self.b['id'],1,'png')
 def test_png_cube_and_project_reload(self):
  r=self.analyze(); m=JobManager(self.s)
  j=m.start_export(r['id'],self.a['id'],.6,'png'); self.assertEqual(self.wait(m,j)['status'],'done')
  np.testing.assert_array_equal(media.decode_image(m.result_file(j)),apply_mapping(media.decode_image(self.p),identity(),.6))
  c=m.start_export(r['id'],self.a['id'],.6,'cube'); self.assertEqual(self.wait(m,c)['status'],'done')
  lines=m.result_file(c).read_text().splitlines(); values=np.array([list(map(float,l.split())) for l in lines if l and l[0].isdigit()])
  self.assertEqual(values.shape,(64**3,3)); np.testing.assert_allclose(values[[0,63,64,-1]],[[0,0,0],[1,0,0],[0,1/63,0],[1,1,1]],atol=1e-6)
  project=self.s.save_project(r['id'],.6,dict(kind='png'))
  reloaded=Store(self.s.root); self.assertEqual(reloaded.get_mapping_record(r['id'])['diagnostics']['parameter_buffer'],[1,None]); self.assertAlmostEqual(json.loads((self.s.root/'projects'/(project['id']+'.json')).read_text())['strength'],.6,places=6)
 def test_corruption_and_invalidation(self):
  r=self.analyze(); (self.s.root/'cache'/(r['cache_key']+'.npz')).write_bytes(b'broken')
  with self.assertRaises((ValueError,OSError)): self.s.get_mapping(r['id'])
  with patch.object(pipeline,'analyze',return_value=identity()) as compute: repaired=self.s.analyze(self.a['id'],self.b['id']); self.assertEqual(compute.call_count,1)
  self.assertEqual(repaired['cache_key'],r['cache_key']); self.assertNotEqual(self.analyze(t=.2)['cache_key'],r['cache_key'])
  with patch.object(pipeline,'algorithm_fingerprint',return_value='new-version'): self.assertNotEqual(self.analyze()['cache_key'],r['cache_key'])
  Image.new('RGB',(13,9),'blue').save(self.p); other=self.s.add_media(self.p); self.assertNotEqual(self.analyze(other)['cache_key'],r['cache_key'])
  meta=self.s.root/'cache'/(r['cache_key']+'.json'); data=json.loads(meta.read_text()); data['version']=-1; meta.write_text(json.dumps(data))
  with patch.object(pipeline,'analyze',return_value=identity()) as compute: self.s.analyze(self.a['id'],self.b['id']); self.assertEqual(compute.call_count,1)
 def test_cancel_during_cpu_and_error(self):
  r=self.analyze(); m=JobManager(self.s); entered=threading.Event(); release=threading.Event(); ended=threading.Event()
  def block(*args):
   entered.set(); release.wait(5); ended.set(); return np.zeros((9,13,3),np.uint8)
  with patch.object(jobs,'apply_mapping',side_effect=block):
   j=m.start_export(r['id'],self.a['id'],1,'png'); self.assertTrue(entered.wait(5)); m.cancel_job(j); release.set(); self.assertTrue(ended.wait(5))
  self.assertEqual(m.get_job(j)['status'],'cancelled')
  with self.assertRaises(ValueError): m.result_file(j)
  with patch.object(jobs,'apply_mapping',side_effect=RuntimeError('controlled failure')):
   e=m.start_export(r['id'],self.a['id'],1,'png'); self.assertEqual(self.wait(m,e)['status'],'error'); self.assertIn('controlled failure',m.get_job(e)['error'])
 def test_actual_mp4_audio_entire_clip(self):
  p=self.root/'video.mkv'
  odd=self.root/'odd.png'; Image.new('RGB',(13,9),'red').save(odd)
  subprocess.run([str(media.ffmpeg_path()),'-hide_banner','-nostdin','-y','-loop','1','-framerate','4','-i',str(odd),'-f','lavfi','-i','sine=frequency=440:duration=2','-t','2','-c:v','ffv1','-c:a','pcm_s16le',str(p)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  source=self.s.add_media(p); self.assertEqual((source['metadata']['width'],source['metadata']['height']),(13,9)); r=self.analyze(source,t=1); m=JobManager(self.s); j=m.start_export(r['id'],source['id'],0,'mp4'); state=self.wait(m,j)
  self.assertEqual(state['status'],'done',state['error']); meta=media.probe_media(m.result_file(j)); self.assertTrue(meta['has_audio']); self.assertGreaterEqual(meta['duration_seconds'],1.9); self.assertIn('bt709',meta['video_stream']); rgb,_=media.decode_frame(m.result_file(j),1.5); self.assertEqual(rgb.shape[:2],(10,14)); self.assertGreater(rgb[:8,:12,0].mean(),235)
 def test_ten_bit_video_export_uses_rgb8_working_pipeline(self):
  p=self.root/'ten-bit.mkv'
  subprocess.run([str(media.ffmpeg_path()),'-hide_banner','-nostdin','-y','-f','lavfi','-i','color=red:size=32x24:rate=4:duration=1','-c:v','ffv1','-pix_fmt','yuv444p10le','-colorspace','bt709',str(p)],check=True,capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  source=self.s.add_media(p); self.assertTrue(source['metadata']['precision_conversion']); r=self.analyze(source); m=JobManager(self.s)
  for strength in (0,.8):
   j=m.start_export(r['id'],source['id'],strength,'mp4'); result=self.wait(m,j)
   self.assertEqual(result['status'],'done',result['error'])
   self.assertTrue(result['settings']['precision_conversion']); self.assertEqual(result['settings']['working_precision_bits'],8)
   rgb,meta=media.decode_frame(m.result_file(j),.5)
   self.assertEqual(meta['component_bit_depths'],[8,8,8]); self.assertEqual(rgb.dtype,np.uint8)
   self.assertGreater(float(rgb[...,0].mean()),235)
 def test_restart_reports_and_interrupted_history(self):
  r=self.analyze(); m=JobManager(self.s); j=m.start_export(r['id'],self.a['id'],0,'png'); self.assertEqual(self.wait(m,j)['status'],'done')
  report=json.loads(m.result_file(j,'report').read_text()); self.assertEqual(report['diagnostics']['parameter_buffer'],[1,None])
  restored=JobManager(Store(self.s.root)); self.assertEqual(restored.get_job(j)['status'],'done'); self.assertTrue(restored.result_file(j).exists())
  report['status']='running'; m.result_file(j,'report').write_text(json.dumps(report)); interrupted=JobManager(self.s); self.assertEqual(interrupted.get_job(j)['status'],'error')
  with self.assertRaises(ValueError): interrupted.result_file(j)
 def test_cancel_interrupts_real_child_and_progress(self):
  r=self.analyze(); original=subprocess.Popen; launched=threading.Event(); process_holder=[]
  def child(*args,**kwargs):
   proc=original([str(media.ffmpeg_path()),'-hide_banner','-nostdin','-re','-f','lavfi','-i','color=red:size=16x16:rate=4:duration=60','-progress','progress.txt','-f','null','-'],**kwargs)
   process_holder.append(proc); launched.set(); return proc
  # Controlled real long-lived FFmpeg replaces only encoding launch; source mapping remains valid.
  video=dict(self.a,kind='video',metadata=dict(duration_seconds=60))
  m=JobManager(self.s)
  with patch.object(self.s,'get_mapping',return_value=identity()), patch.object(self.s,'get_media',return_value=video), patch.object(jobs.subprocess,'Popen',side_effect=child):
   j=m.start_export(r['id'],self.a['id'],1,'mp4'); self.assertTrue(launched.wait(10))
   deadline=time.monotonic()+5
   while time.monotonic()<deadline and m.get_job(j)['progress']==0: time.sleep(.05)
   self.assertGreater(m.get_job(j)['progress'],0); self.assertTrue(m.has_active_jobs()); m.cancel_job(j)
   process_holder[0].wait(timeout=5)
  self.assertEqual(m.get_job(j)['status'],'cancelled'); self.assertFalse(m.has_active_jobs())
  with self.assertRaises(ValueError): m.result_file(j,'report')

 def test_bad_zip_member_recomputes(self):
  import struct, zipfile
  r=self.analyze(); cache=self.s.root/'cache'/(r['cache_key']+'.npz')
  with zipfile.ZipFile(cache,'w',compression=zipfile.ZIP_STORED) as archive:
   import io
   buf=io.BytesIO(); np.save(buf,identity().layers,allow_pickle=False); archive.writestr('layers.npy',buf.getvalue())
  raw=bytearray(cache.read_bytes()); name_len,extra_len=struct.unpack_from('<HH',raw,26); raw[30+name_len+extra_len+200]^=1; cache.write_bytes(raw)
  with self.assertRaises(zipfile.BadZipFile): self.s.get_mapping(r['id'])
  with patch.object(pipeline,'analyze',return_value=identity()) as compute:
   self.s.analyze(self.a['id'],self.b['id']); self.assertEqual(compute.call_count,1)
  self.s.get_mapping(r['id'])
 def test_deflated_npz_corruption_recomputes(self):
  import struct, zipfile, zlib
  r=self.analyze(); cache=self.s.root/'cache'/(r['cache_key']+'.npz')
  with zipfile.ZipFile(cache) as archive:
   info=archive.getinfo('layers.npy'); self.assertEqual(info.compress_type,zipfile.ZIP_DEFLATED)
  raw=bytearray(cache.read_bytes()); name_len,extra_len=struct.unpack_from('<HH',raw,info.header_offset+26)
  payload=info.header_offset+30+name_len+extra_len
  # Damage the real np.savez_compressed DEFLATE stream: reserved BTYPE=3.
  raw[payload]=(raw[payload]&0xf8)|7; cache.write_bytes(raw)
  with self.assertRaises(zlib.error): self.s.get_mapping(r['id'])
  with patch.object(pipeline,'analyze',return_value=identity()) as compute:
   repaired=self.s.analyze(self.a['id'],self.b['id']); self.assertEqual(compute.call_count,1)
  self.assertEqual(repaired['cache_key'],r['cache_key'])
  np.testing.assert_array_equal(self.s.get_mapping(repaired['id']).layers,identity().layers)
