"""Integration checks and reproducible synthetic demo generation."""
import json, sys, time, re, platform
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import grading as g
import server as s

def main():
    checks=[]
    def record(name, passed, **details):
        checks.append({'check':name,'passed':bool(passed),**details})
        print(('PASS ' if passed else 'FAIL ')+name,flush=True)
    rng=np.random.default_rng(7); pixels=rng.random((31,27,3)).astype(np.float32)
    identity=np.array([1]+[0]*8)
    for name,weights,strength in [('identity',identity,1),('zero_weights_identity',[0]*9,1),('zero_strength',np.ones(9),0)]:
        err=float(np.max(np.abs(g.apply_lut(pixels,g.make_lut(weights,strength))-pixels)))
        record(name,err<1e-6,max_error=err)
    rejected=0
    for bad in ([1]*8,[float('nan')]+[0]*8,[-1]+[1]*8):
        try:g.make_lut(bad)
        except ValueError:rejected+=1
    record('invalid_weights_rejected',rejected==3)
    source=g.load_image(s.DEMO/'source.png'); reference=g.load_image(s.DEMO/'reference-v2.png')
    weights,metrics=g.recommend(source,reference)
    record('synthetic_statistical_fit',metrics['rgb_moment_loss_after']<metrics['rgb_moment_loss_before'],**metrics)
    folder=s.OUT/'verification-v2';folder.mkdir(exist_ok=True)
    lut=g.make_lut(weights);(folder/'grade.cube').write_text(g.cube_text(lut),encoding='ascii')
    predicted=g.apply_lut(source,lut);(folder/'python_preview.png').write_bytes(g.png_bytes(predicted))
    ref_png=folder/'ffmpeg_preview.png'
    p=s.run_ff(['-y','-i',str(s.DEMO/'source.png'),'-vf','lut3d=file=grade.cube:interp=trilinear',
                '-frames:v','1','-threads','1',str(ref_png)],cwd=folder,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr.decode('utf8','replace'))
    external=g.load_image(ref_png)
    delta=np.abs(np.round(predicted*255)-np.round(external*255))
    record('cube_order_and_ffmpeg_preview_agree',delta.mean()<=1.1 and delta.max()<=2,
           mean_channel_error_8bit=float(delta.mean()),max_channel_error_8bit=float(delta.max()))
    # Original, reference, and output comparison for the meeting. No photographic claim.
    from PIL import Image, ImageDraw, ImageFont
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(font_path),24) if font_path.exists() else ImageFont.load_default()
    panel=Image.new('RGB',(1440,344),'#171d19');draw=ImageDraw.Draw(panel)
    for i,(name,img) in enumerate([('原始合成色卡',source),('参考图（已知风格）',reference),('统计基线输出',predicted)]):
        thumb=Image.fromarray(np.uint8(np.clip(img,0,1)*255)).resize((480,270))
        panel.paste(thumb,(i*480,45));draw.text((i*480+12,8),name,font=font,fill='#d7edba')
    panel.save(folder/'comparison.png')
    video=s.DEMO/'source.mp4'
    p=s.run_ff(['-y','-loop','1','-framerate','24','-i',str(s.DEMO/'source.png'),
               '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=6',
               '-t','6','-vf',"crop=900:500:x='30+20*sin(t)':y=20,scale=960:540",
               '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
               '-c:a','aac','-b:a','128k','-movflags','+faststart',str(video)],timeout=120)
    if p.returncode:raise RuntimeError(p.stderr.decode('utf8','replace'))
    jid='verified-video-v2';s.JOBS[jid]={'status':'running'}
    # Avoid overwriting a prior test folder on repeat runs.
    if (s.OUT/jid).exists():jid='verified-video-v2-'+str(int(time.time()));s.JOBS[jid]={'status':'running'}
    s.export_worker(jid,video,weights,1.)
    job=s.JOBS[jid];record('video_export_completed',job['status']=='done',elapsed_seconds=job.get('elapsed_seconds'),details=job.get('message'))
    if job['status']!='done':raise RuntimeError(job)
    output=s.OUT/jid/'graded.mp4'
    def frame_info(path):
        p=s.run_ff(['-i',str(path),'-map','0:v:0','-f','framemd5','-'],timeout=90)
        if p.returncode:raise RuntimeError(p.stderr.decode('utf8','replace'))
        lines=[l for l in p.stdout.decode().splitlines() if l and not l.startswith('#')]
        return [','.join(l.split(',')[:5]) for l in lines]
    before,after=frame_info(video),frame_info(output)
    record('video_frame_count_and_timestamps',len(before)==144 and before==after,
           input_frames=len(before),output_frames=len(after),scope='6s 24fps synthetic CFR only')
    p=s.run_ff(['-i',str(output),'-map','0:a:0','-ac','1','-ar','48000','-f','f32le','-'],timeout=60)
    samples=np.frombuffer(p.stdout,dtype=np.float32)
    duration=len(samples)/48000;rms=float(np.sqrt(np.mean(samples**2))) if len(samples) else 0
    record('audio_present_and_duration',p.returncode==0 and abs(duration-6)<.06 and rms>.01,
           decoded_duration_seconds=duration,rms=rms,
           scope='synthetic tone preserved; lip sync and real VFR footage not validated')
    # Verify absent audio is accepted by the optional audio map.
    silent=s.DEMO/'silent.mp4'
    p=s.run_ff(['-y','-i',str(video),'-t','1','-an','-c:v','copy',str(silent)],timeout=30)
    if p.returncode:raise RuntimeError('silent fixture failed')
    sid='silent-'+str(int(time.time()));s.JOBS[sid]={'status':'running'};s.export_worker(sid,silent,identity,1.)
    record('video_without_audio',s.JOBS[sid]['status']=='done')
    report={'palette_version':g.PALETTE_VERSION,'created_local':time.strftime('%Y-%m-%d %H:%M:%S'),'python':sys.version.split()[0],
            'platform':platform.platform(),'ffmpeg':s.run_ff(['-version'],timeout=10).stdout.decode().splitlines()[0],
            'all_passed':all(x['passed'] for x in checks),'checks':checks,
            'verified_video':str(output),'comparison':str(folder/'comparison.png'),
            'limitations':['Synthetic calibration material only; no real footage quality evaluation',
              'No neural training, semantic skin protection, HDR/Log or DaVinci roundtrip tests',
              'Performance is one run on a 960x540 six-second sample, not a 1080p real-time benchmark']}
    (ROOT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['all_passed']:raise SystemExit(1)

if __name__=='__main__':main()
