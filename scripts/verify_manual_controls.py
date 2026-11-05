"""Reproducible synthetic PNG/CUBE/video integration check (no private media)."""
from pathlib import Path
import argparse
import copy
import json
import subprocess
import sys
import time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference_color_app import media
from reference_color_app.storage import Store
from reference_color_app.jobs import JobManager
from reference_color_app.contracts import Mapping
from reference_color_app.lut import apply_mapping
from reference_color_app.adjustments import compose, transform


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    y,x=np.mgrid[:96,:128]
    rgb=np.stack([20+x*1.5,20+y*2,30+(x+y)*.8],axis=-1).clip(0,255).astype(np.uint8)
    source=out/'source.png';Image.fromarray(rgb).save(source)
    video=out/'source.mp4'
    subprocess.run([str(media.ffmpeg_path()),'-v','error','-y','-loop','1','-i',str(source),'-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=2',
        '-t','2','-r','24','-c:v','libx264','-crf','12','-pix_fmt','yuv420p','-c:a','aac',str(video)],check=True,capture_output=True)
    store=Store(out/'workspace');manager=JobManager(store)
    image=store.add_media(source);clip=store.add_media(video);reference=store.add_media(source)
    settings={'basic':{'temperature':25,'brightness':12,'contrast':10},'hsl':{'blue':[10,-20,5]},'curves':{'rgb':[[0,0],[.5,.55],[1,1]]}}
    frozen=copy.deepcopy(settings);results={}
    for target,kinds in [(image,('png','cube')),(clip,('mp4',))]:
        record=store.analyze(target['id'],reference['id']);mapping=store.get_mapping(record['id'])
        effective,strength=compose(mapping,0,frozen)
        for kind in kinds:
            state=copy.deepcopy(frozen);identifier=manager.start_export(record['id'],target['id'],0,kind,state)
            state['basic']['brightness']=-100
            for _ in range(600):
                job=manager.get_job(identifier)
                if job['status'] not in ('queued','running'):break
                time.sleep(.05)
            assert job['status']=='done',job
            assert job['adjustments']['basic']['brightness']==12,'Export settings not frozen'
            path=manager.result_file(identifier);destination=out/('result.'+kind);destination.write_bytes(path.read_bytes())
            if kind=='png':
                expected=apply_mapping(rgb,effective,strength);actual=np.array(Image.open(path))
                assert np.array_equal(expected,actual)
                results['png_max_error']=int(np.abs(actual.astype(int)-expected).max())
            elif kind=='cube':
                rows=np.loadtxt(path,skiprows=4,dtype=np.float32)
                grid=rows.reshape(64,64,64,3).transpose(2,1,0,3)
                parsed=Mapping(np.stack([grid,grid]),{})
                error=np.abs(apply_mapping(rgb,parsed,1).astype(int)-expected.astype(int))
                assert error.max()<=1;results['cube_max_error']=int(error.max())
            else:
                actual,metadata=media.decode_frame(path,0)
                input_rgb,_=media.decode_frame(video,0)
                expected_video=apply_mapping(input_rgb,effective,strength)
                error=np.abs(actual.astype(float)-expected_video)
                assert error.mean()<3.5 and np.percentile(error,95)<10,(error.mean(),np.percentile(error,95))
                subprocess.run([str(media.ffmpeg_path()),'-v','error','-i',str(path),'-f','null','-'],check=True,capture_output=True)
                probe=subprocess.run([str(media.ffmpeg_path()),'-hide_banner','-i',str(path),'-map','0:v:0','-f','null','-'],capture_output=True)
                diagnostic=probe.stderr.decode('utf-8','replace')
                assert 'Audio:' in diagnostic and '24 fps' in diagnostic and 'frame=   48' in diagnostic,diagnostic
                assert abs(metadata['duration_seconds']-2)<.1,metadata
                results['video']={'frames':48,'fps':24,'audio':True,'mae':float(error.mean()),'p95':float(np.percentile(error,95)),'duration_seconds':metadata['duration_seconds']}
    values=np.linspace(0,1,256,dtype=np.float32);gray=np.stack([values]*3,axis=-1)[None]
    plus=transform(gray,{'basic':{'shadows':30}});minus=transform(gray,{'basic':{'highlights':-30}})
    assert (plus-gray)[0,30].mean()>(plus-gray)[0,220].mean()
    assert (gray-minus)[0,220].mean()>(gray-minus)[0,30].mean()
    neutral,strength=compose(mapping,0,None)
    assert neutral is mapping and strength==0
    np.testing.assert_array_equal(apply_mapping(rgb,neutral,strength),rgb)
    results.update(export_settings_frozen=True,neutral_compatibility=True,synthetic_only=True)
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
