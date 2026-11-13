"""Bounded isolated author-program trial, using one consumed development pair."""
import ctypes
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from PIL import Image
import numpy as np
from independent_coco import ROOT,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quiet-log',action='store_true');args=parser.parse_args()
    case_name='dev4-default-quiet-v1' if args.quiet_log else 'dev4-default-v1'
    out=ROOT/'outputs/adaptive-reference-20261113/nct-native'/case_name;out.mkdir(parents=True,exist_ok=False)
    native=ROOT/'.tools/NCT-native';manifest=json.loads((native/'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        if digest(native/entry['path'])!=entry['sha256']:raise ValueError('Native asset changed')
    model=native/'model/vgg19/VGG_ILSVRC_19_layers.caffemodel'
    if model.stat().st_size!=574671192 or digest(model)!='31b4c627c40c6ee151325f079b7ac557fddbd1f79af932888796127a4f4f6954':
        raise ValueError('VGG19 asset differs')
    proto=ROOT/'.tools/NCT-source/demo/model/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt'
    shutil.copyfile(proto,model.with_name(proto.name))
    case=native/'runs'/case_name;(case/'inputs').mkdir(parents=True,exist_ok=False);(case/'results').mkdir()
    data=ROOT/'outputs/adaptive-reference-20261113/pst50'
    inputs={}
    for name,path in [('target',data/'content_709/in4.png'),('reference',data/'paired_style/tar4.png')]:
        with Image.open(path) as image:
            image=image.convert('RGB');image.thumbnail((640,640),Image.Resampling.BILINEAR);image.save(case/f'inputs/{name}.png')
        inputs[name]=dict(source_sha256=digest(path),working_sha256=digest(case/f'inputs/{name}.png'))
    (case/'inputs/pairs.txt').write_text('target.png reference.png 2.0\n',encoding='ascii')
    arguments=['exe/neural_color_transfer.exe','-m','model','-i',f'runs/{case_name}/inputs','-o',f'runs/{case_name}/results',
        '-g','0','-eps','0.6','-nl','2.0','-l','0.125','-w','0.024']
    if (native/arguments[arguments.index('-i')+1]).resolve()!=(case/'inputs').resolve():
        raise ValueError('Native CLI input path does not match the prepared case')
    if (native/arguments[arguments.index('-o')+1]).resolve()!=(case/'results').resolve():
        raise ValueError('Native CLI output path does not match the prepared case')
    report=dict(scope='One known development pair, author binary compatibility/quality/time, not product acceptance',
        binary_manifest_sha256=digest(native/'manifest.json'),vgg_sha256=digest(model),prototxt_sha256=digest(proto),
        script_sha256=digest(Path(__file__)),protocol_sha256=digest(Path(__file__).with_name('nct-native-protocol.md')),
        arguments=arguments,inputs=inputs,timeout_seconds=180,status='starting',quiet_log=args.quiet_log)
    (out/'check_nct_native.py').write_bytes(Path(__file__).read_bytes())
    (out/'nct-native-protocol.md').write_bytes(Path(__file__).with_name('nct-native-protocol.md').read_bytes())
    def save(): (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save();ctypes.windll.kernel32.SetErrorMode(0x8003)
    started=time.perf_counter()
    environment=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if args.quiet_log:environment.update(GLOG_minloglevel='2',GLOG_logtostderr='1',GLOG_alsologtostderr='1')
    with (out/'native.log').open('wb') as log:
        process=subprocess.Popen([str(native/arguments[0]),*arguments[1:]],cwd=native,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=environment,creationflags=subprocess.CREATE_NO_WINDOW)
        report.update(pid=process.pid,status='running');save();print(json.dumps(dict(pid=process.pid,output=str(out))),flush=True)
        while process.poll() is None and time.perf_counter()-started<180:time.sleep(.2)
        timed_out=process.poll() is None
        if timed_out:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
    report.update(seconds=time.perf_counter()-started,returncode=process.returncode,returncode_hex=hex(process.returncode&0xffffffff),
        timed_out=timed_out,status='terminated_timeout' if timed_out else 'exited',log_sha256=digest(out/'native.log'))
    expected=case/'results/target_reference_2.00.png'
    report['output_produced']=expected.is_file()
    if expected.is_file():
        shutil.copyfile(expected,out/'result.png');report['output_sha256']=digest(out/'result.png');save()
        # Author result is saved before any desired-image read in this invocation.
        with Image.open(out/'result.png') as image: actual=np.array(image.convert('RGB'))
        with Image.open(data/'paired_gt/gt4.png') as image:
            image=image.convert('RGB');image.thumbnail((640,640),Image.Resampling.BILINEAR);truth=np.array(image)
        if actual.shape!=truth.shape:report['quality_failure']='Output/desired shape mismatch'
        else: report['rgb_mae']=float(np.abs(actual.astype(np.float32)-truth.astype(np.float32)).mean())
    save()
    # JSON avoids the system console codepage changing or losing captured native bytes.
    tail=(out/'native.log').read_bytes()[-5000:].decode('utf-8',errors='replace')
    print(json.dumps(dict(result=report,log_tail=tail)),flush=True)


if __name__=='__main__':main()
