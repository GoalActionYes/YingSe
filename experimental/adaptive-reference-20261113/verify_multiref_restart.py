"""Owned new-profile launcher/restart verification; never replace a user project."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote
from urllib.request import Request,urlopen
from launch_spatial_workbench import ENHANCED_DATA,ROOT,probe,pid_alive
sys.path.insert(0,str(ROOT))
from reference_color_app.adjustments import normalize


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-owned',action='store_true');args=parser.parse_args()
    out=ROOT/'outputs/adaptive-reference-20261113/spatial-workbench/restart-multiref-v1';out.mkdir(exist_ok=args.resume_owned)
    if args.resume_owned:
        ownership=json.loads((out/'ownership.json').read_text(encoding='utf-8'))
        if ownership['root']!=str(ENHANCED_DATA.resolve()):raise ValueError('Wrong owned verification root')
    elif (ENHANCED_DATA/'latest-project.json').exists() or probe(ENHANCED_DATA,True):raise RuntimeError('An enhanced project/service already exists; do not replace it for this first-start test')
    launcher=Path(__file__).with_name('launch_spatial_workbench.py');flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    def command(*args):return [sys.executable,'-X','utf8',str(launcher),'--multiref',*args]
    started=subprocess.run(command('--check'),capture_output=True,text=True,encoding='utf-8',timeout=90,creationflags=flags)
    if started.returncode:raise RuntimeError(started.stderr)
    record=probe(ENHANCED_DATA,True)
    if record is None:raise RuntimeError('New service did not identify itself')
    def api(path,data=None):
        request=Request(record['url']+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=90) as response:return json.load(response)
    def upload(path):
        request=Request(record['url']+'/api/media',data=path.read_bytes(),headers={'Content-Type':'application/octet-stream','X-Filename':quote(path.name)})
        with urlopen(request,timeout=90) as response:return json.load(response)
    fixture=ROOT/'outputs/adaptive-reference-20261113/spatial-workbench/verification-multiref-v1'
    if args.resume_owned:
        current=api('/api/projects/latest');mid=ownership['mapping_id']
        if current['mapping_id']!=mid:raise ValueError('Owned test project changed; do not replace it')
    else:
        target,reference=upload(fixture/'target.png'),upload(fixture/'reference.png')
        mapping=api('/api/analyze',dict(target_id=target['id'],reference_id=reference['id']));mid=mapping['mapping_id']
        (out/'ownership.json').write_text(json.dumps(dict(root=str(ENHANCED_DATA.resolve()),mapping_id=mid)),encoding='utf-8')
    adjustments=normalize();adjustments['basic']['brightness']=12
    settings=dict(kind='png',clone_enabled=True,adjustments=adjustments)
    if args.resume_owned and (abs(current['strength']-.47)>1e-7 or current['output_settings']!=settings):raise ValueError('Owned test settings changed; do not overwrite them')
    api('/api/projects',dict(mapping_id=mid,strength=.47,output_settings=settings))
    before=api('/api/preview',dict(mapping_id=mid,strength=.47,adjustments=adjustments,interactive=True))
    first_pid=record['pid'];stopped=subprocess.run(command('--stop'),capture_output=True,text=True,encoding='utf-8',timeout=20,creationflags=flags)
    if stopped.returncode:raise RuntimeError(stopped.stderr)
    deadline=time.monotonic()+10
    while pid_alive(first_pid) and time.monotonic()<deadline:time.sleep(.1)
    if pid_alive(first_pid):raise RuntimeError('Owned first service remains live; do not restart')
    children=[subprocess.Popen(command('--check'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=flags) for _ in range(2)]
    urls=[]
    for child in children:
        output,error=child.communicate(timeout=90)
        if child.returncode:raise RuntimeError(error.decode('utf-8','replace'))
        urls.append(output.decode('utf-8').strip())
    if urls[0]!=urls[1]:raise ValueError('Concurrent launch created distinct services')
    record=probe(ENHANCED_DATA,True)
    if record is None or record['pid']==first_pid:raise ValueError('Restart did not create the correct new service')
    restored=api('/api/projects/latest')
    assert restored['mapping_id']==mid and not restored['needs_analysis'] and abs(restored['strength']-.47)<1e-7 and restored['output_settings']==settings
    after=api('/api/preview',dict(mapping_id=mid,strength=.47,adjustments=adjustments,interactive=True))
    assert after['after']==before['after'] and after['before']==before['before']
    reused=subprocess.run(command('--check'),capture_output=True,text=True,encoding='utf-8',timeout=20,creationflags=flags)
    assert reused.returncode==0 and reused.stdout.strip()==record['url'] and probe(ENHANCED_DATA,True)['pid']==record['pid']
    # Leave a simple, explicitly created demo for the first opening, with no manual edits.
    api('/api/projects',dict(mapping_id=mid,strength=.8,output_settings=dict(kind='png',clone_enabled=True,adjustments=normalize())))
    report=dict(restart_verified=True,concurrent_start_same_url=True,reuse_verified=True,project_and_adjustments_restored=True,
        preview_after_restart_exact=True,variant='multiref-temporal-v1',demo_strength=.8,passed=True,
        scope='Actual owned process restart and HTTP persistence, not visual browser acceptance')
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(report,url=record['url']),indent=2),flush=True)


if __name__=='__main__':main()
