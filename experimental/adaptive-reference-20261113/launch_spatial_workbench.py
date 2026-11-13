"""Reuse only an identified owned spatial service; never kill by process name."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import webbrowser
from launch_workbench import startup_lock,saved_session,pid_alive
from spatial_build import build_tag

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'dev-data/spatial-reference'
ENHANCED_DATA=ROOT/'dev-data/spatial-multiref'


def stop_owned(record):
    req=Request(record['url']+'/api/stop',data=json.dumps(dict(token=record['token'])).encode(),headers={'Content-Type':'application/json'})
    try:
        with urlopen(req,timeout=5) as response:response.read()
    except HTTPError as exc:
        try:message=json.loads(exc.read()).get('error','服务暂时无法关闭')
        except (ValueError,OSError):message='服务暂时无法关闭'
        raise RuntimeError(message) from exc


def probe(data=None,enhanced=False):
    data=DATA if data is None else data;record=saved_session(data)
    if record:
        try:
            with urlopen(record['url']+'/api/health',timeout=1) as response:health=json.load(response)
            variant='multiref-temporal-v1' if enhanced else 'app8'
            if health.get('app')=='spatial-reference-workbench' and health.get('pid')==record['pid'] and health.get('instance')==hashlib.sha256(str(data.resolve()).encode()).hexdigest() and health.get('variant','app8')==variant:return dict(record,build=health.get('build'))
        except (OSError,ValueError):pass
    return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--stop',action='store_true');parser.add_argument('--multiref',action='store_true');args=parser.parse_args()
    data=ENHANCED_DATA if args.multiref else DATA
    with startup_lock(data):
        record=probe(data,args.multiref);pending=data/'starting.json'
        if args.stop:
            if record:
                stop_owned(record)
            print('局部研究服务已停止' if record else '没有可验证的局部研究服务');return
        if record and record['build']!=build_tag(args.multiref):
            stop_owned(record)
            deadline=time.monotonic()+10
            while pid_alive(record['pid']) and time.monotonic()<deadline:time.sleep(.1)
            if pid_alive(record['pid']):raise RuntimeError('旧版服务正在结束，请稍后重试')
            record=None
        previous=saved_session(data)
        if record is None:
            if previous is None or not pid_alive(previous['pid']):
                try:previous=json.loads(pending.read_text(encoding='utf-8'))
                except (OSError,ValueError):previous=None
            if previous and type(previous.get('pid')) is int and pid_alive(previous['pid']):
                deadline=time.monotonic()+30
                while time.monotonic()<deadline and pid_alive(previous['pid']):
                    record=probe(data,args.multiref)
                    if record:break
                    time.sleep(.2)
                if record is None and pid_alive(previous['pid']):raise RuntimeError('已记录的研究进程仍在运行；服务未通过验证，不会重复启动')
        if record is None:
            with (data/'workbench.log').open('ab') as log:
                command=[sys.executable,str(Path(__file__).with_name('spatial_workbench.py')),'--data-dir',str(data)]+(['--multiref'] if args.multiref else [])
                child=subprocess.Popen(command,cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUTF8='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            temporary=pending.with_suffix('.tmp');temporary.write_text(json.dumps(dict(pid=child.pid)),encoding='utf-8');os.replace(temporary,pending);deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                record=probe(data,args.multiref)
                if record:break
                if child.poll() is not None:raise RuntimeError(f'研究版启动失败，详见{data}/workbench.log')
                time.sleep(.2)
            if record is None:raise RuntimeError('服务仍在启动，不重复启动；请稍后重试')
        pending.unlink(missing_ok=True);print(record['url'])
        if not args.check:webbrowser.open(record['url'])


if __name__=='__main__':
    try:main()
    except Exception as exc:print(str(exc),file=sys.stderr);raise SystemExit(1)
