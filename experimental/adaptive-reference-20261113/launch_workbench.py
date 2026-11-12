"""Windows research launcher: reuse verified service, log failures, never kill by name."""
import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import Request,urlopen
import webbrowser

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'dev-data/adaptive-reference'


@contextmanager
def startup_lock(root,timeout=45):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.startup.lock').open('a+b') as stream:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        acquired=False;deadline=time.monotonic()+timeout
        try:
            while not acquired:
                try:
                    stream.seek(0)
                    if os.name=='nt':
                        import msvcrt
                        msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    acquired=True
                except OSError:
                    if time.monotonic()>=deadline:raise TimeoutError('另一启动操作正在进行，请稍后重试')
                    time.sleep(.05)
            yield
        finally:
            if acquired:
                stream.seek(0)
                if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def saved_session(root):
    try:
        value=json.loads((Path(root)/'session.json').read_text(encoding='utf-8'))
        if (isinstance(value,dict) and isinstance(value.get('url'),str) and
                re.fullmatch(r'http://127\.0\.0\.1:[0-9]{1,5}',value['url']) and
                type(value.get('pid')) is int and value['pid']>0 and isinstance(value.get('token'),str)):
            return value
    except (OSError,ValueError):pass
    return None


def probe_session(root):
    record=saved_session(root)
    if record is None:return None
    try:
        with urlopen(record['url']+'/api/health',timeout=1) as response:health=json.loads(response.read())
        if (isinstance(health,dict) and health.get('app')=='adaptive-reference-workbench' and
                health.get('pid')==record['pid'] and
                health.get('instance')==hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()):
            return record
    except (OSError,ValueError):pass
    return None


def pid_alive(pid):
    if os.name!='nt':
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return ctypes.get_last_error()==5
    try:
        code=wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
    finally:kernel.CloseHandle(handle)


def _run(args):
    starting=DATA/'starting.json'
    record=probe_session(DATA)
    if args.stop:
        if record:
            request=Request(record['url']+'/api/stop',data=json.dumps(dict(token=record['token'])).encode(),headers={'Content-Type':'application/json'})
            with urlopen(request,timeout=5) as response:response.read()
        print('研究服务已停止' if record else '没有可验证的研究服务');return
    if record is None:
        # A timeout is not proof of process exit. Give a live recorded process
        # time to respond instead of launching duplicates or killing it.
        prior=saved_session(DATA)
        if prior is None or not pid_alive(prior['pid']):
            try:
                pending=json.loads(starting.read_text(encoding='utf-8'))
                if isinstance(pending,dict) and type(pending.get('pid')) is int and pending['pid']>0:prior=pending
            except (OSError,ValueError):pass
        if prior and pid_alive(prior['pid']):
            deadline=time.monotonic()+30
            while time.monotonic()<deadline and pid_alive(prior['pid']):
                record=probe_session(DATA)
                if record:break
                time.sleep(.2)
            if record is None and pid_alive(prior['pid']):
                raise RuntimeError('已有记录的进程仍在运行，但服务暂未通过验证，请稍后重试')
        if record is None:
            with (DATA/'workbench.log').open('ab') as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('workbench.py')),'--data-dir',str(DATA)],
                       cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                       env=dict(os.environ,PYTHONUTF8='1'),
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            temporary=starting.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(pid=child.pid)),encoding='utf-8');os.replace(temporary,starting)
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                record=probe_session(DATA)
                if record:break
                if child.poll() is not None:raise RuntimeError('研究服务启动失败，详情见 dev-data/adaptive-reference/workbench.log')
                time.sleep(.2)
            if record is None:raise RuntimeError('服务仍在启动，请稍后重试；不会重复启动进程')
    starting.unlink(missing_ok=True)
    print(record['url'])
    if not args.check:webbrowser.open(record['url'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--stop',action='store_true')
    args=parser.parse_args()
    with startup_lock(DATA):_run(args)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(str(exc),file=sys.stderr)
        raise SystemExit(1)
