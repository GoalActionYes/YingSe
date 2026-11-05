"""Isolated portable service and authenticated lifecycle controller."""
from pathlib import Path
import hmac, json, os, secrets, subprocess, sys, time, traceback, urllib.request

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / 'data'
READY = STATE / 'session.json'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
os.environ.pop('PYTHONPATH', None)
os.environ.pop('IMAGEIO_FFMPEG_EXE', None)

def request(url, payload=None, token=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
    if token: req.add_header('X-Reference-Token',token)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=3) as response:
        return json.load(response)

def live():
    try:
        from urllib.parse import urlsplit
        value = json.loads(READY.read_text(encoding='utf8'))
        url = value['url']; parsed = urlsplit(url)
        if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.path or parsed.query or parsed.fragment or parsed.username or not parsed.port:
            return None
        health = request(url+'/api/health')
        auth = request(url+'/desktop/health', token=value['token'])
        if health['app'] == 'reference-color-app' and auth == {'token':value['token'],'pid':value['pid']}:
            return value
    except Exception:
        pass
    return None

def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)

def serve():
    import numpy, PIL
    from reference_color_app import server, media
    ff = media.ffmpeg_path().resolve()
    for path in (Path(sys.executable), Path(numpy.__file__), Path(PIL.__file__), ff):
        if not path.resolve().is_relative_to(ROOT):
            raise RuntimeError('Dependency outside portable package: '+str(path))
    subprocess.run([str(ff),'-version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0), timeout=20)
    token = secrets.token_hex(32)
    httpd = server.make_server(STATE/'workspace', port=0, token=token)
    class Handler(server.Handler):
        def do_GET(self):
            if self.path == '/desktop/health':
                try: self.guard()
                except server.RequestError as exc: return self.json({'error':exc.message}, exc.status)
                if not hmac.compare_digest(self.headers.get('X-Reference-Token',''), token):
                    return self.json({'error':'关闭令牌无效'},403)
                return self.json({'token':token,'pid':os.getpid()})
            super().do_GET()
    httpd.RequestHandlerClass = Handler
    value = dict(url=f'http://127.0.0.1:{httpd.server_port}',token=token,pid=os.getpid())
    write(STATE/'runtime-check.json', dict(python=sys.executable,numpy=numpy.__file__,pillow=PIL.__file__,ffmpeg=str(ff),sys_path=sys.path))
    write(READY, value)
    try: httpd.serve_forever()
    finally:
        httpd.server_close()
        try:
            if json.loads(READY.read_text(encoding='utf8')).get('token') == token: READY.unlink()
        except (OSError,ValueError): pass

def main():
    STATE.mkdir(exist_ok=True)
    if '--serve' in sys.argv:
        with (STATE/'server.log').open('a',encoding='utf8',buffering=1) as log:
            sys.stdout = sys.stderr = log
            try: serve()
            except Exception: traceback.print_exc(); return 1
        return 0
    current = live()
    if '--stop' in sys.argv:
        if current:
            request(current['url']+'/api/stop', {'token':current['token']})
            for _ in range(100):
                if live() is None: return 0
                time.sleep(.1)
            raise RuntimeError('服务关闭超时；未强制结束任何进程')
        return 0
    if current is None:
        child = subprocess.Popen([sys.executable,'-I','-B',str(Path(__file__).resolve()),'--serve'],cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(300):
            current = live()
            if current: break
            if child.poll() is not None: raise RuntimeError('服务启动失败，请查看 data/server.log；请确认 FFmpeg 和运行时完整')
            time.sleep(.1)
        if not current: raise RuntimeError('服务启动超时，请查看 data/server.log')
    (STATE/'launch-url.txt').write_text(current['url'],encoding='ascii')
    return 0

if __name__ == '__main__':
    try: sys.exit(main())
    except Exception:
        STATE.mkdir(exist_ok=True)
        (STATE/'launcher-error.txt').write_text(traceback.format_exc(),encoding='utf8')
        sys.exit(1)
