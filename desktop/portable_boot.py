"""Entry point for the bundled Windows runtime. No global Python installation needed."""
from pathlib import Path
import os, sys, json, secrets, threading, time, traceback

ROOT = Path(__file__).resolve().parent.parent
DESKTOP = ROOT/'desktop'
log = (DESKTOP/'server.log').open('a', encoding='utf8', buffering=1)
sys.stdout = sys.stderr = log
try:
    sys.path.insert(0, str(ROOT/'prototype'))
    os.environ.pop('IMAGEIO_FFMPEG_EXE', None)
    os.environ.pop('GRADE_PORT', None)
    import server
    import numpy, PIL
    ff = Path(server.ffmpeg()).resolve()
    if not ff.is_relative_to(ROOT):
        raise RuntimeError('FFmpeg must come from the portable package')
    result = server.run_ff(['-version'], timeout=30)
    if result.returncode:
        raise RuntimeError('Bundled FFmpeg failed to run')
    token = secrets.token_hex(24)

    class Handler(server.Handler):
        def do_GET(self):
            if self.path == '/desktop/health':
                return self.send(token.encode('ascii'), ctype='text/plain')
            return super().do_GET()

        def do_POST(self):
            if self.path == '/desktop/stop':
                if self.headers.get('X-Yingse-Token') != token:
                    return self.send({'error':'Unauthorized'},403)
                with server.LOCK:
                    if any(j['status']=='running' for j in server.JOBS.values()):
                        return self.send({'error':'Export still running'},409)
                self.send({'ok':True})
                threading.Thread(target=httpd.shutdown, daemon=True).start()
                return
            return super().do_POST()

    httpd = server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.PORT = httpd.server_address[1]
    url = f'http://127.0.0.1:{server.PORT}'
    audit = {
        'created_local':time.strftime('%Y-%m-%d %H:%M:%S'), 'url':url,
        'python':sys.executable, 'numpy':numpy.__file__, 'pillow':PIL.__file__,
        'sys_path':sys.path, 'ffmpeg':str(ff), 'version':'0.2.2-portable', 'pid':os.getpid()
    }
    (DESKTOP/'runtime-check.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    ready = DESKTOP/'session.txt'
    temporary = DESKTOP/'session.tmp'
    temporary.write_text(url+'\n'+token+'\n'+str(os.getpid())+'\n',encoding='ascii')
    temporary.replace(ready)
    print('Ready:',url,flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        if ready.exists() and token in ready.read_text(encoding='ascii'):
            ready.unlink()
except Exception:
    traceback.print_exc()
    sys.exit(1)
