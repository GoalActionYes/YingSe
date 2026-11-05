"""Loopback-only HTTP adapter for the independent reference-color workspace."""
import argparse
import base64
import hmac
import io
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit, quote

from PIL import Image
from . import media, pipeline
from .lut import apply_mapping
from .storage import Store, _json
from .jobs import JobManager

MAX_UPLOAD = 512 * 1024 * 1024
MAX_JSON = 65536
WEB = Path(__file__).parent / 'web'


def public_media(record):
    return dict(id=record['id'], name=record['name'], kind=record['kind'], metadata=public_metadata(record['metadata']))


def public_metadata(metadata):
    # Decoder's free-form stderr may contain owned absolute paths. Keep the
    # normal workflow to structured dimensions/times/color support metadata.
    return {key: value for key, value in metadata.items() if key != 'metadata_text'}


def png(rgb):
    image = Image.fromarray(rgb)
    image.thumbnail((1440, 960))  # Only after full-resolution grading.
    out = io.BytesIO()
    image.save(out, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode('ascii')


def frame(record, time_seconds=0.0):
    if record['kind'] == 'image':
        return media.decode_image(record['path']), record['metadata']
    return media.decode_frame(record['path'], time_seconds)


class RequestError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def send_bytes(self, data, mime, status=200, filename=None):
        self.send_headers(len(data), mime, status, filename)
        self.wfile.write(data)

    def send_headers(self, size, mime, status=200, filename=None):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        if filename:
            self.send_header('Content-Disposition', "attachment; filename=download; filename*=UTF-8''" + quote(filename, safe=''))
        self.end_headers()

    def send_file(self, file, mime, filename):
        with file.open('rb') as stream:
            self.send_headers(os.fstat(stream.fileno()).st_size, mime, filename=filename)
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                self.wfile.write(chunk)

    def json(self, data, status=200):
        self.send_bytes(json.dumps(pipeline.json_safe(data), ensure_ascii=False, allow_nan=False).encode('utf-8'), 'application/json; charset=utf-8', status)

    def guard(self):
        port = self.server.server_port
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host', '') not in allowed:
            raise RequestError(403, '仅允许本机访问')
        origin = self.headers.get('Origin')
        if origin and origin not in {'http://' + host for host in allowed}:
            raise RequestError(403, '不允许外部站点访问')
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise RequestError(403, '不允许跨站访问')
        if self.headers.get('Transfer-Encoding'):
            raise RequestError(400, '不支持分块请求')

    def length(self, maximum):
        raw = self.headers.get('Content-Length', '')
        if not raw.isdigit():
            raise RequestError(411, '需要有效 Content-Length')
        size = int(raw)
        if size <= 0 or size > maximum:
            raise RequestError(413, '请求内容为空或超过大小限制')
        return size

    def body(self, required, optional=()):
        if self.headers.get_content_type() != 'application/json':
            raise RequestError(415, '需要 application/json')
        size = self.length(MAX_JSON)
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise RequestError(400, '请求内容不完整')
        try:
            value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        except (ValueError, UnicodeError):
            raise RequestError(400, 'JSON 内容无效')
        if not isinstance(value, dict) or set(required) - value.keys() or value.keys() - set(required) - set(optional):
            raise RequestError(400, '请求字段不匹配')
        return value

    def number(self, value, high=None):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (high is not None and value > high):
            raise RequestError(400, '数值参数无效')
        return value

    def identifier(self, value):
        self.server.store._record('media', value)
        return value

    def handle_route(self, method):
        self.guard()
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            raise RequestError(404, '接口不存在')
        path = parsed.path
        store, manager = self.server.store, self.server.manager
        if method == 'GET':
            static = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'application/javascript; charset=utf-8'), '/styles.css': ('styles.css', 'text/css; charset=utf-8')}
            if path in static:
                name, mime = static[path]
                return self.send_bytes((WEB / name).read_bytes(), mime)
            if path == '/api/health':
                return self.json(dict(app='reference-color-app', version=1, active_exports=len(manager.active_jobs())))
            parts = path.strip('/').split('/')
            if len(parts) == 3 and parts[:2] == ['api', 'media']:
                return self.json(public_media(store.get_media(self.identifier(parts[2]))))
            if len(parts) == 3 and parts[:2] == ['api', 'diagnostics']:
                identifier = self.identifier(parts[2]); store.get_mapping(identifier)
                data = store.get_mapping_record(identifier)['diagnostics']
                return self.send_bytes(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8'), 'application/json', filename='映色-诊断.json')
            if len(parts) in (3, 4) and parts[:2] == ['api', 'jobs']:
                identifier = self.identifier(parts[2])
                if len(parts) == 3:
                    job = manager.get_job(identifier)
                    return self.json(self.job_public(job))
                if parts[3] in ('result', 'report'):
                    file = manager.result_file(identifier, parts[3])
                    job = manager.get_job(identifier)
                    source = store.get_media(job['media_id'])
                    name = Path(source['name']).stem + '-映色.' + ('json' if parts[3] == 'report' else job['kind'])
                    return self.send_file(file, {'png':'image/png','mp4':'video/mp4','cube':'application/octet-stream'}.get(file.suffix[1:], 'application/json'), filename=name)
            if len(parts) == 3 and parts[:2] == ['api', 'projects']:
                project_id = json.loads((store.root / 'latest-project.json').read_text(encoding='utf-8'))['project_id'] if parts[2] == 'latest' else self.identifier(parts[2])
                project = json.loads(store._record('projects', self.identifier(project_id)).read_text(encoding='utf-8'))
                mapping_id = project['mapping']['id']; store.get_mapping(mapping_id)
                record = store.get_mapping_record(mapping_id)
                return self.json(dict(id=project['id'], mapping_id=mapping_id, strength=project['strength'], output_settings=project['output_settings'], analysis_time=record['analysis_time'], target=public_media(store.get_media(record['target_id'])), reference=public_media(store.get_media(record['reference_id']))))
            raise RequestError(404, '接口不存在')
        if path == '/api/media':
            if self.headers.get_content_type() != 'application/octet-stream':
                raise RequestError(415, '请上传原始文件内容')
            size = self.length(MAX_UPLOAD)
            name = unquote(self.headers.get('X-Filename', '素材'), encoding='utf-8', errors='strict')
            name = name.replace('\\', '/').split('/')[-1]
            if not name or len(name) > 255 or any(ord(c) < 32 for c in name):
                raise RequestError(400, '文件名无效')
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=store.root, prefix='upload-', suffix='.tmp', delete=False) as temp:
                    temp_path = Path(temp.name)
                    remaining = size
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk: raise RequestError(400, '上传未完成')
                        temp.write(chunk); remaining -= len(chunk)
                metadata = media.probe_media(temp_path)
                if metadata['kind'] == 'image': media.decode_image(temp_path)
                else: media.decode_frame(temp_path, 0.0)
                record = store.add_media(temp_path, name)
                return self.json(public_media(record), 201)
            finally:
                if temp_path is not None: temp_path.unlink(missing_ok=True)
        if path == '/api/frame':
            data = self.body(('media_id',), ('time_seconds',))
            record = store.get_media(self.identifier(data['media_id']))
            rgb, metadata = frame(record, self.number(data.get('time_seconds', 0)))
            return self.json(dict(image=png(rgb), metadata=public_metadata(metadata)))
        if path == '/api/demo':
            data = self.body((), ('kind',))
            if data.get('kind', 'image') not in ('image', 'video'): raise RequestError(400, '示例类型无效')
            demo = Path(__file__).parent / 'demo'
            video = data.get('kind') == 'video'
            return self.json(dict(target=public_media(store.add_media(demo / ('source-motion.mp4' if video else 'source.png'), '受控示例 · ' + ('运动画面.mp4' if video else '原画.png'))), reference=public_media(store.add_media(demo / 'reference.png', '受控示例 · 暖色参考.png'))))
        if path == '/api/analyze':
            data = self.body(('target_id', 'reference_id'), ('time_seconds',))
            target = self.identifier(data['target_id']); reference = self.identifier(data['reference_id'])
            if store.get_media(reference)['kind'] != 'image': raise RequestError(400, '参考素材必须是图片')
            record = store.analyze(target, reference, self.number(data.get('time_seconds', 0)))
            metadata = record['diagnostics']['inputs']['target']['metadata']
            return self.json(dict(mapping_id=record['id'], analysis_time=record['analysis_time'], metadata=public_metadata(metadata)))
        if path == '/api/preview':
            data = self.body(('mapping_id', 'strength'), ('skin_protection',))
            protection = data.get('skin_protection', False)
            if type(protection) is not bool: raise RequestError(400, '人物肤色保护必须为开关值')
            identifier = self.identifier(data['mapping_id']); strength = self.number(data['strength'], 1)
            mapping = store.get_mapping(identifier); record = store.get_mapping_record(identifier)
            rgb, metadata = frame(store.get_media(record['target_id']), record['analysis_time'])
            after = apply_mapping(rgb, mapping, strength)
            skin_info = dict(enabled=False, skin_detected=None)
            if protection and strength > 0:
                from .skin import protect_image
                after, skin_info = protect_image(rgb, after)
                skin_info = dict(skin_info, enabled=True)
            return self.json(dict(mapping_id=identifier, strength=strength, before=png(rgb), after=png(after), metadata=public_metadata(metadata), skin=skin_info))
        if path == '/api/export':
            data = self.body(('mapping_id', 'media_id', 'strength', 'kind'), ('skin_protection',))
            protection = data.get('skin_protection', False)
            if type(protection) is not bool: raise RequestError(400, '人物肤色保护必须为开关值')
            if not isinstance(data['kind'], str): raise RequestError(400, '导出格式无效')
            with self.server.operation_lock:
                if self.server.stopping: raise RequestError(409, '服务正在关闭')
                identifier = manager.start_export(self.identifier(data['mapping_id']), self.identifier(data['media_id']), self.number(data['strength'], 1), data['kind'], skin_protection=protection)
            return self.json(self.job_public(manager.get_job(identifier)), 202)
        parts = path.strip('/').split('/')
        if len(parts) == 4 and parts[:2] == ['api', 'jobs'] and parts[3] == 'cancel':
            self.body(())
            return self.json(self.job_public(manager.cancel_job(self.identifier(parts[2]))))
        if path == '/api/projects':
            data = self.body(('mapping_id', 'strength', 'output_settings'))
            settings = data['output_settings']
            if (not isinstance(settings, dict) or 'kind' not in settings
                    or set(settings) - {'kind', 'clone_enabled', 'skin_protection'}
                    or settings['kind'] not in ('png', 'mp4', 'cube')
                    or ('clone_enabled' in settings and not isinstance(settings['clone_enabled'], bool))
                    or ('skin_protection' in settings and type(settings['skin_protection']) is not bool)):
                raise RequestError(400, '输出设置无效')
            project = store.save_project(self.identifier(data['mapping_id']), self.number(data['strength'], 1), settings)
            _json(store.root / 'latest-project.json', dict(project_id=project['id']))
            return self.json(dict(project_id=project['id']), 201)
        raise RequestError(404, '接口不存在')

    def job_public(self, job):
        result = {key: job.get(key) for key in ('id', 'status', 'kind', 'mapping_id', 'media_id', 'strength', 'error', 'progress', 'settings', 'skin_protection')}
        if job['status'] == 'done':
            result.update(result_url=f"/api/jobs/{job['id']}/result", report_url=f"/api/jobs/{job['id']}/report")
        return result

    def dispatch(self, method):
        try:
            # Stop is a launcher action protected by a per-run secret.
            if method == 'POST' and self.path == '/api/stop':
                self.guard(); data = self.body(('token',))
                if not isinstance(data['token'], str) or not hmac.compare_digest(data['token'], self.server.token):
                    raise RequestError(403, '关闭令牌无效')
                with self.server.operation_lock:
                    if self.server.manager.has_active_jobs() or self.server.manager.has_pending_cleanup(): raise RequestError(409, '请等待导出完成或取消清理结束')
                    self.server.stopping = True
                self.json(dict(stopping=True))
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self.handle_route(method)
        except RequestError as exc:
            self.close_connection = True; self.json(dict(error=exc.message), exc.status)
        except (FileNotFoundError, KeyError):
            self.close_connection = True; self.json(dict(error='素材、映射或任务不存在，请重新导入或分析'), 404)
        except (ValueError, TypeError, UnicodeError) as exc:
            self.close_connection = True; self.json(dict(error=str(exc)), 400)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:
            self.close_connection = True; self.json(dict(error='本地处理失败，请检查素材并重试'), 500)

    def setup(self):
        super().setup(); self.connection.settimeout(120)

    def do_GET(self): self.dispatch('GET')
    def do_POST(self): self.dispatch('POST')
    def do_OPTIONS(self):
        self.close_connection = True; self.json(dict(error='不允许跨域请求'), 403)

    def do_PUT(self):
        self.close_connection = True; self.json(dict(error='不支持此请求方法'), 405)

    do_DELETE = do_PUT
    do_PATCH = do_PUT


def make_server(data_root: Path, port=0, token=None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.store = Store(data_root)
    server.manager = JobManager(server.store)
    server.token = token or secrets.token_urlsafe(32)
    server.operation_lock = threading.RLock()
    server.stopping = False
    return server


def main():
    parser = argparse.ArgumentParser(description='映色 · 本地参考图调色')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'workspace')
    parser.add_argument('--ready-file', type=Path)
    args = parser.parse_args()
    server = make_server(args.data_dir, args.port)
    url = f'http://127.0.0.1:{server.server_port}'
    if args.ready_file:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        _json(args.ready_file, dict(url=url, token=server.token, pid=os.getpid()))
    print(url, flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()
