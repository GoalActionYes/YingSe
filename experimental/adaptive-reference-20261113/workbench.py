"""Isolated research UI using the stable app's upload/preview/export workflows."""
import argparse
import hashlib
import os
from pathlib import Path
import secrets
import sys
import threading
from http.server import ThreadingHTTPServer
from canonical_model import ROOT
sys.path.insert(0,str(ROOT))
from reference_color_app.server import Handler,WEB,RequestError,public_metadata,frame,png
from reference_color_app.adjustments import compose
from reference_color_app.lut import apply_mapping
import numpy as np
from PIL import Image
from reference_color_app.jobs import JobManager
from reference_color_app.preview import PreviewFrames
from reference_color_app.storage import _json
from learned_store import LearnedStore
from model_provider import ModelProvider


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('稳定版界面结构已改变，实验入口需重新适配')
    return text.replace(old,new,1)


def static_assets():
    html=(WEB/'index.html').read_text(encoding='utf-8')
    html=replace_once(html,'<title>映色 · 参考图调色</title>','<title>映色 · 自适应调色实验版</title>')
    html=replace_once(html,'</head>','<script src="/experiment.js" defer></script></head>')
    html=replace_once(html,'<div class="strength-row">',
        '<div class="experimental-engine"><label for="engine">调色方法</label><select id="engine">'
        '<option value="baseline">现有算法</option><option value="model" disabled>模型基线 · 实验</option></select>'
        '<small id="engine-state">模型准备中，首次启动可能较慢</small>'
        '<small>可切换比较。模型结果不保证更好；目前不含人物与背景独立调色。</small></div><div class="strength-row">')
    html=replace_once(html,'映色 · 1.3 可撤销调色','映色 · 自适应调色实验版（研究）')
    html=replace_once(html,'本地独立实现 · 算法一致性仍在验证','研究版 · 可比较现有算法与作者预训练模型')
    js=(WEB/'app.js').read_text(encoding='utf-8')
    js=replace_once(js,"time_seconds:Number($('frame-time').value)});",
                       "time_seconds:Number($('frame-time').value),engine:document.getElementById('engine').value});")
    js=replace_once(js,'state.mapping=project.mapping_id;',
                       "state.mapping=project.mapping_id;$('engine').value=project.engine||'baseline';")
    js=replace_once(js,"const ready=!!state.mapping, waiting=state.analyzing || state.uploading>0;",
                       "const ready=!!state.mapping, waiting=state.analyzing || state.uploading>0;$('engine').disabled=waiting;")
    css=(WEB/'styles.css').read_text(encoding='utf-8')+'\n.experimental-engine{display:grid;gap:8px;padding:12px 0;border-bottom:1px solid #444;margin-bottom:12px}.experimental-engine select{width:100%}.experimental-engine small{color:#aaa;line-height:1.5}\n'
    return {'/':(html.encode(),'text/html; charset=utf-8'),
            '/app.js':(js.encode(),'application/javascript; charset=utf-8'),
            '/styles.css':(css.encode(),'text/css; charset=utf-8'),
            '/experiment.js':((Path(__file__).parent/'experiment.js').read_bytes(),'application/javascript; charset=utf-8')}


class ResearchHandler(Handler):
    def json(self,data,status=200):
        if self.command=='GET' and self.path.startswith('/api/projects/') and isinstance(data,dict) and 'mapping_id' in data:
            record=self.server.store.get_mapping_record(data['mapping_id'])
            data=dict(data,engine=record['diagnostics'].get('engine','baseline'))
        return super().json(data,status)

    def handle_route(self,method):
        self.guard()
        if method=='GET' and self.path in self.server.assets:
            body,mime=self.server.assets[self.path]
            return self.send_bytes(body,mime)
        if method=='GET' and self.path=='/api/engine-state':
            return self.json(self.server.provider.status())
        if method=='GET' and self.path=='/api/health':
            return self.json(dict(app='adaptive-reference-workbench',version=1,pid=os.getpid(),
                                  instance=hashlib.sha256(str(self.server.store.root).encode()).hexdigest(),
                                  active_exports=len(self.server.manager.active_jobs())))
        if method=='POST' and self.path=='/api/preview':
            data=self.body(('mapping_id','strength'),('adjustments','interactive'))
            interactive=data.get('interactive',False)
            if type(interactive) is not bool:raise RequestError(400,'交互预览标记无效')
            identifier=self.identifier(data['mapping_id']);strength=self.number(data['strength'],1)
            mapping=self.server.store.get_mapping(identifier);record=self.server.store.get_mapping_record(identifier)
            source=self.server.store.get_media(record['target_id'])
            key=(source['id'],source['sha256'],record['analysis_time'],interactive)
            def load_preview():
                rgb,metadata=frame(source,record['analysis_time'])
                if interactive:
                    image=Image.fromarray(rgb);image.thumbnail((960,640),Image.Resampling.BILINEAR);rgb=np.array(image)
                return rgb,metadata
            rgb,metadata,before=self.server.preview_frames.get(key,load_preview,png)
            effective,effective_strength=compose(mapping,strength,data.get('adjustments'))
            if hasattr(self.server.provider,'preview'):
                after,renderer=self.server.provider.preview(rgb,effective,effective_strength)
            else:after,renderer=apply_mapping(rgb,effective,effective_strength),'cpu'
            return self.json(dict(mapping_id=identifier,strength=strength,before=before,after=png(after),
                                  metadata=public_metadata(metadata),preview_renderer=renderer))
        if method=='POST' and self.path=='/api/analyze':
            data=self.body(('target_id','reference_id'),('time_seconds','engine'))
            engine=data.get('engine','baseline')
            if engine not in ('baseline','model'):raise RequestError(400,'未知调色方法')
            record=self.server.store.analyze_engine(self.identifier(data['target_id']),self.identifier(data['reference_id']),
                                                   self.number(data.get('time_seconds',0)),engine)
            metadata=record['diagnostics']['inputs']['target']['metadata']
            return self.json(dict(mapping_id=record['id'],analysis_time=record['analysis_time'],
                                  metadata=public_metadata(metadata),engine=engine,cache_hit=record.get('cache_hit')))
        return super().handle_route(method)


def make_workbench(data_root,port=0,provider=None,identity_tag=None):
    assets=static_assets()
    provider=provider if provider is not None else ModelProvider()
    server=ThreadingHTTPServer(('127.0.0.1',port),ResearchHandler)
    server.daemon_threads=True
    server.assets=assets;server.provider=provider
    server.store=LearnedStore(data_root,provider,identity_tag=identity_tag)
    server.manager=JobManager(server.store);server.preview_frames=PreviewFrames()
    server.token=secrets.token_urlsafe(32);server.operation_lock=threading.RLock();server.stopping=False
    return server


def main():
    parser=argparse.ArgumentParser(description='映色 · 自适应调色独立实验入口')
    parser.add_argument('--port',type=int,default=0)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'dev-data/adaptive-reference')
    args=parser.parse_args()
    server=make_workbench(args.data_dir,args.port)
    url=f'http://127.0.0.1:{server.server_port}'
    _json(server.store.root/'session.json',dict(url=url,token=server.token,pid=os.getpid()))
    print(url,flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
