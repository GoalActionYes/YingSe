"""Independent loopback spatial-grading app, preserving the stable editor UI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import zipfile
import zlib
from http.server import ThreadingHTTPServer
import numpy as np
from PIL import Image
from canonical_model import ROOT
sys.path.insert(0,str(ROOT))
from reference_color_app.server import Handler,WEB,RequestError,png,public_metadata,public_media
from reference_color_app.preview import PreviewFrames
from reference_color_app.storage import _json
from reference_color_app.adjustments import normalize,transform
from spatial_render import SpatialEngine
from spatial_store import SpatialStore
from spatial_jobs import SpatialJobs
from spatial_build import build_tag


def assets():
    html=(WEB/'index.html').read_text(encoding='utf-8')
    html=html.replace('<title>映色 · 参考图调色</title>','<title>映色 · 局部参考调色研究版</title>')
    html=html.replace('<option value="cube">.cube · 颜色映射</option>','')
    if '<option value="cube">' in html:raise ValueError('界面导出选项发生变化，需重新适配')
    html=html.replace('映色 · 1.3 可撤销调色','映色 · 局部参考调色（课程研究）')
    html=html.replace('本地独立实现 · 算法一致性仍在验证','本地课程研究 · 空间自适应调色')
    html=html.replace('图片保留原分辨率；.cube 包含当前强度。','图片保留原尺寸；视频导出为30fps。局部效果请导出图片或视频。')
    html=html.replace('分析帧','预览帧')
    js=(WEB/'app.js').read_text(encoding='utf-8').replace('图片、视频和 CUBE 导出','图片和视频导出').replace('导出使用相同映射与当前强度','图片导出包含当前局部效果；视频逐帧自适应处理')
    js=js.replace('原尺寸视频（奇数边补齐偶数）；H.264 / AAC 有损编码，保留首条可用音轨。','局部调色视频 · 统一30fps（可补帧／丢帧）；原尺寸，奇数边补齐偶数；H.264 / AAC 有损编码，保留首条可用音轨。')
    restore_line="$('diagnostics').href='/api/diagnostics/'+state.mapping;$('diagnostics').hidden=false;setButtons();requestPreview();"
    if js.count(restore_line)!=1:raise ValueError('工程恢复脚本发生变化，需重新适配')
    js=js.replace(restore_line,"""if(project.needs_analysis){
      $('diagnostics').hidden=true;setButtons();
      await Promise.all([showFrame('target'),showFrame('reference')]);
      if(gen!==state.generation)return;
      status('素材和微调已恢复，正在重新分析更新后的调色效果。','busy');
      await analyze();return;
    }
    """+restore_line)
    return {'/':(html.encode(),'text/html; charset=utf-8'),'/app.js':(js.encode(),'application/javascript; charset=utf-8')}


class SpatialHandler(Handler):
    def handle_route(self,method):
        self.guard()
        if method=='GET' and self.path in self.server.assets:
            body,mime=self.server.assets[self.path];return self.send_bytes(body,mime)
        if method=='GET' and self.path=='/api/health':
            return self.json(dict(app='spatial-reference-workbench',version=1,pid=os.getpid(),build= self.server.build_tag,instance=hashlib.sha256(str(self.server.store.root).encode()).hexdigest(),active_exports=len(self.server.manager.active_jobs())))
        parts=self.path.strip('/').split('/')
        if method=='GET' and len(parts)==3 and parts[:2]==['api','projects']:
            store=self.server.store
            identifier=json.loads((store.root/'latest-project.json').read_text(encoding='utf-8'))['project_id'] if parts[2]=='latest' else parts[2]
            project=json.loads(store._record('projects',self.identifier(identifier)).read_text(encoding='utf-8'))
            record=project['mapping'];target=store.get_media(record['target_id']);reference=store.get_media(record['reference_id'])
            # Owned-source integrity failures must not be masked as an outdated model.
            needs_analysis=False
            try:store.get_prepared(record['id'])
            except (ValueError,OSError,KeyError,EOFError,zipfile.BadZipFile,zlib.error):needs_analysis=True
            return self.json(dict(id=project['id'],mapping_id=None if needs_analysis else record['id'],needs_analysis=needs_analysis,
                strength=project['strength'],output_settings=project['output_settings'],analysis_time=record['analysis_time'],
                target=public_media(target),reference=public_media(reference)))
        if method=='POST' and self.path=='/api/frame':
            data=self.body(('media_id',),('time_seconds',));source=self.server.store.get_media(self.identifier(data['media_id']))
            rgb,meta=self.server.store.decode(source,self.number(data.get('time_seconds',0)))
            return self.json(dict(image=png(rgb),metadata=public_metadata(meta)))
        if method=='POST' and self.path=='/api/preview':
            data=self.body(('mapping_id','strength'),('adjustments','interactive'));interactive=data.get('interactive',False)
            if type(interactive)!=bool:raise RequestError(400,'交互预览标记无效')
            identifier=self.identifier(data['mapping_id']);strength=self.number(data['strength'],1);store=self.server.store
            record=store.get_mapping_record(identifier);source=store.get_media(record['target_id'])
            def load():
                rgb,meta=store.decode(source,record['analysis_time']);image=Image.fromarray(rgb)
                image.thumbnail((960,640) if interactive else (1440,960),Image.Resampling.BILINEAR)
                return np.array(image),meta
            rgb,meta,before=self.server.preview_frames.get((source['id'],source['sha256'],record['analysis_time'],interactive),load,png)
            after=store.render(identifier,rgb,strength,data.get('adjustments'),preview=True,verified_target=source)
            return self.json(dict(mapping_id=identifier,strength=strength,before=before,after=png(after),metadata=public_metadata(meta),preview_renderer='spatial-cuda'))
        if method=='POST' and self.path=='/api/curve-sample':
            data=self.body(('mapping_id','strength','x','y'),('adjustments',));store=self.server.store
            identifier=self.identifier(data['mapping_id']);record=store.get_mapping_record(identifier);source=store.get_media(record['target_id'])
            x,y=self.number(data['x'],1),self.number(data['y'],1);rgb,_=store.decode(source,record['analysis_time'])
            settings=normalize(data.get('adjustments'));settings['curves']['enabled']=False
            graded=store.render(identifier,rgb,self.number(data['strength'],1),settings,preview=False,verified_target=source)
            px=min(rgb.shape[1]-1,int(x*rgb.shape[1]));py=min(rgb.shape[0]-1,int(y*rgb.shape[0]));value=graded[py,px].astype(float)/255
            curves=normalize(data.get('adjustments'))['curves'];points=np.asarray(curves['rgb'])
            channels=np.interp(value,points[:,0],points[:,1]) if curves['enabled'] else value
            return self.json(dict(rgb=value.tolist(),channels=channels.tolist(),pixel=[px,py]))
        return super().handle_route(method)


def make_spatial_workbench(root,port=0,engine=None):
    engine=engine if engine is not None else SpatialEngine()
    server=ThreadingHTTPServer(('127.0.0.1',port),SpatialHandler);server.daemon_threads=True
    server.assets=assets();server.store=SpatialStore(root,engine);server.manager=SpatialJobs(server.store)
    server.build_tag=build_tag()
    server.preview_frames=PreviewFrames();server.token=secrets.token_urlsafe(32);server.operation_lock=threading.RLock();server.stopping=False
    return server


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=ROOT/'dev-data/spatial-reference');parser.add_argument('--port',type=int,default=0)
    args=parser.parse_args();server=make_spatial_workbench(args.data_dir,args.port)
    url=f'http://127.0.0.1:{server.server_address[1]}'
    _json(server.store.root/'session.json',dict(url=url,pid=os.getpid(),token=server.token,app='spatial-reference-workbench'))
    print(url,flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
