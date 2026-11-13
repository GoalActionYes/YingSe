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
from spatial_render import SpatialEngine,finish_spatial
from spatial_store import SpatialStore
from spatial_jobs import SpatialJobs
from spatial_build import build_tag


def strength_preview_script(js):
    def once(old,new):
        nonlocal js
        if js.count(old)!=1:raise ValueError('强度预览脚本发生变化，需重新适配')
        js=js.replace(old,new)
    once('const status =',"""const fastStrength=window.SpatialStrengthPreview?new window.SpatialStrengthPreview({
      target:$('after'),
      current:()=>({mapping:state.mapping,generation:state.generation,strength:effectiveStrength(),neutral:controls.count()===0,version:manualVersion,adjustments:controls.snapshot()}),
      onPaint:()=>{$('after-placeholder').hidden=true;$('preview-state').textContent=controls.count()?`微调 ${controls.count()} 项 · 预览中`:`强度 ${Math.round(effectiveStrength()*100)} · 预览中`;}
    }):{clear(){},render(){return false;},needsBasis(){return false;},accept(){},serverApplied(){}};
    const status =""")
    once('++state.generation; ++pickVersion;','fastStrength.clear(); ++state.generation; ++pickVersion;')
    once('function requestPreview() {','function requestPreview() {\n  fastStrength.render();')
    once('function strengthChanged(){','function strengthChanged(){ fastStrength.render();')
    once("window.addEventListener('manual-change',()=>{","window.addEventListener('manual-change',()=>{ fastStrength.render();")
    once('interactive:true});','interactive:true,include_basis:fastStrength.needsBasis(request.mapping,request.generation)});')
    check='      if(request.generation!==state.generation || request.mapping!==state.mapping || request.strength!==effectiveStrength() || request.manualVersion!==manualVersion)continue;'
    once(check,"      if(result.basis && request.generation===state.generation && request.mapping===state.mapping)fastStrength.accept(request.mapping,request.generation,result.before,result.basis);\n"+check)
    once("      image('before',result.before);image('after',result.after);","      fastStrength.serverApplied(); image('before',result.before);image('after',result.after);")
    once('} catch(error) {if(request.generation', '} catch(error) {if(request.generation===state.generation && request.mapping===state.mapping)fastStrength.clear();if(request.generation')
    return js


def assets(enhanced=False):
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
    if enhanced:
        html=html.replace('局部参考调色研究版','自适应参考调色研究版').replace('局部参考调色（课程研究）','自适应参考调色（课程研究）')
        html=html.replace('图片保留原尺寸；视频导出为30fps。局部效果请导出图片或视频。','图片保留原尺寸；视频以30fps导出并平滑色彩过渡，静帧预览可能略有差异。')
        if html.count('<script src="/app.js" defer></script>')!=1:raise ValueError('预览脚本入口发生变化')
        html=html.replace('<script src="/app.js" defer></script>','<script src="/spatial-strength-preview.js" defer></script><script src="/app.js" defer></script>')
        js=strength_preview_script(js)
    result={'/':(html.encode(),'text/html; charset=utf-8'),'/app.js':(js.encode(),'application/javascript; charset=utf-8')}
    if enhanced:
        for name in ['strength-preview','manual-worker','manual-preview']:
            result['/spatial-'+name+'.js']=(Path(__file__).with_name('spatial_'+name.replace('-','_')+'.js').read_bytes(),'application/javascript; charset=utf-8')
    return result


class SpatialHandler(Handler):
    def handle_route(self,method):
        self.guard()
        if method=='GET' and self.path in self.server.assets:
            body,mime=self.server.assets[self.path];return self.send_bytes(body,mime)
        if method=='GET' and self.path=='/api/health':
            return self.json(dict(app='spatial-reference-workbench',version=1,pid=os.getpid(),build=self.server.build_tag,variant=self.server.variant,instance=hashlib.sha256(str(self.server.store.root).encode()).hexdigest(),active_exports=len(self.server.manager.active_jobs())))
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
            data=self.body(('mapping_id','strength'),('adjustments','interactive','include_basis'));interactive=data.get('interactive',False);include_basis=data.get('include_basis',False)
            if type(interactive)!=bool or type(include_basis)!=bool:raise RequestError(400,'交互预览标记无效')
            identifier=self.identifier(data['mapping_id']);strength=self.number(data['strength'],1);store=self.server.store
            record=store.get_mapping_record(identifier);source=store.get_media(record['target_id'])
            def load():
                rgb,meta=store.decode(source,record['analysis_time']);image=Image.fromarray(rgb)
                image.thumbnail((960,640) if interactive else (1440,960),Image.Resampling.BILINEAR)
                return np.array(image),meta
            rgb,meta,before=self.server.preview_frames.get((source['id'],source['sha256'],record['analysis_time'],interactive),load,png)
            if include_basis:
                full=store.render(identifier,rgb,1,None,preview=True,verified_target=source)
                after=finish_spatial(rgb,full,strength,data.get('adjustments'))
            else:after=store.render(identifier,rgb,strength,data.get('adjustments'),preview=True,verified_target=source)
            result=dict(mapping_id=identifier,strength=strength,before=before,after=png(after),metadata=public_metadata(meta),preview_renderer='spatial-cuda')
            if include_basis:result['basis']=dict(version=1,width=rgb.shape[1],height=rgb.shape[0],full=png(full))
            return self.json(result)
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


def make_spatial_workbench(root,port=0,engine=None,enhanced=False):
    if type(enhanced)!=bool:raise ValueError('Expected explicit enhanced mode')
    engine=engine if engine is not None else SpatialEngine('multiref-v1' if enhanced else 'app8')
    server=ThreadingHTTPServer(('127.0.0.1',port),SpatialHandler);server.daemon_threads=True
    server.assets=assets(enhanced);server.store=SpatialStore(root,engine);server.manager=SpatialJobs(server.store,temporal=enhanced)
    server.build_tag=build_tag(enhanced);server.variant='multiref-temporal-v1' if enhanced else 'app8'
    server.preview_frames=PreviewFrames();server.token=secrets.token_urlsafe(32);server.operation_lock=threading.RLock();server.stopping=False
    return server


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path);parser.add_argument('--port',type=int,default=0);parser.add_argument('--multiref',action='store_true')
    args=parser.parse_args();directory=args.data_dir or ROOT/('dev-data/spatial-multiref' if args.multiref else 'dev-data/spatial-reference');server=make_spatial_workbench(directory,args.port,enhanced=args.multiref)
    url=f'http://127.0.0.1:{server.server_address[1]}'
    _json(server.store.root/'session.json',dict(url=url,pid=os.getpid(),token=server.token,app='spatial-reference-workbench'))
    print(url,flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
