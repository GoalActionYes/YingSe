"""Isolated real HTTP verification; never replaces the user's latest project."""
import base64
import io
import json
import sys
import threading
import time
import uuid
from urllib.request import Request,urlopen
import numpy as np
from PIL import Image
from canonical_model import ROOT
from workbench import make_workbench
from independent_coco import digest


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/workbench-native-verification'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    server=make_workbench(out/'data',port=0)
    worker=threading.Thread(target=server.serve_forever,daemon=True); worker.start()
    url=f'http://127.0.0.1:{server.server_port}'
    def raw(path,data=None,headers=None):
        with urlopen(Request(url+path,data=data,headers=headers or {},method='POST' if data is not None else 'GET'),timeout=180) as response:
            return response.read()
    def api(path,data=None):
        return json.loads(raw(path,None if data is None else json.dumps(data).encode(),{'Content-Type':'application/json'}))
    def upload(path):return json.loads(raw('/api/media',path.read_bytes(),{'Content-Type':'application/octet-stream','X-Filename':path.name}))
    def decode(value):return np.array(Image.open(io.BytesIO(base64.b64decode(value.split(',',1)[1]))).convert('RGB'))
    try:
        deadline=time.monotonic()+120
        while api('/api/engine-state')['state']=='loading':
            if time.monotonic()>deadline: raise TimeoutError('Owned test model not ready')
            time.sleep(.1)
        assert api('/api/engine-state')['state']=='ready'
        data=ROOT/'outputs/adaptive-reference-20261113/pst50'
        target=upload(data/'content_709/in1.png'); reference=upload(data/'paired_style/tar1.png')
        started=time.perf_counter()
        mapping=api('/api/analyze',dict(target_id=target['id'],reference_id=reference['id'],engine='model'))
        analysis_seconds=time.perf_counter()-started
        record=server.store.get_mapping_record(mapping['mapping_id'])
        assert record['diagnostics']['analysis_dimensions']==[1920,1080],record['diagnostics']
        again=api('/api/analyze',dict(target_id=target['id'],reference_id=reference['id'],engine='model'))
        assert again['cache_hit'] is True
        full=api('/api/preview',dict(mapping_id=mapping['mapping_id'],strength=1))
        actual=decode(full['after'])
        # The UI full-preview path separately caps frames at1440. Compare the
        # compiled LUT against author interpolation on those exact preview pixels.
        import torch
        before=decode(full['before'])
        with server.provider.lock,torch.inference_mode():
            direct=torch.from_numpy(before.copy()).permute(2,0,1)[None].to(server.provider.model.device,dtype=torch.float32)/255
            for lut in server.provider.model.last_luts:
                direct=server.provider.model.model.TrilinearInterpolation(direct,lut)
            preview_direct=direct[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()
        with Image.open(data/'development-salut/1-Canon_E2E.png') as image: golden=np.array(image.convert('RGB'))
        delta=np.abs(actual.astype(np.float32)-preview_direct.astype(np.float32))
        # A 64-cube is an approximation to two17-cube stages. The previous
        # small-example maximum8 was not a bound for every native photo color.
        # Retain the actual worst pixel and tail fraction rather than claim that bound.
        assert np.quantile(delta,.95)<=2 and delta.mean()<=1,(float(delta.mean()),float(delta.max()))
        preview_seconds=[]
        for strength in (.25,.5,.75):
            started=time.perf_counter(); preview=api('/api/preview',dict(mapping_id=mapping['mapping_id'],strength=strength,interactive=True))
            preview_seconds.append(time.perf_counter()-started)
            assert max(decode(preview['after']).shape[:2])<=960
        job=api('/api/export',dict(mapping_id=mapping['mapping_id'],media_id=target['id'],strength=1,kind='png'))
        deadline=time.monotonic()+120
        while job['status'] not in ('done','error','cancelled'):
            assert time.monotonic()<deadline
            time.sleep(.1); job=api('/api/jobs/'+job['id'])
        assert job['status']=='done',job
        image=raw(job['result_url']); (out/'native-result.png').write_bytes(image)
        exported=np.array(Image.open(io.BytesIO(image)).convert('RGB'))
        with Image.open(data/'content_709/in1.png') as im: native_rgb=np.array(im.convert('RGB'))
        from reference_color_app.lut import apply_mapping
        cpu_export=apply_mapping(native_rgb,server.store.get_mapping(mapping['mapping_id']),1)
        export_error=int(np.abs(exported.astype(int)-cpu_export.astype(int)).max())
        assert export_error<=1
        native_delta=np.abs(exported.astype(np.float32)-golden.astype(np.float32))
        assert np.quantile(native_delta,.95)<=2 and native_delta.mean()<=1,(native_delta.mean(),native_delta.max())
        summary=dict(real_http=True,original_analysis_dimensions=record['diagnostics']['analysis_dimensions'],
            preview_shape=list(actual.shape),native_export_shape=list(exported.shape),first_analysis_seconds=analysis_seconds,cache_hit_on_repeat=again['cache_hit'],
            compiled_vs_native_mae=float(delta.mean()),compiled_vs_native_p95=float(np.quantile(delta,.95)),
            compiled_vs_native_max=float(delta.max()),png_vs_cpu_mapping_max=export_error,
            preview_native_fraction_over5=float((delta>5).mean()),
            png_vs_author_native_mae=float(native_delta.mean()),png_vs_author_native_p95=float(np.quantile(native_delta,.95)),
            png_vs_author_native_max=float(native_delta.max()),
            png_native_fraction_over5=float((native_delta>5).mean()),
            interactive_preview_seconds=preview_seconds,
            source_sha256={name:digest(ROOT/'experimental/adaptive-reference-20261113'/name) for name in ('learned_store.py','model_provider.py','canonical_model.py','verify_native_model_input.py')},
            scope='One 1080p natural pair, actual HTTP analysis/cache/GPU preview/PNG export; not overall style quality')
        (out/'native-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(dict(output=str(out),native=summary)),flush=True)
        import verify_workbench
        old_args=sys.argv
        try:
            sys.argv=['verify_workbench.py','--url',url,'--out',str(out/'standard')]
            verify_workbench.main()
        finally:sys.argv=old_args
        standard=json.loads((out/'standard/summary.json').read_text(encoding='utf-8'))
        (out/'results.json').write_text(json.dumps(dict(native=summary,standard=standard),indent=2),encoding='utf-8')
        print(json.dumps(dict(finished=True,output=str(out))),flush=True)
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=5)


if __name__=='__main__': main()
