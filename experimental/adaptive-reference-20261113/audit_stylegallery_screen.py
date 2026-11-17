"""Recompute every completed validated-screen image and the necessary stop gate."""
import json,ast,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from screen_stylegallery import ROOT,BASE,digest


def main():
    load=lambda p:json.loads(Path(p).read_text());out=BASE/'stylegallery-screen-validated-v1';r=load(out/'results.json');data=BASE/'cohort-training-coverage-v2'
    assert r['strict_deterministic'] and r['activation_checkpoint'] and r['trim_after_init'] and r['computational_child_ast_exact']
    assert digest(data/'cohort.json')=='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077' and digest(data/'srgb-v1/manifest.json')=='da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f'
    rows=load(data/'cohort.json')['groups']['validation'][:8];lookup={e['original_path']:e for e in load(data/'srgb-v1/manifest.json')['files']};byindex={e['index']:e for e in rows}
    assert [e['index'] for e in r['completed']]==[e['index'] for e in rows[:len(r['completed'])]]
    for n,h in r['source_sha256'].items():assert digest(out/'frozen-source'/n)==digest(Path(__file__).with_name(n))==h
    proof=BASE/'stylegallery-screen-cachetrim-checkpoint-v1';p=load(proof/'results.json');assert p['controlled_bridge']['passed'] and digest(proof/'results.json')==r['validated_execution_results_sha256']
    def body(path):return [ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in ['child','checkpoint_key_map']]
    assert body(out/'frozen-source/screen_stylegallery.py')==body(proof/'frozen-source/screen_stylegallery.py')
    baseline=load(BASE/'training-coverage-expanded-v2/results.json')['baseline'];old={e['index']:e for e in baseline['cases']};measurements=[];panels=[];input_arrays={};inputs_verified=0
    def read(relative):
        e=lookup[relative];path=data/e['prepared_path'];assert path.resolve().is_relative_to(BASE.resolve()) and digest(path)==e['sha256']
        with Image.open(path) as im:return np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
    for row in r['rows']:
        index=row['index'];input_arrays[index]={}
        for role,e in row['inputs'].items():
            path=out/e['path'];assert digest(path)==e['sha256']
            with Image.open(path) as im:a=np.array(im)
            assert hashlib.sha256(a.tobytes()).hexdigest()==e['pixel_sha256'] and np.array_equal(a,read(byindex[index][role]));input_arrays[index][role]=a;inputs_verified+=1
    reused=0
    for item in r['completed']:
        index=item['index'];assert item['strict_deterministic'] and item['activation_checkpoint'] and not item['warn_only'] and item['cublas_workspace_config']==':4096:8' and item['init_cache_release']['state_exact']
        assert sum(e['state_tensors_verified'] for e in item['components'].values())==1130 and all(e['initial_values_exact'] for e in item['components'].values())
        folder=out/str(index)
        for filename,key in [('native512.png','native_sha256'),('metric256.png','metric_sha256')]:assert digest(folder/filename)==item[key]
        with Image.open(folder/'native512.png') as im:assert im.size==(512,512);expected=np.array(im.resize((256,256),Image.Resampling.BILINEAR))
        with Image.open(folder/'metric256.png') as im:a=np.array(im)
        assert np.array_equal(a,expected)
        if 'reused_from' in item:
            assert index==22927 and item['reused_from']==proof.name and item['metric_sha256']==p['completed'][0]['metric_sha256'] and item['native_sha256']==p['completed'][0]['native_sha256'];reused+=1
        truth=read(byindex[index]['gt']);path=BASE/'multiref-control-v1/epoch-6'/f'{index}.png';assert digest(path)==old[index]['sha256']
        with Image.open(path) as im:b=np.array(im)
        mae=float(np.abs(a.astype(float)-truth).mean());base_mae=float(np.abs(b.astype(float)-truth).mean());assert base_mae==old[index]['mae']
        measurements.append(dict(index=index,mae=mae,baseline_mae=base_mae,severe=bool(mae-base_mae>5 and mae>base_mae*1.25)));panels.append((index,[input_arrays[index]['content'],input_arrays[index]['reference'],truth,b,a]))
    assert measurements==r['measurements']==r['partial_measurements'] and reused==r['reused_predictions']==1 and len(measurements)-reused==r['fresh_predictions']
    severe=sum(e['severe'] for e in measurements);assert severe==r['severe_regressions'] and (severe>=3)==r['full64_severe_gate_impossible'] and (len(measurements)==8)==r['all8_completed']
    if r.get('stop_reason')=='necessary_severe_gate':assert severe==3 and sum(e['severe'] for e in measurements[:-1])==2 and r['finished']
    else:assert r['finished']==(len(measurements)==8)
    panel=Image.new('RGB',(1280,len(panels)*280),(245,245,245));draw=ImageDraw.Draw(panel)
    for j,(index,arrays) in enumerate(panels):
        for i,(label,a) in enumerate(zip(['Content','Reference','GT','Current13','StyleGallery'],arrays)):draw.text((i*256+3,j*280+3),str(index)+' '+label,fill='black');panel.paste(Image.fromarray(a),(i*256,j*280+24))
    panel.save(out/'diagnostic-panel.png')
    evidence=dict(scope=r['scope'],results_sha256=digest(out/'results.json'),auditor_sha256=digest(Path(__file__)),all_completed_scores_recomputed=True,measurements=measurements,inputs_verified=inputs_verified,native_and_metric_pngs_verified=len(measurements)*2,reused_cases=reused,fresh_predictions=len(measurements)-reused,severe_regressions=severe,all8_completed=r['all8_completed'],full64_severe_gate_impossible=severe>=3,stop_reason=r.get('stop_reason'),computational_child_ast_exact=True,diagnostic_panel_sha256=digest(out/'diagnostic-panel.png'),full_quality_passed=False,timing_scope='Parent wall excludes prior production of reused22927; per-case timings retain original production scope',parent_wall_seconds=r['elapsed_seconds'],case_timings=[dict(index=e['index'],load=e['load_seconds'],cluster=e['cluster_seconds'],optimization=e['optimization_seconds'],reused='reused_from' in e) for e in r['completed']],product_integrated=False)
    (ROOT/'docs/research/2026-11-17-stylegallery-screen-evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8');print(json.dumps(dict(verified=True,cases=len(measurements),severe=severe,all8_completed=r['all8_completed'],stop=r.get('stop_reason'))))


if __name__=='__main__':main()
