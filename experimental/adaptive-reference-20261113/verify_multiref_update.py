"""Verify owned updated-engine restoration without changing user choices."""
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import Request,urlopen
from launch_spatial_workbench import ROOT,ENHANCED_DATA,probe


def main():
    out=ROOT/'outputs/adaptive-reference-20261113/spatial-workbench/update-multiref-v1';out.mkdir(exist_ok=False)
    record=probe(ENHANCED_DATA,True)
    if record is None:raise RuntimeError('Expected the identified enhanced service')
    def api(path,data=None):
        request=Request(record['url']+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=90) as response:return json.load(response)
    before=api('/api/projects/latest');health=api('/api/health')
    if health['active_exports']:raise RuntimeError('Do not restart during export')
    request=dict(mapping_id=before['mapping_id'],strength=before['strength'],adjustments=before['output_settings'].get('adjustments'),interactive=True)
    image=api('/api/preview',request)
    result=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('launch_spatial_workbench.py')),'--multiref','--check'],capture_output=True,text=True,encoding='utf-8',timeout=90,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError(result.stderr)
    record=probe(ENHANCED_DATA,True)
    if record is None:raise RuntimeError('Updated service did not identify itself')
    restored=api('/api/projects/latest')
    assert restored['target']['id']==before['target']['id'] and restored['reference']['id']==before['reference']['id']
    assert abs(restored['strength']-before['strength'])<1e-7 and restored['output_settings']==before['output_settings']
    stale=restored['needs_analysis']
    if stale:
        mapping=api('/api/analyze',dict(target_id=restored['target']['id'],reference_id=restored['reference']['id'],time_seconds=restored['analysis_time']))
        api('/api/projects',dict(mapping_id=mapping['mapping_id'],strength=restored['strength'],output_settings=restored['output_settings']))
        restored=api('/api/projects/latest')
    assert not restored['needs_analysis']
    request['mapping_id']=restored['mapping_id'];updated=api('/api/preview',request)
    assert updated['before']==image['before'] and updated['after']==image['after']
    report=dict(passed=True,stale_grid_reanalyzed=stale,inputs_and_settings_preserved=True,png_preview_exact=True,
        variant='multiref-temporal-v1',scope='Real owned service update and HTTP restore; no visual browser acceptance')
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(report,url=record['url']),indent=2),flush=True)


if __name__=='__main__':main()
