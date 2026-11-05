"""v3 paired ablation; reuses frozen v1/v2 fixtures without changing them."""
from pathlib import Path
import sys,json,runpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate as e
import check_natural as natural
from composition import extract_pair

OUT=e.ROOT/'outputs/content-balance-20261105/reference-composition-v3'

def mapping(a,b,am,bm,amount):
 source,reference,diag=extract_pair(a,b,am,bm,amount)
 return e.build_mapping(source,reference),diag

def main():
 OUT.mkdir(exist_ok=True)
 e.OUT=OUT/'controlled';e.OUT.mkdir(exist_ok=True);e.mapping=mapping
 e.main()
 runpy.run_path(str(Path(__file__).with_name('repeat_controls.py')),run_name='__main__')
 assets=natural.OUT/'assets';natural.OUT=OUT/'natural';natural.OUT.mkdir(exist_ok=True)
 original_load=natural.load
 natural.load=lambda path:original_load(path if path.exists() else assets/path.name)
 natural.fit=lambda a,b,am,bm,amount:mapping(a,b,am,bm,amount)[0]
 natural.main()
 report=json.loads((natural.OUT/'results.json').read_text('utf8'))
 report['method']='Reference-only composition adaptation; source feature packet fixed'
 report['research_only']=True
 (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')

if __name__=='__main__':main()
