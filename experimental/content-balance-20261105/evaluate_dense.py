"""V2 evaluation in a new output directory; preserves v1 counterexamples."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate as e
from dense_features import extract_dense

def dense_mapping(source,ref,sm,rm,amount):
 a=extract_dense(source,0,sm,amount);b=extract_dense(ref,1,rm,amount)
 return e.build_mapping(a,b),dict(source=a.diagnostics,reference=b.diagnostics)

e.mapping=dense_mapping
e.OUT=e.ROOT/'outputs/content-balance-20261105/dense-v2';e.OUT.mkdir(exist_ok=True)
e.main()
code=(Path(__file__).parent/'repeat_controls.py').read_text('utf8')
exec(compile(code,'dense-repeat-controls','exec'),{'__file__':str(Path(__file__).parent/'repeat_controls.py')})
