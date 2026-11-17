"""Pin author source snapshots only; never run their data-moving scripts."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'.tools/PPR10K-audit'
REV='239490b8a8e917f2c29e3016ee2f6543c3c1bfba'
FILES={
  "README.md": "b74ad748305b3c81946b35e4f08b1d6133fc945a",
  "docs/dataset_usage.md": "86139c14b07b73aa17b7c08ac28a36bb4718f7a0",
  "code_3DLUT/datasets.py": "305a1a02afbdead8e923f9e85b0b9c63b3b70ffb",
  "code_3DLUT/datasets_GLC.py": "01a27542e2346baf0d6a559f3a725050d05fd639",
  "utils/split_train_val.py": "a02d3ac93dd49d57e354691052baf37b054ca5f0",
  "utils/count_number_each_group.py": "316eb06a9679f7a1feec2b5e3712b8e5a5cbf3ec"
}
def main():
    BASE.mkdir(exist_ok=True)
    op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
    entries=[]
    for name,expected in FILES.items():
        p=BASE/'source'/name
        if p.exists():data=p.read_bytes()
        else:
            with op.open(f'https://raw.githubusercontent.com/csjliang/PPR10K/{REV}/{name}',timeout=30) as r:data=r.read(30001)
        assert len(data)<=30000
        assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==expected
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        entries.append(dict(path=name,sha=expected,size=len(data)))
    (BASE/'source-manifest.json').write_text(json.dumps(dict(revision=REV,files=entries),indent=2),encoding='utf-8')
    print('Verified six pinned source snapshots; no author scripts executed')
if __name__=='__main__':main()
