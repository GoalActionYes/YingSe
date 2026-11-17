"""Download only pinned public FP16 decomposition assets, no global install."""
import hashlib,json,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'.tools/Marigold-IID-audit'
REPO='prs-eth/marigold-iid-lighting-v1-1'
REV='08c3930bb641abf786ba44ce92547507ebefbc16'
OP=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
FILES=['README.md','model_index.json','scheduler/scheduler_config.json','text_encoder/config.json',
       'tokenizer/merges.txt','tokenizer/vocab.json','tokenizer/tokenizer_config.json','tokenizer/special_tokens_map.json',
       'unet/config.json','vae/config.json','unet/diffusion_pytorch_model.fp16.safetensors',
       'text_encoder/model.fp16.safetensors','vae/diffusion_pytorch_model.fp16.safetensors']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()

def main():
    BASE.mkdir(exist_ok=True);metadata=json.loads(OP.open(f'https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true',timeout=30).read())
    assert metadata['sha']==REV and not metadata['gated']
    lookup={f['rfilename']:f for f in metadata['siblings']}
    assert sum(lookup[n]['size'] for n in FILES)<2_585_000_000
    records=[];models=BASE/'models';models.mkdir(exist_ok=True)
    for name in FILES:
        entry=lookup[name];p=models/name;p.parent.mkdir(parents=True,exist_ok=True)
        expected=entry.get('lfs',{}).get('sha256')
        if not p.exists():
            partial=p.with_suffix(p.suffix+'.partial');start=time.monotonic();downloaded=0
            # A partial interrupted artifact is retained; resume only after checking transport identity separately.
            assert not partial.exists(),f'Inspect existing partial before resuming: {partial}'
            with OP.open(f'https://huggingface.co/{REPO}/resolve/{REV}/{name}?download=true',timeout=60) as response,partial.open('wb') as stream:
                while block:=response.read(1024*1024):
                    downloaded+=len(block);assert downloaded<=entry['size'] and time.monotonic()-start<900
                    stream.write(block)
                    if downloaded%(128*1024*1024)==0:print(json.dumps(dict(file=name,bytes=downloaded)),flush=True)
            assert downloaded==entry['size'];partial.rename(p)
        assert p.stat().st_size==entry['size']
        actual=sha(p)
        if expected:assert actual==expected,(name,actual)
        else:
            data=p.read_bytes();assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==entry['blobId']
        records.append(dict(path=name,bytes=entry['size'],sha256=actual,git_blob=entry['blobId']))
        print(json.dumps(dict(verified=name,bytes=entry['size'])),flush=True)
    (BASE/'assets.json').write_text(json.dumps(dict(repository=REPO,revision=REV,files=records,finished=True),indent=2),encoding='utf-8')

if __name__=='__main__':main()
