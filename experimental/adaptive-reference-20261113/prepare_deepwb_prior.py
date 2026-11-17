"""Pinned, local-only research assets; no global environment changes."""
import hashlib, json, urllib.request, pathlib, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[2] / '.tools/DeepWB-audit'
github_opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
REV = 'd70afa227c94152fd8421db75b3ee6ac06a417dc'
FILES = {
 'PyTorch/arch/__init__.py':'fb23c94fc9f912b4fec74113fc07f3ae9b6b9b7f',
 'PyTorch/arch/deep_wb_blocks.py':'0177f1d0827bd581aaeb1f3e7954b5ad0f091a4f',
 'PyTorch/arch/deep_wb_model.py':'908b309801bd9be533369c89adf0d2d87d55b71a',
 'PyTorch/arch/deep_wb_single_task.py':'b5bad473e79f139b8c96f059f2239857b40eee6a',
 'PyTorch/utilities/deepWB.py':'f98d70ba644c8c7899c06d37ff884e3d6a30f534',
 'PyTorch/utilities/utils.py':'ea2eadd4441fbc0b2f924eba99747297e7a397fd',
 'PyTorch/models/net_awb.pth':'54177b02dee127e7a3ef078f479a0e787ee5644e',
}
manifest={'revision':REV,'files':[],'wheels':[]}
for name,expected in FILES.items():
 p=ROOT/'source'/name
 if p.exists(): data=p.read_bytes()
 else:
  data=github_opener.open(f'https://raw.githubusercontent.com/mahmoudnafifi/Deep_White_Balance/{REV}/{name}',timeout=90).read()
 actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
 assert actual==expected,(name,actual)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 manifest['files'].append({'path':name,'bytes':len(data),'git_blob':actual,'sha256':hashlib.sha256(data).hexdigest()})
 print(name,len(data),flush=True)
for name,ver in [('scikit-learn','1.7.2'),('scipy','1.16.3'),('joblib','1.5.2'),('threadpoolctl','3.6.0')]:
 d=json.load(github_opener.open(f'https://pypi.org/pypi/{name}/{ver}/json',timeout=30))
 fs=[f for f in d['urls'] if ('cp312-cp312-win_amd64' in f['filename'] or 'py3-none-any.whl' in f['filename'] or 'py2.py3-none-any.whl' in f['filename'])]
 assert len(fs)==1,(name,len(fs))
 f=fs[0];p=ROOT/'wheels'/f['filename'];p.parent.mkdir(exist_ok=True)
 data=p.read_bytes() if p.exists() else github_opener.open(f['url'],timeout=60).read()
 assert len(data)==f['size'] and hashlib.sha256(data).hexdigest()==f['digests']['sha256']
 p.write_bytes(data)
 vendor=ROOT/'vendor';vendor.mkdir(exist_ok=True)
 with zipfile.ZipFile(p) as z:
  assert all((vendor/n).resolve().is_relative_to(vendor.resolve()) for n in z.namelist())
  z.extractall(vendor)
 manifest['wheels'].append({'name':name,'version':ver,'filename':p.name,'bytes':len(data),'sha256':f['digests']['sha256'],'url':f['url'],'requirements':d['info']['requires_dist'],'license':d['info'].get('license_expression') or d['info'].get('license')})
 print(p.name,len(data),flush=True)
(ROOT/'assets.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
