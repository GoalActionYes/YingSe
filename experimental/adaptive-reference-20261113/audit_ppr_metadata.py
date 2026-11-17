"""Whitelisted Camera Raw numeric metadata audit; never export original XMP."""
import collections,hashlib,json,re,time,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'.tools/PPR10K-audit'
OUT=ROOT/'outputs/adaptive-reference-20261113/ppr-metadata-v2'
NS='http://ns.adobe.com/camera-raw-settings/1.0/'
FIELDS=['Temperature','Tint','Exposure','Exposure2012','Highlights','Highlights2012','Contrast','Contrast2012','Saturation']
ARCHIVES={'source':('xmp_source.zip',29168614,'ba675d3ef314f6ba0446f50b2b67c940c631c9235d3a95e18a1614a612c9ee9a'),
          'target_a':('xmp_target_a.zip',27200780,'b02e46d25ef0aeee4c6c6a1bc5dc517066bf8b009696a70d7db1f48083d04a7a')}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def parse_archive(path):
    data={};tags=collections.Counter();ignored=0;uncompressed=0
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None,'ZIP CRC failed'
        assert sum(i.file_size for i in z.infolist())<=150_000_000
        for item in z.infolist():
            name=Path(item.filename).name
            if item.is_dir() or item.filename.startswith('__MACOSX/') or name.startswith('._'):
                ignored+=1;continue
            assert re.fullmatch(r'\d+_\d+\.xmp',name,re.I), 'Unexpected sample naming'
            assert 0<item.file_size<=2_000_000
            raw=z.read(item);assert b'<!DOCTYPE' not in raw.upper() and b'<!ENTITY' not in raw.upper()
            xml=ET.fromstring(raw);values={};present=set()
            for node in xml.iter():
                if node.tag.startswith('{'+NS+'}'):present.add(node.tag.split('}',1)[1])
                for key,value in node.attrib.items():
                    if not key.startswith('{'+NS+'}'):continue
                    field=key.split('}',1)[1];present.add(field)
                    if field in FIELDS:
                        number=float(value);assert np.isfinite(number)
                        assert field not in values or values[field]==number
                        values[field]=number
            tags.update(present)
            key=name.lower()[:-4];assert key not in data
            data[key]=values;uncompressed+=len(raw)
    return data,dict(samples=len(data),ignored_entries=ignored,xmp_bytes=uncompressed,crs_field_presence=dict(sorted(tags.items())))

def summarize(source,target):
    assert set(source)==set(target) and len(source)==11161
    groups=collections.Counter(int(k.split('_')[0]) for k in source)
    out=dict(pairs=len(source),groups=len(groups),train_pairs=sum(n for g,n in groups.items() if g<1356),
             validation_pairs=sum(n for g,n in groups.items() if g>=1356),fields={})
    for field in FIELDS:
        pairs=[(key,target[key][field]-source[key][field]) for key in sorted(source) if field in source[key] and field in target[key]]
        values=np.array([v for _,v in pairs],dtype=float);by_group=collections.defaultdict(list)
        for key,v in pairs:by_group[int(key.split('_')[0])].append(v)
        eligible=[v for v in by_group.values() if len(v)>=2]
        out['fields'][field]=dict(paired_values=len(pairs),missing_pairs=len(source)-len(pairs),
            changed=int(np.count_nonzero(np.abs(values)>1e-9)),unique_delta_rounded_6=len(set(np.round(values,6))),
            delta_quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist() if len(values) else None,
            groups_with_multiple_values=len(eligible),groups_with_nonconstant_delta=int(sum(np.ptp(v)>1e-9 for v in eligible)))
    return out

def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    manifest=json.loads((ASSETS/'source-manifest.json').read_text());sources={}
    for f in manifest['files']:
        p=ASSETS/'source'/f['path'];raw=p.read_bytes()
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==f['sha']
        sources[f['path']]=dict(git_blob=f['sha'],sha256=digest(p))
    config=dict(scope='numeric expert-edit metadata only; no photos, model, training or independent evaluation',
        revision=manifest['revision'],sources=sources,archives={},finished=False,product_modified=False,
        protocol_sha256=digest(Path(__file__).with_name('ppr-metadata-protocol.md')),driver_sha256=digest(__file__))
    for name,(file,size,expected) in ARCHIVES.items():
        p=ASSETS/file;assert p.stat().st_size==size and digest(p)==expected
        config['archives'][name]=dict(file=file,bytes=size,sha256=expected)
    (OUT/'frozen-config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (OUT/'protocol.md').write_bytes(Path(__file__).with_name('ppr-metadata-protocol.md').read_bytes())
    data={}
    for name,(file,_,_) in ARCHIVES.items():
        assert time.perf_counter()-start<120
        data[name],stats=parse_archive(ASSETS/file);config['archives'][name].update(stats)
    config['summary']=summarize(data['source'],data['target_a'])
    config.update(finished=True,seconds=time.perf_counter()-start,private_fields_exported=False)
    assert digest(__file__)==config['driver_sha256']
    (OUT/'results.json').write_text(json.dumps(config,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(summary=config['summary'],seconds=config['seconds'])),flush=True)

if __name__=='__main__':main()
