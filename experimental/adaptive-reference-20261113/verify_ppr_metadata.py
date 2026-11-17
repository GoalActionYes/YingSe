"""Independent SAX/count/quantile audit; never outputs source XML or EXIF."""
import collections,hashlib,io,json,math,re,xml.sax,zipfile
from pathlib import Path
from xml.sax.handler import ContentHandler,feature_namespaces,feature_external_ges,feature_external_pes

ROOT=Path(__file__).resolve().parents[2]
ASSETS=ROOT/'.tools/PPR10K-audit'
OUT=ROOT/'outputs/adaptive-reference-20261113/ppr-metadata-v2'
URI='http://ns.adobe.com/camera-raw-settings/1.0/'
FIELDS=['Temperature','Tint','Exposure','Exposure2012','Highlights','Highlights2012','Contrast','Contrast2012','Saturation']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class CameraRaw(ContentHandler):
    def __init__(self):super().__init__();self.values={};self.present=set()
    def startElementNS(self,name,qname,attrs):
        if name[0]==URI:self.present.add(name[1])
        for (ns,key),value in attrs.items():
            if ns!=URI:continue
            self.present.add(key)
            if key in FIELDS:
                v=float(value);assert math.isfinite(v)
                assert key not in self.values or v==self.values[key]
                self.values[key]=v

def scan(path):
    rows={};tags=collections.Counter();ignored=0;size=0
    parser=xml.sax.make_parser();parser.setFeature(feature_namespaces,True)
    parser.setFeature(feature_external_ges,False);parser.setFeature(feature_external_pes,False)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            leaf=Path(info.filename).name
            if info.is_dir() or info.filename.startswith('__MACOSX/') or leaf.startswith('._'):ignored+=1;continue
            assert re.fullmatch(r'\d+_\d+\.xmp',leaf,re.I)
            raw=z.read(info);assert 0<len(raw)<=2_000_000
            assert b'<!DOCTYPE' not in raw.upper() and b'<!ENTITY' not in raw.upper()
            handler=CameraRaw();parser.setContentHandler(handler);parser.parse(io.BytesIO(raw))
            key=leaf.lower()[:-4];assert key not in rows;rows[key]=handler.values
            tags.update(handler.present);size+=len(raw)
    return rows,dict(samples=len(rows),ignored_entries=ignored,xmp_bytes=size,crs_field_presence=dict(sorted(tags.items())))

def quantile(values,p):
    a=sorted(values);x=(len(a)-1)*p;lo=math.floor(x);hi=math.ceil(x)
    return a[lo]*(1-(x-lo))+a[hi]*(x-lo)

def main():
    report=json.loads((OUT/'results.json').read_text());assert report['finished'] and not report['product_modified'] and not report['private_fields_exported']
    assert digest(OUT/'protocol.md')==report['protocol_sha256']
    datasets={}
    for name,item in report['archives'].items():
        p=ASSETS/item['file'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
        datasets[name],meta=scan(p)
        assert all(meta[k]==item[k] for k in meta)
    source,target=datasets['source'],datasets['target_a'];assert set(source)==set(target) and len(source)==11161
    groups=collections.Counter(int(k.split('_')[0]) for k in source);s=report['summary']
    assert s['pairs']==len(source) and s['groups']==len(groups)
    assert s['train_pairs']==sum(v for g,v in groups.items() if g<1356)
    assert s['validation_pairs']==sum(v for g,v in groups.items() if g>=1356)
    for field in FIELDS:
        pairs=[(k,target[k][field]-v[field]) for k,v in source.items() if field in v and field in target[k]]
        deltas=[v for _,v in pairs];g=collections.defaultdict(list)
        for k,v in pairs:g[int(k.split('_')[0])].append(v)
        enough=[v for v in g.values() if len(v)>=2]
        expected=dict(paired_values=len(pairs),missing_pairs=len(source)-len(pairs),changed=sum(abs(v)>1e-9 for v in deltas),
            unique_delta_rounded_6=len(set(round(v,6) for v in deltas)),groups_with_multiple_values=len(enough),
            groups_with_nonconstant_delta=sum(max(v)-min(v)>1e-9 for v in enough))
        assert all(s['fields'][field][k]==v for k,v in expected.items())
        q=[quantile(deltas,p) for p in [0,.25,.5,.75,1]] if deltas else None
        recorded=s['fields'][field]['delta_quantiles']
        assert (q is None and recorded is None) or all(abs(a-b)<=1e-9 for a,b in zip(q,recorded,strict=True))
    result=dict(finished=True,pairs=11161,groups=len(groups),independent_sax=True,
        results_sha256=digest(OUT/'results.json'),scope='metadata structure and aggregate statistics only, not retouching quality')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
