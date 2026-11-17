"""Link known training headers by in-memory hashed document IDs, never filenames."""
import collections,hashlib,json,re,struct,time,zipfile,zlib
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/adaptive-reference-20261113/tst100k'
DATA=BASE/'cohort-training-coverage-v2'
ASSETS=ROOT/'.tools/PPR10K-audit'
OUT=ROOT/'outputs/adaptive-reference-20261113/ppr-image-link-v2'
CRS='http://ns.adobe.com/camera-raw-settings/1.0/'
MM='http://ns.adobe.com/xap/1.0/mm/'
EXIF='http://ns.adobe.com/exif/1.0/'
KEYS={(MM,'DocumentID'):'document',(MM,'OriginalDocumentID'):'document',(MM,'InstanceID'):'instance',
      (CRS,'RawFileDigest'):'raw_digest',(EXIF,'ImageUniqueID'):'image_unique'}
FIELDS=['Temperature','Tint','Exposure2012','Highlights2012','Contrast2012','Saturation']
EXPECTED={'xmp_source.zip':'ba675d3ef314f6ba0446f50b2b67c940c631c9235d3a95e18a1614a612c9ee9a',
          'xmp_target_a.zip':'b02e46d25ef0aeee4c6c6a1bc5dc517066bf8b009696a70d7db1f48083d04a7a'}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def token(kind,value):
    if not isinstance(value,str) or len(value)>256:return None
    text=value.strip().lower()
    for prefix in ['adobe:docid:photoshop:','xmp.did:','xmp.iid:','uuid:']:
        if text.startswith(prefix):text=text[len(prefix):];break
    text=text.replace('-','')
    if not re.fullmatch(r'[0-9a-f]{16,64}',text) or set(text)=={'0'}:return None
    return kind,hashlib.sha256(text.encode('ascii')).hexdigest()

def xml_info(raw):
    if isinstance(raw,str):raw=raw.encode('utf-8')
    assert len(raw)<=2_000_000 and b'<!DOCTYPE' not in raw.upper() and b'<!ENTITY' not in raw.upper()
    root=ET.fromstring(raw);ids=set();values={}
    for node in root.iter():
        for (ns,key),kind in KEYS.items():
            attribute='{'+ns+'}'+key
            value=node.attrib.get(attribute)
            if value is None and node.tag==attribute:value=node.text
            if value is not None:
                item=token(kind,value)
                if item:ids.add(item)
        for key in FIELDS:
            value=node.attrib.get('{'+CRS+'}'+key)
            if value is not None:
                n=float(value);assert abs(n)<1e8
                assert key not in values or values[key]==n;values[key]=n
    return ids,values

def png_xmp(raw):
    assert raw[:8]==b'\x89PNG\r\n\x1a\n';offset=8;packets=[];exif=None
    while offset<len(raw):
        length=int.from_bytes(raw[offset:offset+4],'big');kind=raw[offset+4:offset+8];end=offset+8+length
        assert end+4<=len(raw)
        data=raw[offset+8:end]
        if kind==b'eXIf':exif=data
        if kind in [b'iTXt',b'tEXt',b'zTXt']:
            key,rest=data.split(b'\0',1)
            if key.lower()==b'xml:com.adobe.xmp':
                assert len(rest)<=4_000_000
                if kind==b'tEXt':text=rest
                elif kind==b'zTXt':
                    assert rest[0]==0;text=zlib.decompressobj().decompress(rest[1:],2_000_001)
                else:
                    compressed,method=rest[:2];rest=rest[2:];_,rest=rest.split(b'\0',1);_,text=rest.split(b'\0',1)
                    assert method==0 and compressed in (0,1)
                    if compressed:text=zlib.decompressobj().decompress(text,2_000_001)
                assert len(text)<=2_000_000;packets.append(text)
        offset=end+4
        if kind==b'IEND':break
    return packets,exif

def metadata_packets(path):
    raw=path.read_bytes();packets=[]
    with Image.open(path) as im:
        fmt=im.format
        if fmt=='PNG':
            packets,raw_exif=png_xmp(raw);exif=Image.Exif()
            if raw_exif:exif.load(raw_exif)
        else:
            if hasattr(im,'tag_v2') and 700 in im.tag_v2:packets.append(bytes(im.tag_v2[700]))
            elif im.info.get('xmp'):packets.append(im.info['xmp'])
            exif=Image.Image.getexif(im)
        value=exif.get(42016)
        if value is None and 34665 in exif:value=exif.get_ifd(34665).get(42016)
        if isinstance(value,bytes):value=value.decode('ascii',errors='ignore').rstrip('\0')
    return packets,value,fmt

def image_info(path):
    packets,value,fmt=metadata_packets(path);extra=[]
    t=token('image_unique',value)
    if t:extra.append(t)
    ids=set(extra);values={}
    for packet in packets:
        a,b=xml_info(packet);ids|=a
        for k,v in b.items():assert k not in values or values[k]==v;values[k]=v
    return ids,values,dict(format=fmt,xmp_packets=len(packets),valid_identifier_tokens=len(ids),numeric_fields_present=len(values))

def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    cohort_path=DATA/'cohort.json';manifest_path=DATA/'srgb-v1/manifest.json'
    assert digest(cohort_path)=='ed423c03750504c0e478bbebb30e05ce6bfc5228fec940dcbae073827a3fb077'
    assert digest(manifest_path)=='da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f'
    cohort=json.loads(cohort_path.read_text());manifest={r['original_path']:r for r in json.loads(manifest_path.read_text())['files']}
    paths=sorted({r['content'] for r in cohort['groups']['train']});assert len(paths)==1024
    report=dict(scope='known training image header association only, not confirmed pixel pairing or new evaluation',
        cohort_sha256=digest(cohort_path),manifest_sha256=digest(manifest_path),
        protocol_sha256=digest(Path(__file__).with_name('ppr-image-link-protocol.md')),driver_sha256=digest(__file__),
        archives=EXPECTED,rows=[],finished=False,training_started=False,private_identifiers_exported=False)
    (OUT/'protocol.md').write_bytes(Path(__file__).with_name('ppr-image-link-protocol.md').read_bytes())
    (OUT/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    mapping=collections.defaultdict(set);settings={};archive_stats={}
    for side,name in [('source','xmp_source.zip'),('target_a','xmp_target_a.zip')]:
        p=ASSETS/name;assert digest(p)==EXPECTED[name];counts=collections.Counter();samples=0
        with zipfile.ZipFile(p) as z:
            for item in z.infolist():
                if item.is_dir():continue
                leaf=Path(item.filename).name;assert re.fullmatch(r'\d+_\d+\.xmp',leaf,re.I)
                key=leaf[:-4];ids,values=xml_info(z.read(item));settings[side,key]=values;samples+=1
                for ident in ids:mapping[ident].add(key);counts[ident[0]]+=1
        assert samples==11161;archive_stats[side]=dict(samples=samples,identifier_occurrences=dict(counts))
    for name in paths:
        assert time.perf_counter()-start<180
        found=[BASE/root/'inputs'/name for root in ['cohort-training-coverage-v2','cohort-training-coverage-v1','cohort-multiref-v1','cohort-v2','cohort-v1'] if (BASE/root/'inputs'/name).is_file()]
        assert found,'Missing original asset'
        p=found[0];assert digest(p)==manifest[name]['original_sha256']
        ids,values,meta=image_info(p);sets=[mapping[i] for i in ids if i in mapping]
        candidates=set.intersection(*sets) if sets else set();conflict=bool(sets and not candidates)
        status='conflict' if conflict else 'unique' if len(candidates)==1 else 'multiple' if candidates else 'none'
        equal={}
        if status=='unique':
            key=next(iter(candidates))
            for side in ['source','target_a']:
                old=settings[side,key];complete=all(f in values and f in old for f in FIELDS)
                equal[side]=dict(all_six_present=complete,all_six_equal=complete and all(values[f]==old[f] for f in FIELDS))
        report['rows'].append(dict(path=name,original_sha256=manifest[name]['original_sha256'],**meta,
            matched_identifier_tokens=len(sets),candidate_count=len(candidates),status=status,candidate_ids=sorted(candidates),settings_equal=equal))
    report.update(finished=True,seconds=time.perf_counter()-start,archive_identifier_stats=archive_stats,
        status_counts=dict(collections.Counter(r['status'] for r in report['rows'])),
        format_counts=dict(collections.Counter(r['format'] for r in report['rows'])),
        files_with_xmp=sum(r['xmp_packets']>0 for r in report['rows']),files_with_identifiers=sum(r['valid_identifier_tokens']>0 for r in report['rows']))
    assert digest(__file__)==report['driver_sha256']
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['finished','seconds','status_counts','format_counts','files_with_xmp','files_with_identifiers']}))

if __name__=='__main__':main()
