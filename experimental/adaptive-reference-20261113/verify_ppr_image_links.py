"""Direct-string identity crosscheck; shared header reader is guarded against load()."""
import collections,json,xml.etree.ElementTree as ET,zipfile
from pathlib import Path
from PIL import PngImagePlugin,JpegImagePlugin,TiffImagePlugin
import link_ppr_training_metadata as source

ROOT=source.ROOT;OUT=source.OUT;BASE=source.BASE

def canonical(kind,value):
    if not isinstance(value,str) or len(value)>256:return None
    s=value.strip().lower()
    allowed=['adobe:docid:photoshop:','xmp.did:','xmp.iid:','uuid:']
    prefix=next((p for p in allowed if s.startswith(p)),None)
    if prefix:s=s[len(prefix):]
    s=s.translate(str.maketrans('','','-'))
    if not 16<=len(s)<=64 or any(c not in '0123456789abcdef' for c in s) or not int(s,16):return None
    return kind,s

def extract(raw):
    root=ET.fromstring(raw);ids=set();settings={}
    for (namespace,name),kind in source.KEYS.items():
        tag='{'+namespace+'}'+name
        for node in root.iter():
            values=[]
            if tag in node.attrib:values.append(node.attrib[tag])
            if node.tag==tag and node.text is not None:values.append(node.text)
            for value in values:
                key=canonical(kind,value)
                if key:ids.add(key)
    for node in root.iter():
        for key in source.FIELDS:
            tag='{'+source.CRS+'}'+key
            if tag in node.attrib:
                value=float(node.attrib[tag]);assert key not in settings or settings[key]==value;settings[key]=value
    return ids,settings

def main():
    d=json.loads((OUT/'results.json').read_text());assert d['finished'] and not d['training_started'] and not d['private_identifiers_exported']
    assert source.digest(OUT/'protocol.md')==d['protocol_sha256']
    paths=source.DATA/'cohort.json';assert source.digest(paths)==d['cohort_sha256']
    cohort=json.loads(paths.read_text());names=sorted({r['content'] for r in cohort['groups']['train']})
    assert names==[r['path'] for r in d['rows']] and len(names)==1024
    mapping=collections.defaultdict(set);records={}
    for role,file in [('source','xmp_source.zip'),('target_a','xmp_target_a.zip')]:
        p=source.ASSETS/file;assert source.digest(p)==source.EXPECTED[file]
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if name.endswith('/'):continue
                key=Path(name).stem;identifiers,settings=extract(z.read(name));records[role,key]=settings
                for identifier in identifiers:mapping[identifier].add(key)
    classes=[PngImagePlugin.PngImageFile,JpegImagePlugin.JpegImageFile,TiffImagePlugin.TiffImageFile]
    saved=[cls.load for cls in classes]
    def forbid(*args,**kwargs):raise AssertionError('Pixel load attempted during metadata-only audit')
    for cls in classes:cls.load=forbid
    try:
        for row in d['rows']:
            p=next(BASE/r/'inputs'/row['path'] for r in ['cohort-training-coverage-v2','cohort-training-coverage-v1','cohort-multiref-v1','cohort-v2','cohort-v1'] if (BASE/r/'inputs'/row['path']).is_file())
            assert source.digest(p)==row['original_sha256']
            packets,unique,fmt=source.metadata_packets(p);ids=set();values={}
            key=canonical('image_unique',unique)
            if key:ids.add(key)
            for packet in packets:
                a,b=extract(packet);ids|=a;values.update(b)
            assert len(packets)==row['xmp_packets'] and len(ids)==row['valid_identifier_tokens'] and fmt==row['format']
            matched=[mapping[k] for k in ids if k in mapping];common=set.intersection(*matched) if matched else set()
            status='conflict' if matched and not common else 'unique' if len(common)==1 else 'multiple' if common else 'none'
            assert sorted(common)==row['candidate_ids'] and status==row['status'] and len(matched)==row['matched_identifier_tokens']
            if status=='unique':
                candidate=next(iter(common))
                for role in ['source','target_a']:
                    v=records[role,candidate];complete=all(k in values and k in v for k in source.FIELDS)
                    equal=complete and all(values[k]==v[k] for k in source.FIELDS)
                    assert row['settings_equal'][role]==dict(all_six_present=complete,all_six_equal=equal)
    finally:
        for cls,method in zip(classes,saved,strict=True):cls.load=method
    counts=dict(collections.Counter(r['status'] for r in d['rows']));assert counts==d['status_counts']
    audit=dict(finished=True,rows=1024,status_counts=counts,pixel_load_guard_passed=True,
        identity_matcher='independent direct normalized strings, no SHA token comparison',
        shared_component='metadata byte extraction only',results_sha256=source.digest(OUT/'results.json'))
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit))

if __name__=='__main__':main()
