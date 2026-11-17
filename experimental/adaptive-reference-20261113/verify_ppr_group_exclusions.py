"""Rebuild query coverage, direct-ID matching, and conservative exclusion sets."""
import collections,json,zipfile
from pathlib import Path
from PIL import PngImagePlugin,JpegImagePlugin,TiffImagePlugin
import link_ppr_training_metadata as link
from verify_ppr_image_links import canonical,extract

ROOT=link.ROOT;BASE=link.BASE
OUT=ROOT/'outputs/adaptive-reference-20261113/ppr-group-exclusions-v2'

def main():
    d=json.loads((OUT/'results.json').read_text());assert d['finished'] and not d['training_started'] and not d['private_identifiers_exported']
    for name,sha in d['source_sha256'].items():assert link.digest(OUT/name)==sha
    p=link.DATA/'srgb-v1/manifest.json';assert link.digest(p)==d['manifest_sha256'];manifest=json.loads(p.read_text())
    index=ROOT/'.tools/TST100K-index/triplet.json';assert link.digest(index)==d['index_sha256']
    originals=collections.defaultdict(set)
    for row in json.loads(index.read_text()):originals[Path(row['content']).stem].add(row['content'])
    expected={};exposure=[]
    for row in manifest['heldout']:
        name=row['path'];family=Path(name).stem if name.startswith('content_images/') else Path(name).stem.rsplit('_',1)[0]
        choices=originals[family];raw=next(iter(choices)) if len(choices)==1 else None
        exposure.append(dict(path=name,direct_query=name,original_query=raw,source_family_options=len(choices)))
        expected.setdefault(name,set()).add('registered_consumed_asset')
        if raw:expected.setdefault(raw,set()).add('consumed_family_original')
    assert exposure==d['exposure'] and len(exposure)==542
    assert set(expected)=={q['path'] for q in d['queries']}
    registry={};roots=[link.DATA/'inputs']
    for name,sha in d['input_manifests_sha256'].items():
        p=BASE/name;assert link.digest(p)==sha;data=json.loads(p.read_text());is_reserved=name.startswith('reserved')
        roots.append(p.parent/('originals' if is_reserved else 'inputs'))
        for row in data['assets' if is_reserved else 'files']:
            key=row['path'];value=row['original_sha256'] if is_reserved else row['sha256']
            assert key not in registry or registry[key]==value;registry[key]=value
    for row in manifest['files']:registry[row['original_path']]=row['original_sha256']
    for row in manifest['heldout']:registry[row['path']]=row['original_sha256']
    identities=collections.defaultdict(set)
    for file,sha in link.EXPECTED.items():
        p=link.ASSETS/file;assert link.digest(p)==sha
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if name.endswith('/'):continue
                ids,_=extract(z.read(name))
                for token in ids:identities[token].add(Path(name).stem)
    classes=[PngImagePlugin.PngImageFile,JpegImagePlugin.JpegImageFile,TiffImagePlugin.TiffImageFile];methods=[c.load for c in classes]
    def forbid(*a,**k):raise AssertionError('Pixel load forbidden')
    for cls in classes:cls.load=forbid
    reasons=collections.defaultdict(set)
    try:
        for q in d['queries']:
            assert sorted(expected[q['path']])==q['roles'];name=q['path']
            if name not in registry:assert q['status']=='missing_registered_identity';continue
            raw=next((r/name for r in roots if (r/name).is_file()),None)
            if raw is None:assert q['status']=='missing_original_file';continue
            assert link.digest(raw)==q['original_sha256']==registry[name]
            packets,extra,_=link.metadata_packets(raw);ids=set();t=canonical('image_unique',extra)
            if t:ids.add(t)
            for packet in packets:a,_=extract(packet);ids|=a
            matched=[identities[t] for t in ids if t in identities]
            common=set.intersection(*matched) if matched else set()
            status='conflict' if matched and not common else 'unique' if len(common)==1 else 'multiple' if common else 'none'
            possible=set.union(*matched) if status=='conflict' else common
            groups={int(x.split('_')[0]) for x in possible}
            assert status==q['status'] and sorted(possible)==q['candidate_ids'] and sorted(groups)==q['possible_groups']
            for g in groups:reasons[g].add(name)
    finally:
        for cls,method in zip(classes,methods,strict=True):cls.load=method
    assert d['group_reasons']==[dict(group=g,queries=sorted(paths)) for g,paths in sorted(reasons.items())]
    previous=link.OUT/'results.json';assert link.digest(previous)==d['previous_training_links_sha256']
    old=[r for r in json.loads(previous.read_text())['rows'] if r['status']=='unique'];assert len(old)==537
    for row,record in zip(old,d['training_candidates'],strict=True):
        key=row['candidate_ids'][0];group=int(key.split('_')[0])
        assert record==dict(path=row['path'],ppr_candidate_id=key,ppr_group=group,exclude_for_known_consumed_group=group in reasons)
    excluded=sum(r['exclude_for_known_consumed_group'] for r in d['training_candidates']);s=d['summary']
    assert s['known_excluded_groups']==len(reasons) and s['training_candidates_excluded']==excluded and s['training_candidates_provisionally_remaining']==537-excluded
    assert not s['full_training_eligibility_established']
    audit=dict(finished=True,consumed_assets=542,queries=len(d['queries']),known_excluded_groups=len(reasons),
        training_candidates_excluded=excluded,training_candidates_provisionally_remaining=537-excluded,pixel_load_guard_passed=True,
        direct_normalized_identifier_matching=True,results_sha256=link.digest(OUT/'results.json'),
        scope='known cached TST metadata exclusions only; missing links remain unresolved')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit))

if __name__=='__main__':main()
