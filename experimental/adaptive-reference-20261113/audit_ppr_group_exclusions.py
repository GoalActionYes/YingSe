"""Conservative metadata group exclusions, never new image scoring or training."""
import collections,json,time,zipfile
from pathlib import Path
from PIL import PngImagePlugin,JpegImagePlugin,TiffImagePlugin
import link_ppr_training_metadata as link

ROOT=link.ROOT;BASE=link.BASE
OUT=ROOT/'outputs/adaptive-reference-20261113/ppr-group-exclusions-v2'

def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False)
    manifest_path=link.DATA/'srgb-v1/manifest.json';assert link.digest(manifest_path)=='da0b2162eea340fb75c2a1cfb62066658865a72873bbf951a01202add692603f'
    manifest=json.loads(manifest_path.read_text());held=manifest['heldout'];assert len(held)==542
    previous_path=link.OUT/'results.json';previous=json.loads(previous_path.read_text());assert previous['finished']
    assert link.digest(Path(link.__file__))==previous['driver_sha256']
    index=ROOT/'.tools/TST100K-index/triplet.json';assert link.digest(index)=='5e71ea9eb3255d0169f042d5fe4f9904f5ef9849024cba98fce74c5299159413'
    originals=collections.defaultdict(set)
    for row in json.loads(index.read_text()):originals[Path(row['content']).stem].add(row['content'])
    inputs=sorted(BASE.glob('*/inputs-manifest.json'));registered={};input_hashes={}
    for p in inputs:
        input_hashes[str(p.relative_to(BASE))]=link.digest(p)
        for row in json.loads(p.read_text())['files']:
            if row['path'] in registered:assert registered[row['path']]==row['sha256']
            registered[row['path']]=row['sha256']
    for row in manifest['files']:
        if row['original_path'] in registered:assert registered[row['original_path']]==row['original_sha256']
        registered[row['original_path']]=row['original_sha256']
    roots=[link.DATA/'inputs']+[p.parent/'inputs' for p in inputs]
    for p in sorted(BASE.glob('reserved*/results.json')):
        assets=json.loads(p.read_text()).get('assets')
        if not isinstance(assets,list):continue
        input_hashes[str(p.relative_to(BASE))]=link.digest(p);roots.append(p.parent/'originals')
        for row in assets:
            name=row['path'];sha=row['original_sha256']
            if name in registered:assert registered[name]==sha
            registered[name]=sha
    queries={};exposure=[]
    for row in held:
        name=row['path'];expected=row['original_sha256']
        if name in registered:assert registered[name]==expected
        registered[name]=expected
        key=Path(name).stem if name.startswith('content_images/') else Path(name).stem.rsplit('_',1)[0]
        choices=originals[key];original=next(iter(choices)) if len(choices)==1 else None
        exposure.append(dict(path=name,direct_query=name,original_query=original,source_family_options=len(choices)))
        queries.setdefault(name,set()).add('registered_consumed_asset')
        if original:queries.setdefault(original,set()).add('consumed_family_original')
    names=['audit_ppr_group_exclusions.py','ppr-group-exclusion-protocol.md','link_ppr_training_metadata.py']
    report=dict(scope='known consumed TST metadata group exclusion only; not full scene isolation or new acceptance',
        manifest_sha256=link.digest(manifest_path),previous_training_links_sha256=link.digest(previous_path),index_sha256=link.digest(index),
        input_manifests_sha256=input_hashes,source_sha256={n:link.digest(Path(__file__).with_name(n)) for n in names},
        exposure=exposure,queries=[],finished=False,training_started=False,private_identifiers_exported=False)
    for n in names:(OUT/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    (OUT/'frozen-config.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    mapping=collections.defaultdict(set)
    for name,sha in link.EXPECTED.items():
        p=link.ASSETS/name;assert link.digest(p)==sha
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if name.endswith('/'):continue
                ids,_=link.xml_info(z.read(name));key=Path(name).stem
                for ident in ids:mapping[ident].add(key)
    classes=[PngImagePlugin.PngImageFile,JpegImagePlugin.JpegImageFile,TiffImagePlugin.TiffImageFile];saved=[c.load for c in classes]
    def forbid(*a,**k):raise AssertionError('Pixel decoding forbidden')
    for cls in classes:cls.load=forbid
    cache={}
    try:
        for name,roles in sorted(queries.items()):
            assert time.perf_counter()-start<180
            base=dict(path=name,roles=sorted(roles),candidate_ids=[],possible_groups=[])
            if name not in registered:report['queries'].append(dict(base,status='missing_registered_identity'));continue
            found=next((r/name for r in roots if (r/name).is_file()),None)
            if found is None:report['queries'].append(dict(base,status='missing_original_file'));continue
            assert link.digest(found)==registered[name]
            ids,_,meta=link.image_info(found);sets=[mapping[i] for i in ids if i in mapping]
            candidates=set.intersection(*sets) if sets else set();conflict=bool(sets and not candidates)
            status='conflict' if conflict else 'unique' if len(candidates)==1 else 'multiple' if candidates else 'none'
            possible=set.union(*sets) if conflict else candidates
            report['queries'].append(dict(base,original_sha256=registered[name],local_original=str(found.relative_to(ROOT)),**meta,
                status=status,candidate_ids=sorted(possible),possible_groups=sorted({int(i.split('_')[0]) for i in possible})))
    finally:
        for cls,method in zip(classes,saved,strict=True):cls.load=method
    group_to_queries=collections.defaultdict(list)
    for q in report['queries']:
        for g in q['possible_groups']:group_to_queries[g].append(q['path'])
    excluded=set(group_to_queries);training=[]
    for row in previous['rows']:
        if row['status']!='unique':continue
        key=row['candidate_ids'][0];g=int(key.split('_')[0])
        training.append(dict(path=row['path'],ppr_candidate_id=key,ppr_group=g,exclude_for_known_consumed_group=g in excluded))
    assert len(training)==537
    report.update(finished=True,seconds=time.perf_counter()-start,pixel_load_guard_passed=True,
        group_reasons=[dict(group=g,queries=sorted(q)) for g,q in sorted(group_to_queries.items())],training_candidates=training,
        summary=dict(consumed_registry_assets=542,queries=len(queries),query_status=dict(collections.Counter(q['status'] for q in report['queries'])),
            known_excluded_groups=len(excluded),training_candidates=537,training_candidates_excluded=sum(r['exclude_for_known_consumed_group'] for r in training),
            training_candidates_provisionally_remaining=sum(not r['exclude_for_known_consumed_group'] for r in training),full_training_eligibility_established=False))
    assert all(link.digest(Path(__file__).with_name(n))==h for n,h in report['source_sha256'].items())
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(summary=report['summary'],seconds=report['seconds'])))

if __name__=='__main__':main()
