"""Independent direct-string matching and set reconstruction for the 159 additions."""
import collections
import hashlib
import json
from pathlib import Path
import zipfile
from PIL import PngImagePlugin, JpegImagePlugin, TiffImagePlugin
import link_ppr_training_metadata as link
from verify_ppr_image_links import canonical, extract

BASE = link.ROOT/'outputs/adaptive-reference-20261113'
OUT = BASE/'ppr-group-exclusions-v3'


def main():
    d = json.loads((OUT/'results.json').read_text())
    parent_path = BASE/'ppr-group-exclusions-v2/results.json'
    download_path = BASE/'ppr-family-originals-v1/download-results.json'
    plan_path = BASE/'ppr-family-originals-v1/plan.json'
    for path, key in [(parent_path,'parent_sha256'),(download_path,'download_sha256'),(plan_path,'plan_sha256')]:
        assert link.digest(path) == d[key]
    for name, sha in d['source_sha256'].items(): assert link.digest(OUT/name) == sha
    old = json.loads(parent_path.read_text())
    downloads = {r['path']: r for r in json.loads(download_path.read_text())['files']}
    plans = {r['path']: r for r in json.loads(plan_path.read_text())['files']}
    pending = {q['path'] for q in old['queries'] if q['status']=='missing_registered_identity'}
    assert pending == set(downloads) == set(plans) and len(pending)==159
    before = {q['path']: q for q in old['queries']}
    assert len(d['queries'])==701 and {q['path'] for q in d['queries']}==set(before)
    identities = collections.defaultdict(set)
    for name, sha in link.EXPECTED.items():
        path = link.ASSETS/name
        assert link.digest(path)==sha
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith('/'): continue
                ids, _ = extract(z.read(name))
                for ident in ids: identities[ident].add(Path(name).stem)
    classes = [PngImagePlugin.PngImageFile,JpegImagePlugin.JpegImageFile,TiffImagePlugin.TiffImageFile]
    methods = [cls.load for cls in classes]
    def forbid(*args, **kwargs): raise RuntimeError('Pixel decoding forbidden')
    for cls in classes: cls.load=forbid
    reasons = collections.defaultdict(set)
    try:
        for q in d['queries']:
            name=q['path']
            if name not in pending: assert q==before[name]
            else:
                row=downloads[name];raw=(link.ROOT/row['local_path']).read_bytes();planned=plans[name]
                assert len(raw)==planned['size']==row['bytes']
                assert hashlib.sha256(raw).hexdigest()==q['original_sha256']==row['sha256']
                if 'lfs' in planned: assert hashlib.sha256(raw).hexdigest()==planned['lfs']['oid']
                else: assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==planned['oid']
                assert q['roles']==before[name]['roles'] and q['local_original']==row['local_path']
                try:
                    packets, extra, fmt=link.metadata_packets(link.ROOT/row['local_path'])
                    ids=set();unique=canonical('image_unique',extra)
                    if unique: ids.add(unique)
                    for packet in packets: found,_=extract(packet);ids.update(found)
                except (ValueError,OSError,AssertionError) as error:
                    assert q['status']=='metadata_error' and q['error_kind']==type(error).__name__
                    assert not q['candidate_ids'] and not q['possible_groups'];continue
                assert q['format']==fmt and q['xmp_packets']==len(packets) and q['valid_identifier_tokens']==len(ids)
                matches=[identities[i] for i in ids if i in identities]
                common=set.intersection(*matches) if matches else set()
                status='conflict' if matches and not common else 'unique' if len(common)==1 else 'multiple' if common else 'none'
                possible=set.union(*matches) if status=='conflict' else common
                assert q['status']==status and q['candidate_ids']==sorted(possible)
                assert q['possible_groups']==sorted({int(x.split('_')[0]) for x in possible})
            for g in q['possible_groups']: reasons[g].add(name)
    finally:
        for cls,method in zip(classes,methods,strict=True): cls.load=method
    assert d['group_reasons']==[dict(group=g,queries=sorted(paths)) for g,paths in sorted(reasons.items())]
    assert {x['group'] for x in old['group_reasons']}.issubset(reasons)
    for row, previous in zip(d['training_candidates'],old['training_candidates'],strict=True):
        expected=dict(previous,exclude_for_known_consumed_group=previous['ppr_group'] in reasons)
        assert row==expected
    excluded=sum(r['exclude_for_known_consumed_group'] for r in d['training_candidates'])
    assert d['summary']['query_status']==dict(collections.Counter(q['status'] for q in d['queries']))
    assert d['summary']['supplemented_status']==dict(collections.Counter(q['status'] for q in d['queries'] if q['path'] in pending))
    assert d['summary']['known_excluded_groups']==len(reasons)
    assert d['summary']['training_candidates_excluded']==excluded
    assert d['summary']['training_candidates_provisionally_remaining']==537-excluded
    assert d['finished'] and not d['training_started'] and not d['private_identifiers_exported'] and not d['summary']['full_training_eligibility_established']
    result=dict(finished=True,results_sha256=link.digest(OUT/'results.json'),unchanged_queries_verified=542,supplemented_queries_verified=159,group_sets_rebuilt=True,direct_string_matching=True,pixel_load_guard_passed=True,summary=d['summary'])
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__': main()
