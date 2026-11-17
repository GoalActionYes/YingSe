"""Supplement only 159 previously unknown originals; never decode pixels or train."""
import collections
import copy
import json
from pathlib import Path
import time
import zipfile
from PIL import PngImagePlugin, JpegImagePlugin, TiffImagePlugin
import link_ppr_training_metadata as link
from fetch_ppr_family_originals import verify

BASE = link.ROOT / 'outputs/adaptive-reference-20261113'
PARENT = BASE / 'ppr-group-exclusions-v2/results.json'
DOWNLOAD = BASE / 'ppr-family-originals-v1/download-results.json'
PLAN = BASE / 'ppr-family-originals-v1/plan.json'
OUT = BASE / 'ppr-group-exclusions-v3'
PARENT_SHA = '8f7fe25dbb2a4c43e09c577a5742d6a48fbf2e091c06ae6b3e017a69599976a5'
DOWNLOAD_SHA = '29c50a7e2144f2598160bede2a5cd8c4e5376620d75488965be79e184ea4e752'
PLAN_SHA = '8aa1de9f34a8d3fe4f1bcaadb4df2b86ca83504826fcacae44007b1bcab26729'


def main():
    start = time.perf_counter()
    assert link.digest(PARENT) == PARENT_SHA
    assert link.digest(DOWNLOAD) == DOWNLOAD_SHA
    assert link.digest(PLAN) == PLAN_SHA
    old = json.loads(PARENT.read_text())
    assert old['finished'] and link.digest(Path(link.__file__)) == old['source_sha256']['link_ppr_training_metadata.py']
    downloaded = json.loads(DOWNLOAD.read_text())
    assert downloaded['finished'] and downloaded['plan_sha256'] == PLAN_SHA
    plan = {r['path']: r for r in json.loads(PLAN.read_text())['files']}
    files = {r['path']: r for r in downloaded['files']}
    pending = {q['path'] for q in old['queries'] if q['status'] == 'missing_registered_identity'}
    assert len(pending) == 159 and pending == set(plan) == set(files)
    OUT.mkdir(exist_ok=False)
    names = ['complete_ppr_family_links.py', 'verify_completed_ppr_family_links.py', 'ppr-family-originals-protocol.md', 'link_ppr_training_metadata.py', 'fetch_ppr_family_originals.py']
    sources = {name: link.digest(Path(__file__).with_name(name)) for name in names}
    for name in names:
        (OUT / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    mapping = collections.defaultdict(set)
    for name, sha in link.EXPECTED.items():
        path = link.ASSETS / name
        assert link.digest(path) == sha
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith('/'): continue
                ids, _ = link.xml_info(z.read(name))
                for ident in ids: mapping[ident].add(Path(name).stem)
    classes = [PngImagePlugin.PngImageFile, JpegImagePlugin.JpegImageFile, TiffImagePlugin.TiffImageFile]
    methods = [cls.load for cls in classes]
    def forbid(*args, **kwargs): raise RuntimeError('Pixel decoding forbidden')
    for cls in classes: cls.load = forbid
    queries = []
    try:
        for q in old['queries']:
            assert time.perf_counter() - start < 180
            if q['path'] not in pending:
                queries.append(copy.deepcopy(q)); continue
            row = files[q['path']]
            path = link.ROOT / row['local_path']
            assert path.resolve().is_relative_to((link.ASSETS / 'family-originals').resolve())
            assert verify(path, plan[q['path']]) == row['sha256']
            base = dict(path=q['path'], roles=q['roles'], original_sha256=row['sha256'], local_original=row['local_path'])
            try:
                ids, _, meta = link.image_info(path)
            except (ValueError, OSError, AssertionError) as error:
                queries.append(dict(base, status='metadata_error', error_kind=type(error).__name__, candidate_ids=[], possible_groups=[])); continue
            matched = [mapping[i] for i in ids if i in mapping]
            common = set.intersection(*matched) if matched else set()
            status = 'conflict' if matched and not common else 'unique' if len(common) == 1 else 'multiple' if common else 'none'
            possible = set.union(*matched) if status == 'conflict' else common
            queries.append(dict(base, **meta, status=status, candidate_ids=sorted(possible), possible_groups=sorted({int(i.split('_')[0]) for i in possible})))
    finally:
        for cls, method in zip(classes, methods, strict=True): cls.load = method
    reasons = collections.defaultdict(list)
    for q in queries:
        for group in q['possible_groups']: reasons[group].append(q['path'])
    assert {r['group'] for r in old['group_reasons']}.issubset(reasons)
    training = copy.deepcopy(old['training_candidates'])
    for row in training: row['exclude_for_known_consumed_group'] = row['ppr_group'] in reasons
    excluded = sum(row['exclude_for_known_consumed_group'] for row in training)
    summary = dict(queries=len(queries), supplemented_originals=159, query_status=dict(collections.Counter(q['status'] for q in queries)), supplemented_status=dict(collections.Counter(q['status'] for q in queries if q['path'] in pending)), known_excluded_groups=len(reasons), training_candidates=537, training_candidates_excluded=excluded, training_candidates_provisionally_remaining=537-excluded, full_training_eligibility_established=False)
    assert len(queries) == 701 and len(training) == 537
    report = dict(parent_sha256=PARENT_SHA, download_sha256=DOWNLOAD_SHA, plan_sha256=PLAN_SHA, source_sha256=sources, queries=queries, group_reasons=[dict(group=g, queries=sorted(paths)) for g, paths in sorted(reasons.items())], training_candidates=training, summary=summary, finished=True, training_started=False, private_identifiers_exported=False, pixel_load_guard_passed=True, seconds=time.perf_counter()-start)
    assert all(link.digest(Path(__file__).with_name(name)) == sha for name, sha in sources.items())
    (OUT/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(summary=summary, seconds=report['seconds'])))


if __name__ == '__main__': main()
