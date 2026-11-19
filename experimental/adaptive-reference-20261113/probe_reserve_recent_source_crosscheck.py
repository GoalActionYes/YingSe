"""Cross-check recent consumed photo sources against sealed input identities."""
import argparse
from pathlib import Path

from audit_bg_real_reserve import BASE, ROOT, pinned
from probe_chromamill_target_light_cross import PARENT_SHA
from probe_pst50_reserve_fingerprints import distance, fingerprint
from qualify_bg_real_reserve import read_sources, reserve
from probe_target_mirror_chroma import json_to

PST_SHA = '0aff7aa25264c620acbba339a211e23b9133fef15968b42212b8542fff0874bc'
BG_SHA = '134ac49ed3a25e6116af1fa8f7061d2dedf77fc849d99d7ac2caa99b0f36d181'
PS_SHA = '6dbe4870e44c81bd7614a779febb7c1806a51662d5e5d25c675e91280b4c540b'
PS_LOCAL_SHA = '9453038594bbd83b76fe51b2510b200608d123307e898b3fba65f33ef81e07ab'
COCO_SHA = '13cd9d6b043666959340e4f479b6297dc0ddcd91da0b7b31ae92b1d78cee1a01'


def consumed():
    capture = BASE / 'chromamill-capture-cross-v1'
    prepared = pinned(capture / 'results.json', PARENT_SHA)['prepared']
    chroma = [(capture / f'{scene}-{setup}-off.png', row['hashes']['off'])
              for scene, setups in prepared.items() for setup, row in setups.items()]
    desaturation = BASE / 'psbench-background-desaturation-assets-v1'
    files = pinned(BASE / 'psbench-background-desaturation-v1/results.json', PS_SHA)['input_files']
    photoshop = [(desaturation / name, row['sha256']) for name, row in files.items()]
    evidence = ROOT / 'docs/research'
    contrast = BASE / 'psbench-local-intent-pair-v1'
    known = pinned(evidence / '2026-11-19-psbench-local-intent-pair-evidence.json', PS_LOCAL_SHA)['source_checks']
    photoshop += [(contrast / f"{case['task']}_{state}.jpg", receipt['sha256']) for case in known
                  for state, receipt in zip(('start', 'end'), case['files'], strict=True)]
    source = pinned(evidence / '2026-11-19-screen-occlusion-intake-evidence.json', COCO_SHA)['photos']
    coco = [(BASE / 'screen-occlusion-intake' / f"{int(row['id']):012d}.jpg", row['sha256']) for row in source]
    return {'chromamill': chroma, 'psbench': photoshop, 'coco': coco}


def reserved():
    pst = pinned(BASE / 'pst50-reserve-fingerprint-v1/results.json', PST_SHA)
    if pst['isolated'] or len(pst['fingerprints']) != 35:
        raise ValueError('Changed PST50 reserve state')
    cached = {(int(group), role): stamp for group, roles in pst['fingerprints'].items()
              for role, stamp in roles.items()}
    directory = BASE / 'bg-real-reserve-v2'
    qualification = pinned(directory / 'technical-final.json', BG_SHA)
    queue = reserve(directory / 'reserved-manifest.json')
    files, _ = read_sources(queue, directory / 'sealed-assets')
    bg = {key: fingerprint(path) for key, path in files.items() if key[1] != 'mask'}
    if len(bg) != 24 or len(cached) != 70 or len(qualification['technical_pairs']) != 10:
        raise ValueError('Changed sealed input counts')
    return {'bg_real': bg, 'pst50': cached}


def compare(prior, candidates):
    results = {}
    for label, inputs in prior.items():
        gap, checks, alerts = 64, 0, []
        for source in inputs:
            for (group, role), candidate in candidates.items():
                checks += 1
                score = distance(source['fingerprints'], candidate['fingerprints'])
                gap = min(gap, score)
                exact = (source['bytes_sha256'] == candidate['bytes_sha256']
                         or source['pixels_sha256'] == candidate['pixels_sha256'])
                if exact or score <= 6:
                    alerts.append(dict(group=group, role=role, exact=exact, distance=score))
        results[label] = dict(compared=checks, min_distance=gap, alerts=alerts)
    return results


def run(out):
    old = consumed()
    if {key:len(items) for key, items in old.items()} != {'chromamill':8, 'psbench':8, 'coco':9}:
        raise ValueError('Unexpected audit cohort')
    images = {}
    for name, rows in old.items():
        images[name] = []
        for path, want in rows:
            value = fingerprint(path)
            if value['bytes_sha256'] != want:
                raise ValueError('Changed consumed source identity')
            images[name].append(value)
    summary = {name: compare(images, rows) for name, rows in reserved().items()}
    out.mkdir()
    json_to(out / 'results.json', dict(summary=summary, prior_counts={k: len(v) for k,v in images.items()},
                                      pst_manifest_sha256=PST_SHA,bg_qualification_sha256=BG_SHA,models_run=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
