"""Probe how much of the held-out person spill is shared with the background."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from background_intervention import LUMA, target_regions
from independent_coco import ROOT, digest
from probe_local_edit_toggle import mask_for
from probe_reference_chroma_spread import PARENT, read
from reference_color_app.media import decode_image


def color_residual(delta):
    return delta-(delta@LUMA)[...,None]


def mean_map(chroma, background, bins):
    mean = chroma[background].mean(axis=0)
    by_bin = np.zeros((8,3))
    qualified = np.zeros(8,bool)
    for index in range(8):
        area = background & (bins==index)
        if area.sum()>=256:
            qualified[index] = True
            by_bin[index] = chroma[area].mean(axis=0)
    return mean, by_bin[bins], qualified[bins], int(qualified.sum())


def run(source, out):
    docs = ROOT/'docs/research'
    evidence = docs/'2026-11-18-bg-real-pixel-pilot-evidence.json'
    prior = json.loads(evidence.read_bytes())
    manifest = PARENT/'results.json'
    assert prior['parent_sha256']==digest(manifest) and prior['eligible']==4
    index = {case['pair']:case for case in json.loads(manifest.read_bytes())['cases']}
    out.mkdir()
    report = {}
    for key,row in prior['cases'].items():
        entry = index[row['target_pair']]
        target = read(entry['input_file'])
        person = mask_for(entry,row['target_pair'].split('-')[0])
        files = [source/f'{key}-{role}.png' for role in ('C1','F1')]
        assert [digest(path) for path in files]==[row['outputs'][role] for role in ('C1','F1')]
        c1,f1 = (decode_image(path) for path in files)
        difference = f1.astype(float)-c1
        chroma = color_residual(difference)
        bins = ((target@LUMA)//32).astype(int).clip(0,7)
        mean, field, support, bin_count = mean_map(chroma,~person,bins)
        compare_on = person&support
        neutral = target_regions(target,person)['neutral_proxy'] & person
        regions = {'person':person, 'neutral_person':neutral}
        assert len(chroma[compare_on]) and neutral.any()
        checks = {}
        files = {}
        for name, shift in (('shared',np.broadcast_to(mean,chroma.shape)),('luma_bands',field)):
            change = shift[compare_on]
            np.testing.assert_allclose(change@LUMA,0,atol=1e-10)
            raw = f1.astype(float)
            raw[compare_on] -= change
            output = np.rint(raw.clip(0,255)).astype(np.uint8)
            file = out/f'{key}-{name}.png'
            Image.fromarray(output).save(file)
            files[name] = digest(file)
            delta = output.astype(float)-c1
            for scope, area in regions.items():
                partial = area&support
                if not partial.any():
                    checks.setdefault(scope,{})[name] = dict(coverage=0.,pixels=0)
                    continue
                base = chroma[partial]
                after = color_residual(delta[partial])
                base_mae = float(np.abs(base).mean())
                new_mae = float(np.abs(after).mean())
                power = float(np.mean(base**2))
                checks.setdefault(scope,{})[name] = dict(
                    coverage=float(partial.sum()/area.sum()), pixels=int(partial.sum()), initial=base_mae,
                    residual=new_mae, gain=1-new_mae/base_mae if base_mae else None,
                    energy_explained=1-float(np.mean(after**2))/power if power else None,
                    initial_luma_mae=float(np.abs(difference[partial]@LUMA).mean()),
                    output_luma_mae=float(np.abs(delta[partial]@LUMA).mean()),
                    clipped_pixels=int(((raw[partial]<0)|(raw[partial]>255)).any(-1).sum()))
        main = checks['neutral_person']['luma_bands']
        safe = main.get('residual',0.)<=main.get('initial',0.)+1
        passed = bool(safe and main['coverage']>=.8 and main.get('gain',0.)>=.4)
        report[key] = dict(target_pair=row['target_pair'], bins=bin_count, scopes=checks,
                           output_sha256=files, passed=passed, safe=safe)
    passed = sum(row['passed'] for row in report.values())
    result = dict(cases=report, passed=passed, proceeded=passed>=3 and all(row['safe'] for row in report.values()),
                  input_evidence_sha256=digest(evidence), source_sha256=digest(Path(__file__)))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({row['target_pair']:row['scopes']['neutral_person']['luma_bands'] for row in report.values()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    run(args.source,args.out)
