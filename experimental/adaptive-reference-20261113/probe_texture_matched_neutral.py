"""Screen a local chroma constraint using reference patches matched in grayscale."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from background_intervention import target_regions
from probe_bg_real_style_cross import compare, cool, gates, write
from probe_facd_filter_bank import scores as facd_scores
from probe_facd_two_image_baseline import STYLES, read_checked
from probe_gray_edge_correction import PARENT_SHA, authored, controlled as background_controls
from probe_local_chroma_redistribution import PARENT_REPORT
from probe_local_edit_toggle import mask_for
from probe_neutral_relation_balance import PSBENCH_SHA, controlled as local_controls
from probe_reference_chroma_spread import BASE, PARENT, ROOT, digest, read
from probe_roundtrip_grade_choice import checked_json
from probe_sparks_shared_chroma import LUMA, chroma, render
from probe_target_hue_retention import summarize
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


FEATURE_SCALE = np.array([24., 12., 8., 8.])
ROLES = ('C1', 'F1', 'C1-cool', 'F1-cool')


def blocks(array):
    return array.reshape(16,16,16,16,*array.shape[2:]).swapaxes(1,2).reshape(256,16,16,*array.shape[2:])


def describe(rgb):
    thumb = np.array(Image.fromarray(rgb).resize((256,256),Image.Resampling.BILINEAR))
    light = thumb @ LUMA
    tile = blocks(light)
    feature = np.column_stack((tile.mean((1,2)),tile.std((1,2)),
                               np.abs(np.diff(tile,axis=1)).mean((1,2)),
                               np.abs(np.diff(tile,axis=2)).mean((1,2)))) / FEATURE_SCALE
    tone = blocks(chroma(thumb,light)).mean((1,2))
    return feature,tone


def upsample(plane, shape):
    scale = (shape[1],shape[0])
    image = Image.fromarray(plane.reshape(16,16).astype(np.float32))
    return np.array(image.resize(scale,Image.Resampling.BILINEAR),dtype=np.float64)


class TextureProbe:
    def measure(self,target,reference,baseline):
        if np.array_equal(target,reference):
            return target.copy(),dict(support=0.,limited=0.,max_luma_error=0.,same_image=True)
        f,c = describe(target)
        g,d = describe(reference)
        distance = np.square(f[:,None,:]-g[None,:,:]).sum(-1)
        neighbors = np.argsort(distance,axis=1,kind='stable')[:,:4]
        good = distance[np.arange(len(f)),neighbors[:,-1]] <= 1.5**2
        matched = np.median(d[neighbors],axis=1)-c
        shape = target.shape[:2]
        support = upsample(good,shape)
        weighted = np.stack([upsample(matched[:,axis]*good,shape) for axis in (0,1)],axis=-1)
        evidence = np.divide(weighted,support[...,None],out=np.zeros_like(weighted),where=support[...,None]>0)
        y,base_y = target@LUMA,baseline@LUMA
        original = chroma(target,y)
        delta = chroma(baseline,base_y)-original
        denominator = np.square(delta).sum(-1)
        fraction = np.divide((evidence*delta).sum(-1),denominator,
                             out=np.ones_like(denominator),where=denominator>0).clip(0.,1.)
        strength = np.clip((24-np.ptp(target,axis=-1).astype(float))/12,0.,1.)
        retain = 1-support*strength*(1-fraction)
        rgb,_ = render(original+retain[...,None]*delta,base_y)
        rgb[retain==1] = baseline[retain==1]
        y_error = (rgb.astype(np.int32)-baseline) @ np.array([2126,7152,722])
        return rgb,dict(support=float(good.mean()),limited=float((retain<1).mean()),
                        max_luma_error=float(np.abs(y_error).max()/10000),same_image=False)


def crossed(probe,out,cohort,docs):
    source = json.loads((docs/'2026-11-18-bg-real-style-cross-evidence.json').read_bytes())
    before = docs/'2026-11-18-bg-real-pixel-pilot-evidence.json'
    if digest(before) != source['source_evidence_sha256'] or source['parent_sha256'] != PARENT_SHA:
        raise ValueError('Unexpected crossed cooling cohort')
    inputs = json.loads(before.read_bytes())['cases']
    folder = BASE/'bg-real-style-cross'
    output = {}
    for pair,record in source['cases'].items():
        row = cohort[pair]
        target = read(row['input_file'])
        pictures, identities = {}, {}
        for role in ROLES:
            if role in ('C1','F1'):
                entry = inputs[record['source']]['input'][role]
                path,sha = BASE/'bg-real-pixel-inputs'/entry['name'],entry['sha256']
            else:
                path,sha = folder/f'{pair}-ref-{role}.png',record['cool_reference_sha256'][role]
            current = folder/f'{pair}-{role}.png'
            if digest(path) != sha or digest(current) != record['output_sha256'][role]:
                raise ValueError(f'Unexpected control input: {pair}/{role}')
            rgb,stats = probe.measure(target,decode_image(path),decode_image(current))
            pictures[role] = rgb
            identities[role] = dict(stats,sha256=write(out/f'{pair}-{role}.png',rgb))
        human = mask_for(row,pair.split('-')[0])
        masks = target_regions(target,human)
        masks['neutral_person'] = masks['neutral_proxy'] & human
        expected = finish_spatial(target,cool(target),.8).astype(np.int16)-target
        measured = {k:compare(pictures,expected,mask) for k,mask in masks.items()}
        checks = gates(measured)
        output[pair] = dict(metrics=measured,checks=checks,
                            passed=all(v for v in checks.values() if v is not None),outputs=identities)
    return output


def facd(probe,out,docs):
    verified_file = docs/'2026-11-18-facd-border-registration-evidence.json'
    verified = json.loads(verified_file.read_bytes())
    prior = json.loads((docs/'2026-11-18-facd-two-image-baseline-evidence.json').read_bytes())
    if len(prior['references']) != 7 or digest(verified_file) != prior['data_sha256']:
        raise ValueError('Unexpected FACD data')
    aligned = BASE/'facd-border-registration-v1'
    baseline = BASE/'facd-two-image-baseline-v1'
    result = {}
    for family,refs in prior['references'].items():
        target = read_checked(aligned,family,'Origin',verified)
        folder = out/family
        folder.mkdir()
        variants,identities = {},{}
        for style in STYLES:
            variants[style] = {}
            for ref in refs:
                leaf = f'{family}/{ref}-{style}.png'
                path = baseline/leaf
                if digest(path) != prior['artifacts'][leaf]:
                    raise ValueError(f'Unexpected FACD baseline: {leaf}')
                reference = read_checked(aligned,ref,style,verified)
                rgb,stats = probe.measure(target,reference,decode_image(path))
                variants[style][ref] = rgb.astype(np.int16)
                identities[f'{ref}-{style}'] = dict(stats,sha256=write(out/leaf,rgb))
        applied = {style:finish_spatial(target,read_checked(aligned,family,style,verified),.8).astype(np.int16)
                   for style in STYLES}
        report = facd_scores(target,applied,variants,refs)
        result[family] = dict(report,outputs=identities,passed=all(report['gates'].values()))
    return result


def run(out):
    start = time.monotonic()
    docs = ROOT/'docs/research'
    frozen = checked_json(PARENT/'results.json',PARENT_SHA)
    toggle = checked_json(docs/'2026-11-18-local-edit-toggle-evidence.json',PARENT_REPORT)
    ps = docs/'2026-11-18-psbench-action-pair-evidence.json'
    if digest(ps)!=PSBENCH_SHA or toggle['parent_sha256']!=PARENT_SHA or frozen['strength']!=.8:
        raise ValueError('Unexpected frozen controls')
    probe = TextureProbe()
    sample = np.arange(48,dtype=np.uint8).reshape(4,4,3)
    np.testing.assert_array_equal(probe.measure(sample,sample.copy(),255-sample)[0],sample)
    out.mkdir()
    for label in ('background','local','authored','crossed','facd'):
        (out/label).mkdir()
    bg = background_controls(out/'background',frozen,probe.measure)
    cohort = {r['pair']:r for r in frozen['cases']}
    local = local_controls(probe,out/'local',toggle,cohort)
    authored_rows = authored(out/'authored',BASE/'psbench-shared-local-v1',ps,probe.measure)
    cross = crossed(probe,out/'crossed',cohort,docs)
    styles = facd(probe,out/'facd',docs)
    summary = summarize(bg,local,authored_rows)
    summary['crossed_groups'] = sum(r['passed'] for r in cross.values())
    summary['facd_groups'] = sum(r['passed'] for r in styles.values())
    additional = [r['max_luma_error'] for cohorts in (cross,styles) for row in cohorts.values()
                  for r in row['outputs'].values()]
    summary['max_luma_error'] = max([summary['max_luma_error'],*additional])
    summary['passed'] &= summary['crossed_groups']>=3 and summary['facd_groups']>=6 and summary['max_luma_error']<=.5
    if time.monotonic()-start>120:
        raise RuntimeError('Compute budget exceeded')
    source = Path(__file__)
    parent_files = ('bg-real-style-cross','facd-two-image-baseline')
    data = dict(summary=summary, background=bg,local=local,authored=authored_rows,crossed=cross,facd=styles,
                parent_sha256=PARENT_SHA,parents={name:digest(docs/f'2026-11-18-{name}-evidence.json') for name in parent_files},
                source_sha256=digest(source),protocol_sha256=digest(source.with_name('texture-matched-neutral-support-protocol.md')))
    (out/'results.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    run(parser.parse_args().out)
