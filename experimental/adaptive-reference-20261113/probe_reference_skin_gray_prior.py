"""Compare source skin chroma with a frozen grayscale prediction."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

from audit_bg_real_skin_regions import BASE, ROOT, digest
from probe_bg_real_legacy_skin import ROLES, TOGGLE_SHA, protector
from probe_reference_skin_cold_signal import evaluate


def entries():
    docs = ROOT/'docs/research'
    toggle_path = docs/'2026-11-18-local-edit-toggle-evidence.json'
    bg_path = docs/'2026-11-18-bg-real-pixel-pilot-evidence.json'
    assert digest(toggle_path) == TOGGLE_SHA
    toggle = json.loads(toggle_path.read_bytes())
    swap = json.loads(bg_path.read_bytes())
    regions = {'people': {}, 'empty': {}, 'bg': {}}
    for key, row in toggle['cases'].items():
        group = 'people' if row['reference_person'] else 'empty'
        regions[group][key] = {}
        for role in ROLES:
            name = f'{role}-reference.png'
            regions[group][key][role] = (BASE/'local-edit-toggle-v1'/key/name, row['files'][name])
    for key, row in swap['cases'].items():
        regions['bg'][key] = {role: (BASE/'bg-real-pixel-inputs'/row['input'][role]['name'],
                                    row['input'][role]['sha256']) for role in ('C1', 'F1')}
    return regions


def rgb256(path, sha):
    assert digest(path) == sha
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert('RGB').resize((256, 256), Image.Resampling.BILINEAR))


def prepare(skin_source, out):
    started = time.monotonic()
    module = protector(skin_source)
    out.mkdir()
    manifest = {}
    for group in entries().values():
        for key, roles in group.items():
            for role, (path, sha) in roles.items():
                weights, _, _ = module._predict(rgb256(path, sha))
                assert weights.shape == (256, 256)
                mask = weights >= .8
                name = f'{key}-{role}.npy'
                np.save(out/name, mask, allow_pickle=False)
                manifest[name] = {'sha256': digest(out/name), 'rgb_sha256': sha, 'count': int(mask.sum())}
    assert len(manifest) == 26 and time.monotonic()-started <= 30
    (out/'masks.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


def estimate(mask_dir, out):
    import torch
    from probe_reference_colorization_prior import ASSETS, lightness_and_b, load_eccv

    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_eccv(ROOT/'.tools/ECCV16-colorization-audit')
    source_prior = BASE/'reference-colorization-prior-v1'
    cached = json.loads((source_prior/'results.json').read_bytes())
    assert cached['assets'] == ASSETS and cached['sources']['toggle'] == TOGGLE_SHA
    masks_file = mask_dir/'masks.json'
    masks = json.loads(masks_file.read_bytes())
    results = {}
    for group, photos in entries().items():
        results[group] = {}
        for key, roles in photos.items():
            records = {}
            for role, (path, sha) in roles.items():
                level, observed = lightness_and_b(rgb256(path, sha))
                mask_path = mask_dir/f'{key}-{role}.npy'
                row = masks[mask_path.name]
                assert row['rgb_sha256'] == sha and row['sha256'] == digest(mask_path)
                mask = np.load(mask_path, allow_pickle=False)
                assert mask.shape == (256, 256) and mask.dtype == bool and mask.sum() == row['count']
                if group == 'bg':
                    tensor = torch.from_numpy(level.astype(np.float32))[None, None].cuda()
                    with torch.inference_mode():
                        predicted = model(tensor)[0, 1].cpu().numpy()
                    saved = BytesIO()
                    np.save(saved, predicted, allow_pickle=False)
                    predicted_hash = sha256(saved.getvalue()).hexdigest()
                else:
                    prior_path = source_prior/f'{key}-{role}-prior-b.npy'
                    record = cached['first'][key][role]
                    predicted_hash = digest(prior_path)
                    assert record['rgb_sha256'] == sha and record['prior_sha256'] == predicted_hash
                    predicted = np.load(prior_path, allow_pickle=False)
                assert predicted.shape == level.shape and np.isfinite(predicted).all()
                count = int(mask.sum())
                records[role] = dict(skin_pixels=count, prior_sha256=predicted_hash,
                                     raw=float(np.median(-observed[mask])) if count >= 16 else None,
                                     residual=float(np.median((predicted-observed)[mask])) if count >= 16 else None)
            results[group][key] = records
    scores = {field: evaluate(results['people'], results['empty'], results['bg'], field)
              for field in ('raw', 'residual')}
    source = Path(__file__)
    assert time.monotonic()-started <= 60
    with out.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(**results, scores=scores, masks_sha256=digest(masks_file), assets=ASSETS,
                                     cached_report_sha256=digest(source_prior/'results.json'),
                                     source_sha256=digest(source), protocol_sha256=digest(source.with_name(
                                         'reference-skin-gray-prior-protocol.md'))),
                                ensure_ascii=False, indent=2)+'\n')
    print({field: [row[k] for k in ('hits','false_positives','empty_positives','bg_flips_count')]
           for field, row in scores.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('masks', 'score'))
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.phase == 'masks':
        prepare(args.source.resolve(), args.out)
    else:
        estimate(args.source, args.out)
