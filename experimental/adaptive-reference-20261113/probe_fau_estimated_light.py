"""Test a frozen RGB-only lighting estimator on measured FAU relighting pairs."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
from PIL import Image

from probe_intrinsic_prior import ROOT, ASSETS, digest
from audit_fau_two_light_intake import ARCHIVE_SHA, native_rgb, sha
from background_intervention import target_regions
from probe_fau_exposure_control import bounds_mask
from probe_fau_illuminant_replay import replay
from probe_fau_two_light_response import SCENES, chroma_change, evaluate, member
from probe_intrinsic_intervention import FIRST
from reference_color_app.media import decode_image
from spatial_render import finish_spatial


def predictor():
    for package in (ROOT/'.tools/DeepWB-audit/vendor', ROOT/'.tools/StyleGallery-audit/vendor'):
        sys.path.insert(0, str(package))
    import diffusers
    import torch
    from diffusers import MarigoldIntrinsicsPipeline

    prior = json.loads((FIRST/'results.json').read_bytes())
    manifests = json.loads((ASSETS/'assets.json').read_bytes())
    assert prior['finished'] and prior['assets'] == manifests
    for record in manifests['files']:
        assert digest(ASSETS/'models'/record['path']) == record['sha256']
    implementation = Path(diffusers.__file__).parent/'pipelines/marigold'
    assert all(digest(implementation/name) == expected for name,expected in prior['implementation_sha256'].items())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipeline = MarigoldIntrinsicsPipeline.from_pretrained(
        ASSETS/'models', variant='fp16', torch_dtype=torch.float16,
        local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True).to('cuda')
    assert list(pipeline.config.target_properties['target_names']) == ['albedo','shading','residual']
    pipeline.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def predict(rgb):
        value = pipeline(
            Image.fromarray(rgb), num_inference_steps=4, ensemble_size=1,
            processing_resolution=768, batch_size=1, output_type='np',
            generator=torch.Generator(device='cuda').manual_seed(20261117),
        ).prediction
        assert value.shape == (3, *rgb.shape) and np.isfinite(value).all()
        return value

    bridge = decode_image(FIRST/'input.png')
    np.testing.assert_array_equal(predict(bridge), np.load(FIRST/prior['runs'][0]['file'],allow_pickle=False))
    return predict


def estimated_ratio(target, reference):
    neutralized = np.maximum(reference, 1e-4).astype(float)/np.maximum(target, 1e-4)
    return neutralized / neutralized[..., 1, None]


def run(path, inputs, out):
    assert digest(path) == ARCHIVE_SHA
    docs = ROOT/'docs/research'
    source_evidence = docs/'2026-11-18-fau-exposure-evidence.json'
    asset_evidence = docs/'2026-11-18-fau-two-light-evidence.json'
    exposure = json.loads(source_evidence.read_bytes())
    archive_ids = json.loads(asset_evidence.read_bytes())['response']
    assert exposure['prior_sha256'] == digest(asset_evidence)
    predict = predictor()
    out.mkdir()
    pending, cases = {}, {}
    for scene in SCENES:
        inputs_rgb, shading, predicted = [], [], {}
        for role in ('target','reference'):
            file = inputs/f'{scene}-{role}.png'
            assert digest(file) == exposure['predictions'][scene][role]
            rgb = decode_image(file)
            inputs_rgb.append(rgb)
            shade = predict(rgb)[1]
            shading.append(shade)
            saved = out/f'{scene}-{role}-shade.npy'
            np.save(saved,shade,allow_pickle=False)
            predicted[role] = digest(saved)
        target, reference = inputs_rgb
        shape_mask = np.ones(target.shape[:2], bool)
        ratio = estimated_ratio(*shading).reshape(-1,3)
        assert np.array_equal(replay(target, np.ones(3), shape_mask)[0], target)
        records, output = {}, {}
        for mode, gain in (('global', np.exp(np.log(ratio).mean(axis=0))),('local',ratio)):
            image, clipped = replay(target, gain, shape_mask)
            file = out/f'{scene}-{mode}.png'
            Image.fromarray(image).save(file)
            records[mode] = dict(sha256=digest(file), clipped=clipped)
            output[mode] = image
        pending[scene] = (target, reference, shading, output)
        cases[scene] = dict(shading=predicted, modes=records)

    with tarfile.open(path, 'r:gz') as archive:
        for scene, (target, reference, shading, outputs) in pending.items():
            artifacts = {}
            for role in ('img','masks','groundtruth'):
                content = [member(archive, role, scene, side) for side in ('b_r','r_b')]
                key = 'raw_sha256' if role == 'img' else f'{role}_sha256'
                assert [sha(item) for item in content] == archive_ids['cases'][scene][key]
                artifacts[role] = [native_rgb(item,260,452) if role=='img' else np.array(Image.open(io.BytesIO(item)))
                                   for item in content]
            labels = artifacts['groundtruth']
            valid = bounds_mask((target,reference))
            for mask, label in zip(artifacts['masks'],labels):
                valid &= (mask == 255) & (label > 0).all(axis=-1)
            regions = target_regions(target,np.zeros(valid.shape,bool))
            desired = finish_spatial(target,reference,.8)
            row = cases[scene]
            for mode, output in outputs.items():
                row['modes'][mode]['scores'] = {
                    key: evaluate(regions[key],valid,artifacts['img'],labels,target,desired,output)
                    for key in ('whole','neutral_proxy')
                }
            modeled = chroma_change(*[np.maximum(value[valid],1e-4) for value in shading])
            measured = chroma_change(*[value[valid] for value in labels])
            norm = float(np.linalg.norm(modeled)*np.linalg.norm(measured))
            row['relative_label_cosine'] = float(np.sum(modeled*measured)/norm) if norm else None
            row['floor_fractions'] = [float((value[valid] < 1e-4).mean()) for value in shading]
            uniform, local = [row['modes'][mode]['scores']['neutral_proxy']['output_chroma_rms_error']
                              for mode in ('global','local')]
            row['neutral_gain'] = float(1-local/uniform)
            row['passed'] = row['neutral_gain']>=.15 and all(
                item['passed'] for item in row['modes']['local']['scores'].values())
    report = dict(cases=cases, proceeded=all(row['passed'] for row in cases.values()),
                  exposure_sha256=digest(source_evidence), archive_sha256=ARCHIVE_SHA,
                  estimator_manifest_sha256=digest(ASSETS/'assets.json'),
                  source_sha256=digest(Path(__file__)))
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key: dict(cosine=row['relative_label_cosine'], gain=row['neutral_gain'], passed=row['passed'])
                      for key,row in cases.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('path','inputs','out'):
        parser.add_argument(argument,type=Path)
    run(**vars(parser.parse_args()))
