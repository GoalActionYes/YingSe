"""Offline source/state/output audit; GT is used only after producer finished."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

from ictone_flux_stage import ROOT, ASSETS, meta_checkpoint, source_group
from screen_ictone_first import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    assert Path(args.run).name == args.run and args.run not in ('.', '..')
    base = ROOT/'outputs/adaptive-reference-20261113/tst100k'
    out = base/args.run
    report = json.loads((out/'results.json').read_text())
    assert report['finished'] and report['completed_steps']==4
    for name, sha in report['source_sha256'].items():
        assert digest(out/name)==sha==digest(Path(__file__).with_name(name))
    import numpy as np
    import torch
    import gguf
    from PIL import Image, ImageDraw
    from safetensors import safe_open
    from safetensors.torch import load_file
    from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
    from diffusers.quantizers.gguf.utils import GGUFParameter
    metadata = json.loads((ROOT/'outputs/adaptive-reference-20261113/ictone-quantized-metadata-v1/results.json').read_text())
    staged = json.loads((out/'stage.json').read_text())
    reader = gguf.GGUFReader(str(ASSETS/'models/fill/flux1-fill-dev-Q3_K_S.gguf'))
    tensors = {x.name:x for x in reader.tensors}
    checked = set()
    with (ASSETS/'models/fill/flux1-fill-dev-Q3_K_S.gguf').open('rb') as handle, safe_open(
            str(ASSETS/'models/lora/pytorch_lora_weights.safetensors'),framework='pt',device='cpu') as lora:
        for group in staged['groups']:
            prefixes = {source_group(x['name']) for x in group['state']}
            assert len(prefixes)==1
            prefix = prefixes.pop()
            shadow = meta_checkpoint(metadata)
            for name,item in tensors.items():
                selected = (not name.startswith(('double_blocks.','single_blocks.')) if prefix=='root' else name.startswith(prefix))
                if selected:
                    handle.seek(item.data_offset)
                    value = torch.from_numpy(np.frombuffer(handle.read(item.n_bytes),dtype=item.data.dtype).reshape(item.data.shape).copy())
                    shadow[name] = GGUFParameter(value,quant_type=item.tensor_type) if item.tensor_type==gguf.GGMLQuantizationType.Q3_K else value
            converted = convert_flux_transformer_checkpoint_to_diffusers(shadow)
            saved = load_file(str(out/'groups'/group['file']))
            assert set(saved)=={x['key'] for x in group['state']}
            for row in group['state']:
                name = row['name']
                assert name not in checked
                if '.lora_A.' in name or '.lora_B.' in name:
                    expected = lora.get_tensor('transformer.'+name.replace('.ictone.','.'))
                else:
                    expected = converted[name.replace('.base_layer.','.')]
                if isinstance(expected,GGUFParameter):
                    assert int(expected.quant_type)==row['quant_type'] and list(expected.quant_shape)==row['shape']
                    expected = expected.as_tensor()
                else:
                    expected = expected.to(torch.bfloat16)
                value = saved[row['key']]
                assert torch.equal(value,expected) and list(value.shape)==row['stored_shape'],name
                assert hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==row['sha256']
                checked.add(name)
            del saved,converted,shadow,value,expected
            gc.collect()
    assert len(checked)==staged['parameter_count']==report['checked_parameter_count']==1846
    for name,sha in report['output_sha256'].items():
        assert digest(out/name)==sha
    with Image.open(out/'triptych-output.png') as im:
        assert im.size==(1536,512)
        crop=im.crop((1024,0,1536,512))
    with Image.open(out/'native512.png') as im:
        assert np.array_equal(np.array(crop),np.array(im))
    metric = crop.resize((256,256),Image.Resampling.BILINEAR)
    with Image.open(out/'metric256.png') as im:
        assert np.array_equal(np.array(metric),np.array(im))
    data = base/'cohort-training-coverage-v2'
    row=json.loads((data/'cohort.json').read_text())['groups']['validation'][0]
    assert row['index']==22927
    lookup={x['original_path']:x for x in json.loads((data/'srgb-v1/manifest.json').read_text())['files']}
    def read(role):
        item=lookup[row[role]]
        p=data/item['prepared_path']
        assert p.resolve().is_relative_to(base.resolve()) and digest(p)==item['sha256']
        with Image.open(p) as im:return im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
    for role in ['content','reference']:
        with Image.open(out/(role+'256.png')) as im:
            assert np.array_equal(np.array(im),np.array(read(role)))
    truth=np.array(read('gt')).astype(float)
    with Image.open(base/'multiref-control-v1/epoch-6/22927.png') as im:baseline=im.copy()
    prior=json.loads((base/'training-coverage-expanded-v2/results.json').read_text())['baseline']
    old=next(x for x in prior['cases'] if x['index']==22927)
    assert digest(base/'multiref-control-v1/epoch-6/22927.png')==old['sha256']
    baseline_mae=float(np.abs(np.array(baseline).astype(float)-truth).mean())
    candidate_mae=float(np.abs(np.array(metric).astype(float)-truth).mean())
    assert baseline_mae==old['mae']
    panel=Image.new('RGB',(1280,282),'white')
    draw=ImageDraw.Draw(panel)
    for i,(label,picture) in enumerate([('Content',read('content')),('Reference',read('reference')),
            ('GT: offline only',read('gt')),('current13',baseline),('ICTone Q3 README',metric)]):
        panel.paste(picture,(256*i,26))
        draw.text((256*i+5,7),label,fill='black')
    panel.save(out/'diagnostic-panel.png')
    result=dict(finished=True,scope='one consumed development image; full source/state file audit; no independent/human acceptance',
                source_sha256=digest(Path(__file__)),producer_result_sha256=digest(out/'results.json'),
                stage_result_sha256=digest(out/'stage.json'),state_parameters_verified=1846,
                gpu_repeated_by_auditor=False,conversion='same pinned Diffusers official conversion, recomputed from original bytes',
                output_crop_resize_exact=True,inputs_exact=True,index=22927,
                baseline_mae=baseline_mae,candidate_mae=candidate_mae,
                severe=bool(candidate_mae-baseline_mae>5 and candidate_mae>baseline_mae*1.25))
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
