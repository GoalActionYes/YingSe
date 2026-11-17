"""Exercise the real pinned VAE checkpoint schema without allocating its weights."""
import os,sys,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ASSETS=ROOT/'.tools/StyleGallery-audit'
sys.path[:0]=[str(ASSETS/'vendor'),str(ROOT/'.tools/ModFlows-audit/vendor')]
from screen_stylegallery import checkpoint_key_map
import torch
from diffusers import AutoencoderKL
from safetensors import safe_open


class StyleGalleryLoadingTests(unittest.TestCase):
    def test_real_legacy_vae_key_value_and_shape_correspondence(self):
        folder=ASSETS/'source/pretrained_models/runwayml_stable-diffusion-v1-5/vae'
        config=json.loads((folder/'config.json').read_text())
        with torch.device('meta'):model=AutoencoderKL.from_config(config)
        with safe_open(str(folder/'diffusion_pytorch_model.safetensors'),framework='pt',device='cpu') as source:
            original=set(source.keys());mapping=checkpoint_key_map(model,original);actual=model.state_dict()
            self.assertEqual(set(mapping),set(actual));self.assertEqual(set(mapping.values()),original);self.assertEqual(len(mapping),len(original))
            for key,old in mapping.items():self.assertEqual(tuple(actual[key].shape),tuple(source.get_slice(old).get_shape()))
            renamed={k:v for k,v in mapping.items() if k!=v};self.assertEqual(len(renamed),16)
            for prefix in ['encoder','decoder']:
                for old,new in [('query','to_q'),('key','to_k'),('value','to_v'),('proj_attn','to_out.0')]:
                    for suffix in ['weight','bias']:
                        self.assertEqual(mapping[f'{prefix}.mid_block.attentions.0.{new}.{suffix}'],f'{prefix}.mid_block.attentions.0.{old}.{suffix}')


if __name__=='__main__':unittest.main()
