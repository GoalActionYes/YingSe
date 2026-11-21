"""Run the pinned PPST appearance transfer on a real two-image portrait."""

from argparse import ArgumentParser
import ast
from hashlib import file_digest
import sys
from pathlib import Path
import time
from types import ModuleType
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor

from probe_gros_film_b_cross import LIGHTS, masks, score
from probe_gros_render_blend_bound import BASELINE_SHA, REGIONS_SHA
from probe_target_mirror_chroma import BASE, STRENGTH, checked_json, checked_rgb, finish_spatial, json_to, save


REVISION = '7d13291368e166442cfba50312faa41311c45391'
SOURCE = '3c2a0c9153cd7218a8f59e250542f6759137ecf4a91e269a28344542b49b8c34'
WEIGHT = '0378f714135fac2338ead18d1504edfddf56ff78aa2a64cef9537dd1abd454ec'
SIZE = 512


def digest(path):
    with path.open('rb') as reader:
        return file_digest(reader, 'sha256').hexdigest()


def execute_functions(nodes, names, filename, namespace):
    selected = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f'Missing author functions in {filename}')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(filename), 'exec'), namespace)


def load_author(source):
    root = source / f'PPST-{REVISION}'
    archive = source / 'source.zip'
    if digest(archive) != SOURCE:
        raise ValueError('Changed PPST source archive')
    with ZipFile(archive) as z:
        for key in z.namelist():
            if key.endswith('.py'):
                path = root / key.removeprefix(root.name + '/')
                if path.read_bytes() != z.read(key):
                    raise ValueError(f'Changed PPST script: {key}')
    util = ModuleType('util')
    core = root / 'util/util.py'
    util.__dict__['torch'] = torch
    execute_functions(ast.parse(core.read_bytes()).body, ('normalize', 'lerp', 'str2bool'), core, util.__dict__)
    util.is_custom_kernel_supported = lambda: False
    sys.modules['util'] = util
    wrapper = ModuleType('models')
    wrapper.__path__ = [str(root / 'models')]
    sys.modules['models'] = wrapper
    from models.networks.encoder_con import StyleGAN2ResnetEncodercon
    from models.networks.encoder_col import StyleGAN2ResnetEncodercol
    from models.networks.generator import StyleGAN2ResnetGenerator

    parser = ArgumentParser(add_help=False)
    mapping = {'E1': StyleGAN2ResnetEncodercon, 'E2': StyleGAN2ResnetEncodercol,
               'G': StyleGAN2ResnetGenerator}
    for cls in mapping.values():
        cls.modify_commandline_options(parser, False)
    parser.set_defaults(global_code_ch=2048, spatial_code_ch=256, match_kernel=1,
                        num_classes=0, use_antialias=True)
    options = parser.parse_args([])
    models = torch.nn.Module()
    models.opt = options
    for name, cls in mapping.items():
        models.add_module(name, cls(options))

    author = root / 'models/ppst_model.py'
    tree = ast.parse(author.read_bytes())
    part = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name.lower() == 'ppstmodel')
    functions = {'torch': torch, 'F': F, 'sys': sys}
    names = ('encode', 'encode2', 'extract_feat_from_image', 'Rselfcorr', 'corrm')
    execute_functions(part.body, names, author, functions)
    for name in names:
        setattr(models, name, functions[name].__get__(models))
    return models


def load_weights(source, model):
    path = source / 'latest_checkpoint.pth'
    if path.stat().st_size != 411_443_360 or digest(path) != WEIGHT:
        raise ValueError('Changed PPST checkpoint')
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    relevant = {key: value for key, value in state.items() if key.split('.')[0] in ('E1', 'E2', 'G')}
    expected = model.state_dict()
    if relevant.keys() != expected.keys() or any(value.shape != expected[key].shape for key, value in relevant.items()):
        raise ValueError('Checkpoint image modules do not match the pinned model')
    model.load_state_dict(relevant, strict=True, assign=True)
    return model.eval().cuda(), len(relevant)


def letterbox(rgb):
    h, w = rgb.shape[:2]
    factor = SIZE / max(w, h)
    width, height = round(w * factor), round(h * factor)
    frame = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BICUBIC))
    left, top = (SIZE - width) // 2, (SIZE - height) // 2
    padded = np.pad(frame, ((top, SIZE-height-top), (left, SIZE-width-left), (0,0)), mode='edge')
    return padded, (left, top, left+width, top+height)


def restore(square, crop, original):
    h, w = original.shape[:2]
    image = square.crop(crop).resize((w, h), Image.Resampling.BICUBIC)
    return np.array(image)


@torch.inference_mode()
def render(model, target, reference):
    torch.manual_seed(48)
    clean, crop = letterbox(target)
    guide, _ = letterbox(reference)
    signal, style = (to_tensor(Image.fromarray(frame)).unsqueeze(0).cuda() * 2 - 1
                     for frame in (clean, guide))
    spatial, _ = model.encode(signal)
    content, intrinsic = model.extract_feat_from_image(signal)
    look, secondary = model.extract_feat_from_image(style)
    content = torch.cat((content, model.Rselfcorr(intrinsic)), dim=1)
    look = torch.cat((look, model.Rselfcorr(secondary)), dim=1)
    correspondence = model.corrm(look, content)
    _, transferred = model.encode2(style, correspondence)
    prediction = model.G(spatial, transferred)
    if not torch.isfinite(prediction).all():
        raise ValueError('PPST produced nonfinite image')
    rgb = to_pil_image((prediction[0].cpu().clamp(-1., 1.) + 1.) * .5)
    return finish_spatial(target, restore(rgb, crop, target), STRENGTH)


def gradient_correlation(a, b):
    luma = np.array([.2126, .7152, .0722])
    first, second = (np.r_[np.diff(gray, axis=0).ravel(), np.diff(gray, axis=1).ravel()]
                     for gray in (a @ luma, b @ luma))
    return float(np.corrcoef(first, second)[0, 1])


def run(source, out):
    torch.set_num_threads(4)
    model, parameters = load_weights(source, load_author(source))
    photos = BASE / 'gros-film-b-cross-v2'
    manifest = checked_json(photos / 'results.json', BASELINE_SHA)
    roi = checked_json(Path(__file__).with_name('gros-portrait-light-regions.json'), REGIONS_SHA)
    case = manifest['cases']['01-from-02']
    target = checked_rgb(photos / '01-cold-off.png', manifest['inputs']['01-cold']['hashes']['off'])
    expected = checked_rgb(photos / '01-from-02-expected.png', case['expected_sha'])
    areas = masks(roi['01']['cold'], roi['screen'], target.shape[:2])
    out.mkdir()
    prediction, media = {}, {}
    work, crop = letterbox(target)
    roundtrip = restore(Image.fromarray(work), crop, target)
    resize_similarity = gradient_correlation(target, roundtrip)
    for light in LIGHTS:
        prediction[light] = {}
        for state in ('off', 'on'):
            name = f'02-{light}'
            guide = checked_rgb(photos / f'{name}-{state}.png', manifest['inputs'][name]['hashes'][state])
            before = time.monotonic()
            result = render(model, target, guide)
            key = f'{light}-{state}'
            prediction[light][state] = result
            media[key] = {'sha256': save(out / f'{key}.png', result),
                          'gradient_correlation': gradient_correlation(target, result)}
            print(key, media[key], round(time.monotonic()-before, 3), flush=True)
    same = render(model, target, target)
    control = {'sha256': save(out / 'same.png', same),
               'mae': float(np.abs(same.astype(float)-target).mean()),
               'gradient_correlation': gradient_correlation(target, same)}
    result = {'source_sha256': SOURCE, 'weight_sha256': WEIGHT, 'inference_tensors': parameters,
              'resize_gradient_correlation': resize_similarity, 'media': media,
              'self_control': control, 'scores': score(target, expected, prediction, areas),
              'baseline_13': case['scores']['13']}
    json_to(out / 'results.json', result)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for name in ('source','out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
