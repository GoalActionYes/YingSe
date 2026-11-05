"""Deterministic statistical baseline. No trained model or semantic protection."""
from pathlib import Path
import io
import numpy as np
from PIL import Image, ImageOps, ImageDraw

PALETTE_VERSION = 'cinematic-v2'
STYLE_IDS = ['original', 'black-gold', 'cyberpunk', 'teal-orange', 'vintage',
             'japanese', 'cool-cinema', 'sunshine', 'noir']
LABELS = ['原色', '黑金', '赛博朋克', '青橙', '复古胶片', '日系清新', '冷峻电影', '暖阳', '黑白电影']
DESCRIPTIONS = ['保持原始颜色', '深黑背景 · 琥珀金色', '青蓝与洋红 · 霓虹色调',
                '青色暗部 · 橙色暖调', '褪色暗部 · 怀旧暖色', '明亮柔和 · 淡彩青绿',
                '低饱和 · 冷蓝暗调', '金色日光 · 温暖明亮', '去色 · 深黑高对比']
CONTROL_LIMITS = {'exposure': (-3, 3), 'contrast': (-100, 100),
                  'saturation': (-100, 100), 'temperature': (-100, 100)}

def controls(values=None):
    values = {} if values is None else values
    if not isinstance(values, dict) or set(values)-set(CONTROL_LIMITS):
        raise ValueError('基础调整参数不受支持')
    result = {}
    for key, (low, high) in CONTROL_LIMITS.items():
        v = float(values.get(key, 0))
        if not np.isfinite(v) or not low <= v <= high:
            raise ValueError(f'{key} 必须在 {low} 到 {high} 之间')
        result[key] = v
    return result

def luminance(x):
    return (x * np.array([.2126, .7152, .0722], dtype=np.float32)).sum(-1, keepdims=True)

def saturate(x, amount):
    y = luminance(x)
    return y+(x-y)*amount

def tone(x, knots):
    return np.interp(x, [0, .2, .5, .8, 1], knots).astype(np.float32)

def normalized(weights):
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (9,) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('需要9个非负有限权重')
    if w.sum() == 0:
        w[0] = 1
    return w / w.sum()

def basis(rgb):
    """Authored color looks, not neural outputs or literal lighting/grain simulation."""
    x = np.clip(np.asarray(rgb, dtype=np.float32), 0, 1)
    y = luminance(x)
    # Gold selectively emphasizes warm colors; other hues become dark neutral tones.
    warm = np.clip((x[..., 0:1]-x[..., 2:3]) * 4 + .12, 0, 1)
    value = tone(y, [0, .055, .28, .72, 1])
    gold = value * [1.3, .88, .30]
    neutral = value * [.86, .88, .94]
    black_gold = neutral*(1-warm) + gold*warm
    # A continuous cyan/magenta remapping; no claim of adding neon lights or glow.
    cool = np.clip(.35+3*(x[..., 2:3]-x[..., 0:1])+1.5*(x[..., 1:2]-x[..., 0:1]),0,1)
    neon = (1-cool)*np.array([1.1,.12,.78]) + cool*np.array([.03,.87,1.18])
    cyber = tone(.30*x+.95*neon*np.power(y,.9), [0,.11,.52,.91,1])
    split = (1-y)*np.array([.025,.36,.43]) + y*np.array([1.0,.60,.25])
    teal_orange = tone(.63*saturate(x,1.18)+.50*split, [0,.11,.47,.87,1])
    vintage = saturate(tone(x,[.065,.25,.52,.76,.92]),.62)*[1.10,1.00,.81]
    japanese = saturate(np.power(x,.78)*.81+.13,.73)*[.98,1.035,1.04]
    cold = saturate(tone(x,[0,.10,.36,.71,.93]),.57)*[.78,.94,1.16]+[0,.012,.034]
    sunshine = saturate(np.power(x,.88),1.12)*[1.14,1.02,.77]+[.028,.012,0]
    noir = np.repeat(tone(y,[0,.055,.43,.9,1]),3,axis=-1)
    variants = [x, black_gold, cyber, teal_orange, vintage, japanese, cold, sunshine, noir]
    return np.clip(np.stack(variants), 0, 1).astype(np.float32)

def basic_adjust(rgb, adjustments=None):
    """Independent display-referred SDR controls; zero is neutral for every control."""
    a = controls(adjustments)
    x = np.asarray(rgb,dtype=np.float32) * (2.**a['exposure'])
    if a['contrast']:
        x = (x-.5)*(2.**(a['contrast']/60.))+.5
    if a['temperature']:
        t = a['temperature']/100.
        x = x*np.array([1+.35*t,1-.04*abs(t),1-.35*t])+[.06*t,0,-.06*t]
    x = np.clip(x,0,1)
    if a['saturation']:
        x = saturate(x,1+a['saturation']/100.)
    return np.clip(x,0,1).astype(np.float32)

def transform(rgb, weights, strength=1., adjustments=None):
    strength = float(strength)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('强度应在0和1之间')
    w = normalized(weights)
    out = np.tensordot(w, basis(rgb), axes=(0, 0))
    filtered = np.clip((1-strength)*rgb + strength*out, 0, 1).astype(np.float32)
    return basic_adjust(filtered, adjustments)

def make_lut(weights, strength=1., size=33, adjustments=None):
    if size not in (17, 33, 65):
        raise ValueError('LUT尺寸仅支持17、33或65')
    b, g, r = np.meshgrid(*([np.linspace(0, 1, size, dtype=np.float32)]*3), indexing='ij')
    return transform(np.stack([r, g, b], -1), weights, strength, adjustments)

def cube_text(lut):
    n = lut.shape[0]
    header = f'TITLE "Reference Grade {PALETTE_VERSION}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n'
    return header + '\n'.join(' '.join(f'{v:.7f}' for v in row) for row in lut.reshape(-1, 3)) + '\n'

def apply_lut(rgb, lut):
    """Trilinear interpolation, indexed [blue, green, red, channel]."""
    p = np.clip(rgb, 0, 1) * (len(lut)-1)
    lo = np.floor(p).astype(np.int32)
    hi = np.minimum(lo+1, len(lut)-1)
    f = p-lo
    out = np.zeros_like(rgb, dtype=np.float32)
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                rr = hi[..., 0] if dr else lo[..., 0]
                gg = hi[..., 1] if dg else lo[..., 1]
                bb = hi[..., 2] if db else lo[..., 2]
                factor = (f[..., 0] if dr else 1-f[..., 0]) * (f[..., 1] if dg else 1-f[..., 1]) * (f[..., 2] if db else 1-f[..., 2])
                out += lut[bb, gg, rr] * factor[..., None]
    return np.clip(out, 0, 1)

def load_image(path, max_side=1280):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((max_side, max_side))
        return np.asarray(im).astype(np.float32)/255

def png_bytes(rgb):
    f = io.BytesIO()
    Image.fromarray(np.round(np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(f, format='PNG')
    return f.getvalue()

def recommend(source, reference):
    """Fit global RGB mean/std, not semantic or neural style extraction."""
    def small(x):
        im = Image.fromarray(np.uint8(np.clip(x, 0, 1)*255))
        im.thumbnail((80, 80))
        return np.asarray(im).astype(np.float32).reshape(-1, 3)/255
    x, ref = small(source), small(reference)
    bank = basis(x).astype(np.float64)
    means = bank.mean(axis=1)
    seconds = np.einsum('ipc,jpc->ijc', bank, bank)/len(x)
    target_mean, target_std = ref.mean(0), ref.std(0)
    def losses(ws):
        mu = ws @ means
        sq = np.einsum('ki,ijc,kj->kc', ws, seconds, ws)
        sd = np.sqrt(np.maximum(sq-mu**2, 1e-10))
        return ((mu-target_mean)**2).mean(1) + ((sd-target_std)**2).mean(1)
    rng = np.random.default_rng(20260924)
    candidates = np.vstack([np.eye(9), rng.dirichlet(np.full(9, .4), 2500)])
    best = candidates[np.argmin(losses(candidates))]
    for step in (.15, .06, .02, .007):
        for _ in range(10):
            ws = [best]
            for i in range(9):
                for j in range(9):
                    if i == j: continue
                    c = best.copy(); change = min(step, c[i]); c[i] -= change; c[j] += change; ws.append(c)
            ws = np.asarray(ws)
            new = ws[np.argmin(losses(ws))]
            if np.allclose(new, best): break
            best = new
    # Factor identity out of the mix. The UI exposes it as a genuine strength slider.
    return best.tolist(), {'rgb_moment_loss_before': float(losses(np.eye(9)[:1])[0]),
                         'rgb_moment_loss_after': float(losses(best[None])[0]),
                         'note': '优化目标仅为RGB均值和标准差；降低此数值不等于提高视觉质量。'}

def split_strength(weights):
    w = normalized(weights)
    strength = float(1-w[0])
    if strength < 1e-8:
        return [1.]+[0.]*8, 0.
    w[0] = 0
    return (w/strength).tolist(), strength

def create_style_reference(folder):
    folder=Path(folder)
    path=folder/'reference-v2.png'
    if not path.exists():
        source=load_image(folder/'source.png')
        path.write_bytes(png_bytes(apply_lut(source,make_lut([0,0,0,1,0,0,0,0,0]))))

def create_demo(folder):
    """Synthetic calibration material, never claimed to be real footage."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'source.png').exists(): return
    h, w = 540, 960
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    arr = np.stack([.13+.52*xx/w, .17+.45*yy/h, .26+.38*(1-xx/w)], -1)
    im = Image.fromarray(np.uint8(arr*255)); d = ImageDraw.Draw(im)
    colors = ['#bb7858','#dfac86','#596e54','#81999d','#d0b267','#aa5254','#c9c5b9','#4f586e']
    for i, c in enumerate(colors):
        x=50+i*109; d.rounded_rectangle((x, 290, x+90, 455), radius=16, fill=c)
    d.ellipse((130,70,330,270), fill='#dbc3a8')
    d.rectangle((410,95,810,240), fill='#526778')
    d.text((40,500), 'SYNTHETIC CALIBRATION / NOT REAL FOOTAGE', fill='white')
    im.save(folder/'source.png')
    source=np.asarray(im).astype(np.float32)/255
    weights=[0,.35,0,.45,0,.20,0,0,0]
    (folder/'reference.png').write_bytes(png_bytes(apply_lut(source, make_lut(weights))))
