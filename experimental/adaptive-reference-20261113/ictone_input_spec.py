"""Strict two-RGB input schema; truth/mask/profile fields are rejected."""
import hashlib
import json
from pathlib import Path
from PIL import Image


def read_rgb_input_spec(path,base):
    base=Path(base).resolve()
    path=Path(path).resolve()
    assert path.is_relative_to(base)
    spec=json.loads(path.read_text(encoding='utf-8'))
    assert set(spec)=={'index','content','reference'}
    assert isinstance(spec['index'],str) and spec['index'] and Path(spec['index']).name==spec['index']
    images={}
    for role in ['content','reference']:
        item=spec[role]
        assert set(item)=={'path','sha256'}
        p=Path(item['path']).resolve()
        assert p.is_relative_to(base) and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
        with Image.open(p) as im:
            assert im.mode=='RGB' and im.size==(256,256)
            images[role]=im.copy()
    return spec['index'],images
