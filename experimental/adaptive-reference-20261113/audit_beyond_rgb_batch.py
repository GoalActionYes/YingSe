"""Freeze a second measured scene before viewing any of the four raw captures."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from math import dist
from pathlib import Path

from audit_beyond_rgb_exposure import LIGHTS, PROBE, check_camera, measure, read_tags, select
from audit_beyond_rgb_lab_index import CENTRAL_SHA, MEMBERS
from probe_beyond_rgb_transfer import write_json


ANCHOR = 'WHITE_green'
PRIOR = '5990ac045dd3bde239d3e5e1d94164af8b238c6c4bf16d7b0f7bdc06634d7d82'
SCENES = ('BLUE_blue', 'BLUE_green', 'BLUE_red', 'BLUE_rich', 'BLUE_yellow',
          'GREEN_green', 'GREEN_organic', 'RED_red', 'WHITE_blue')
TAG = 'WT/oppo_tags.json'


def evaluate(scene, captures, cache, proxy):
    lights = [measure(cache, scene, light, captures[scene, light][PROBE], proxy) for light in LIGHTS]
    lux = [measurement['lux'] for measurement in lights]
    ratio = max(lux) / min(lux)
    gap = dist(*(measurement['xy'] for measurement in lights))
    record = dict(readings=lights, lux_ratio=ratio, xy_distance=gap, light_pass=ratio <= 1.15 and .025 <= gap <= .20)
    if record['light_pass']:
        tags = [read_tags(cache, f'{scene}-{light}-oppo', captures[scene, light][TAG], proxy)
                for light in LIGHTS]
        record['oppo'] = check_camera(tags) | {'tags': tags}
    return record


def run(central, prior, cache, output, proxy):
    raw = prior.read_bytes().replace(b'\r\n', b'\n')
    if sha256(raw).hexdigest() != PRIOR:
        raise ValueError('Unexpected frozen precheck')
    old = json.loads(raw)
    anchor = old['scenes'][ANCHOR]
    if old['lights'] != list(LIGHTS) or not anchor['light_pass'] or not anchor['cameras']['oppo']['passes']:
        raise ValueError('Changed qualifying anchor')

    captures = select(central, SCENES + (ANCHOR,), MEMBERS)
    cache.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {scene: pool.submit(evaluate, scene, captures, cache, proxy) for scene in SCENES}
        candidates = {scene: future.result() for scene, future in futures.items()}
    eligible = [scene for scene, row in candidates.items() if row['light_pass'] and row['oppo']['passes']]
    eligible.sort(key=lambda scene: (not scene.startswith('WHITE_'), candidates[scene]['lux_ratio'], scene))

    selected = {}
    if eligible:
        for scene in (ANCHOR, eligible[0]):
            selected[scene] = dict(members={light: captures[scene, light] for light in LIGHTS})
    receive = sum(captures[scene, light]['WT/oppo.h5']['compressed_bytes']
                  for scene in selected for light in LIGHTS)
    if receive > 85_000_000:
        raise ValueError('Raw transfer exceeds predeclared budget')
    write_json(output, dict(central_sha256=CENTRAL_SHA, prior_sha256=PRIOR, lights=LIGHTS,
                            anchor=anchor, candidates=candidates, ranked=eligible, selected=selected,
                            raw_images_compressed_bytes=receive))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for arg in ('central', 'prior', 'cache', 'output'):
        parser.add_argument(arg, type=Path)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    run(**vars(parser.parse_args()))
