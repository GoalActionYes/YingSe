"""Compare the archived and ACES display runs on identical source coordinates."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from probe_sparks_archival_grade import ACES_PROTOCOL, FRAMES, SIZE, photo, sha, write_json


PATCHES = {4592: ('face_patch', 183, 72, 205, 86),
           9184: ('sky_patch', 390, 20, 470, 64)}


def read_run(prepared_dir, output_dir):
    prepared_path = prepared_dir / 'prepared.json'
    results_path = output_dir / 'results.json'
    prepared = json.loads(prepared_path.read_bytes())
    results = json.loads(results_path.read_bytes())
    if results['prepared_sha256'] != sha(prepared_path):
        raise ValueError('Render does not match the prepared images')
    frames = {}
    for frame in FRAMES:
        row = results['scores'][str(frame)]
        grade = output_dir / f'grade-{frame}-from-{row["donor"]}.png'
        if sha(grade) != row['output_sha256']:
            raise ValueError(f'Render changed: {grade}')
        with Image.open(grade) as im:
            image = np.array(im)
        frames[frame] = dict(nam=photo(prepared_dir, 'nam', frame, prepared),
                             gam=photo(prepared_dir, 'gam', frame, prepared), output=image)
    return prepared, results, frames, sha(results_path)


def neutral(image):
    return (np.ptp(image, axis=-1) <= 12) & (image.min(-1) >= 32) & (image.max(-1) <= 240)


def scores(images, common, patch):
    base = images['nam'].astype(np.int16) - images['gam']
    output = images['output'].astype(np.int16) - images['gam']
    name, x, y, x2, y2 = patch
    roi = np.s_[y:y2, x:x2, :]
    errors = np.stack([np.abs(base), np.abs(output)])
    return dict(whole_mae=errors.mean(axis=(1, 2, 3)).tolist(),
                common_neutral_mae=errors[:, common].mean(axis=(1, 2)).tolist()
                if common.sum() >= 256 else None,
                patch=name, patch_xyxy=[x, y, x2, y2], output_minus_gam_rgb=output[roi].mean((0, 1)).tolist(),
                nam_minus_gam_rgb=base[roi].mean((0, 1)).tolist())


def comparison(images, path):
    w, h = SIZE
    sheet = Image.new('RGB', (4 * (w + 10) + 10, 2 * (h + 40) + 10), '#15191e')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=19)
    for row, (target, donor) in enumerate((FRAMES, FRAMES[::-1])):
        panels = (('Target NAM', images[target]['nam']), ('Other shot GAM', images[donor]['gam']),
                  ('YingSe 13 / 80%', images[target]['output']), ('Same-frame GAM', images[target]['gam']))
        for col, (label, rgb) in enumerate(panels):
            x, y = 10 + col * (w + 10), 10 + row * (h + 40)
            draw.text((x, y), label, font=font, fill='#eef1f6')
            sheet.paste(Image.fromarray(rgb), (x, y + 26))
    sheet.save(path, quality=92, subsampling=0)


def run(old_prepared, old_output, aces_prepared, aces_output, out):
    old, old_results, before, old_hash = read_run(old_prepared, old_output)
    aces, aces_results, after, aces_hash = read_run(aces_prepared, aces_output)
    if old['intake_sha256'] != aces['intake_sha256'] or 'display' in old or 'display' not in aces:
        raise ValueError('Expected the same EXRs with different display transforms')
    for key in ('weights_sha256', 'strength'):
        if old_results[key] != aces_results[key]:
            raise ValueError(f'Model configuration changed: {key}')

    cases = {}
    for frame in FRAMES:
        case = str(frame)
        common = neutral(before[frame]['nam']) & neutral(after[frame]['nam'])
        left, right = (scores(run[frame], common, PATCHES[frame]) for run in (before, after))
        cases[frame] = dict(common_neutral_pixels=int(common.sum()), previous_display=left, aces_sdr=right)
        for measured, source in ((left, old_results), (right, aces_results)):
            np.testing.assert_allclose(measured['whole_mae'], source['scores'][case]['whole_mae'], rtol=0, atol=1e-12)

    out.mkdir()
    comparison(after, out / 'comparison.jpg')
    write_json(out / 'comparison.json', dict(cases=cases, previous_results_sha256=old_hash,
                                             aces_results_sha256=aces_hash, protocol_sha256=sha(ACES_PROTOCOL),
                                             source_sha256=sha(Path(__file__)),
                                             comparison_sha256=sha(out / 'comparison.jpg')))
    print(json.dumps(cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old_prepared', 'old_output', 'aces_prepared', 'aces_output', 'out'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
