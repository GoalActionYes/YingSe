"""Compare frozen round-trip correction to a magnitude-matched global reduction."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from probe_frozen_candidate_development_crosscheck import COUNTS, film_b
from probe_reference_skin_chroma_attribution import CORE, enough
from probe_reference_skin_chroma_isolation import skin_boxes
from probe_stable_chroma_hybrid import chart
from probe_target_mirror_chroma import (
    BASE, CHECKPOINT, LUMA, LUMA_INTEGER, background, checked_json, checked_rgb, chroma, facd, json_to, render,
)
from verify_development_inputs import verify

PARENT_SHA = '5c75221f19963b29e270235d414dbe93338f9eeaea6bd7c906187cb5bde2125e'
MATCH_TOLERANCE = .15


def matched_reduction(target, baseline, corrected):
    light = baseline @ LUMA
    start = chroma(baseline, light)
    desired = float(np.abs(chroma(corrected, corrected @ LUMA) - start).mean())
    endpoint, _ = render(chroma(target, target @ LUMA), light)
    direction = chroma(endpoint, endpoint @ LUMA) - start
    available = float(np.abs(direction).mean())
    fraction = min(desired / available, 1.) if available else 0.
    reduced, _ = render(start + fraction * direction, light)
    actual = float(np.abs(chroma(reduced, reduced @ LUMA) - start).mean())
    if np.abs((reduced.astype(np.int32) - baseline) @ LUMA_INTEGER).max() > 5000:
        raise ValueError('Changed luminance beyond rounding')
    return reduced, dict(requested=desired, achieved=actual, fraction=fraction,
                         saturated=desired > available, matched=abs(desired - actual) <= MATCH_TOLERANCE)


def artifacts(suite, report):
    if suite == 'chart':
        for state, record in report['outputs'].items():
            yield Path(f'{state}.png'), record
        return
    for key, row in report['cases'].items():
        for state, record in row['outputs'].items():
            if suite == 'film_b':
                scene, donor = key.split('<-')
                path = Path(f'{scene}-from-{donor}-{state}.png')
            else:
                path = Path(key) / (state if suite == 'facd' else f'{state}.png')
            yield path, record


def evaluate_suite(suite, run, engine, out, historical, manifest):
    rows = iter(artifacts(suite, manifest))

    def control(_, target, reference, baseline):
        key, record = next(rows)
        corrected = checked_rgb(historical / suite / key, record['sha256'])
        if record['route'] == 'corrected':
            reverse = checked_rgb(historical / 'reverse' / record['reverse_file'], record['reverse_sha256'])
            delta = reverse.astype(float) - target
            y = baseline @ LUMA
            rebuilt, _ = render(chroma(baseline, y) - .5 * chroma(delta, delta @ LUMA), y)
            np.testing.assert_array_equal(rebuilt, corrected)
        else:
            np.testing.assert_array_equal(corrected, target)
        return matched_reduction(target, baseline, corrected)

    result = run(engine, out / suite, control, 'Magnitude-matched reduction')
    if next(rows, None) is not None:
        raise ValueError('Unused historical outputs')
    return result


def compare_regions(control, round_trip, files):
    core, regressions, rows = 0, [], {}
    for pair, names in control.items():
        rows[pair] = {}
        for label, states in names.items():
            rows[pair][label] = {}
            for state, values in states.items():
                fixed = round_trip[pair][label][state]
                if values['baseline'] != fixed['baseline']:
                    raise ValueError('Changed baseline region score')
                eligible = all(files[pair]['outputs'][f'{role}-{state}']['matched'] for role in ('C1', 'F1'))
                uniform, spatial = values['candidate'], fixed['candidate']
                if eligible and label in CORE.get(pair, ()):
                    core += int(enough(uniform, spatial))
                elif eligible and spatial - uniform > .25:
                    regressions.append(dict(pair=pair, region=label, state=state, increase=spatial - uniform))
                rows[pair][label][state] = dict(round_trip=spatial, uniform=uniform, matched=eligible,
                                                baseline=fixed['baseline'])
    return dict(core_improved=core, regressions=regressions, rows=rows, passed=core >= 4 and not regressions)


def run(out):
    inputs = verify(BASE, COUNTS)
    historical = BASE / 'chroma-round-trip-v1'
    prior = checked_json(historical / 'results.json', PARENT_SHA)
    if prior['baseline_sha256'] != CHECKPOINT:
        raise ValueError('Unexpected frozen baseline')
    engine = SimpleNamespace(checkpoint_sha=CHECKPOINT)  # Only the identity is used by the frozen scoring entry points.
    sample = np.array([[[32, 64, 128], [123, 241, 227]]], dtype=np.uint8)
    stable, details = matched_reduction(sample, sample, sample)
    np.testing.assert_array_equal(stable, sample)
    if not details['matched'] or details['fraction'] != 0:
        raise ValueError('Identity comparison changed')
    out.mkdir()
    checks = (('background', background), ('facd', facd), ('film_b', film_b), ('chart', chart))
    suites = {name: evaluate_suite(name, check, engine, out, historical, prior['suites'][name]) for name, check in checks}
    uniform = skin_boxes(out, suites['background'])
    regions = compare_regions(uniform, prior['regions'], suites['background']['cases'])
    json_to(out / 'results.json', dict(parent_sha256=PARENT_SHA, input_counts=inputs, regions=regions, suites=suites))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
