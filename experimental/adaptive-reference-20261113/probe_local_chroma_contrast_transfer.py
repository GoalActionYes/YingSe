"""Transfer within-block chroma contrast without copying the reference average cast."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from probe_arri_shared_look import metrics
from probe_beyond_rgb_matched_cast import compare
from probe_gros_film_b_cross import ARCHIVE, masks, score
from probe_multilum_gray_probe import chroma, valid
from probe_physical_blue_global_fit import display, linear
from probe_target_luma_edit_mix import LUMA, beyond_inputs, mid_inputs
from probe_target_mirror_chroma import BASE, checked_json, checked_rgb, digest, json_to, panel, save
from probe_two_image_chroma_quantile_gain import archive


BLOCK = 16
RIDGE = .001


def centered_blocks(color):
    h, w = color.shape[:2]
    padded = np.pad(color, ((0, (-h) % BLOCK), (0, (-w) % BLOCK), (0, 0)), mode="edge")
    groups = padded.reshape(padded.shape[0] // BLOCK, BLOCK, padded.shape[1] // BLOCK, BLOCK, 2)
    residual = (groups - groups.mean(axis=(1, 3), keepdims=True)).reshape(padded.shape)
    return residual[:h, :w]


def covariance(image, detail):
    support = valid(image)
    rows = detail[support]
    if len(rows) < 256:
        return None
    return rows.T @ rows / len(rows) + np.eye(2) * RIDGE


def power(matrix, exponent):
    roots, directions = np.linalg.eigh(matrix)
    return (directions * roots ** exponent) @ directions.T


def local_render(target, reference, baseline):
    if np.array_equal(target, reference):
        return target.copy(), dict(reason="same")
    ctx = chroma(baseline)
    detail = centered_blocks(ctx)
    dst, src = covariance(baseline, detail), covariance(reference, centered_blocks(chroma(reference)))
    if dst is None or src is None:
        return baseline.copy(), dict(reason="support")
    half, inverse = power(dst, .5), power(dst, -.5)
    op = inverse @ power(half @ src @ half, .5) @ inverse
    values, vectors = np.linalg.eigh((op + op.T) / 2)
    mapping = (vectors * np.clip(values, .5, 2)) @ vectors.T
    edited = ctx + detail @ (mapping - np.eye(2))
    gains = np.exp(np.stack((edited[..., 0], np.zeros_like(edited[..., 0]), edited[..., 1]), axis=-1))
    y = linear(baseline) @ LUMA
    mapped = gains * (y / (gains @ LUMA))[..., None]
    peak = mapped.max(axis=-1)
    scale = np.divide(1 - y, peak - y, out=np.ones_like(y), where=peak > 1)
    limited = y[..., None] + (mapped - y[..., None]) * scale[..., None]
    peak_error = float(np.abs(limited @ LUMA - y).max())
    if peak_error > 1e-12:
        raise ValueError("Changed baseline linear luma")
    result = display(limited)
    return result, dict(reason="transfer", gains=np.linalg.eigvalsh(mapping).tolist(),
                        luma_residual=peak_error, seam=seam(baseline, result))


def seam(before, after):
    readings = []
    for axis in (0, 1):
        delta = np.abs(np.diff(after.astype(float), axis=axis)) - np.abs(np.diff(before.astype(float), axis=axis))
        ordered = np.moveaxis(delta, axis, 0)
        edges = np.arange(BLOCK - 1, ordered.shape[0] - 1, BLOCK)
        readings.append(float(ordered[edges].mean() - ordered[edges + 1].mean()) if len(edges) else 0.)
    return readings


def emit(out, name, target, expected, original, candidate, audit, scores):
    return dict(audit=audit, scores=scores,
                hashes={key: save(out / f"{name}-{key}.png", rgb) for key, rgb in candidate.items()},
                panel_sha=panel(out / f"{name}-comparison.png",
                                {"target": {"original": target, "desired": expected}, "13": original, "contrast": candidate}))


def run(out, film_sha, exposure_sha):
    out.mkdir()
    film = BASE / "gros-film-b-cross-v2"
    directory = BASE / "two-image-chroma-quantile-gain-v1"
    effects = checked_json(film / "results.json", film_sha)
    exposure = checked_json(directory / "results.json", exposure_sha)
    source = checked_json(BASE / "gros-portrait-light-transfer-v1" / "results.json", ARCHIVE)
    labels = Path(__file__).with_name("gros-portrait-light-regions.json")
    if digest(labels) != source["regions_sha256"]:
        raise ValueError("Changed saved scoring regions")
    zones = json.loads(labels.read_bytes())
    cases = {}
    for name, row in effects["cases"].items():
        person, donor = name.split("-from-")
        def read(person, state):
            light = state.split("-")[0]
            sha = effects["inputs"][f"{person}-{light}"]["hashes"][state.split("-")[1]]
            return checked_rgb(film / f"{person}-{state}.png", sha)
        target = read(person, "cold-off")
        expected = checked_rgb(film / f"{name}-expected.png", row["expected_sha"])
        regions = masks(zones[person]["cold"], zones["screen"], target.shape[:2])
        old, new, audit = {}, {}, {}
        for state in ("cold-off", "cold-on", "warm-off", "warm-on"):
            old[state] = checked_rgb(film / f"{name}-13-{state}.png", row["hashes"][f"13-{state}"])
            new[state], audit[state] = local_render(target, read(donor, state), old[state])
        grouped = {key: {light: {role: frames[f"{light}-{role}"] for role in ("off", "on")}
                         for light in ("cold", "warm")} for key, frames in (("13", old), ("contrast", new))}
        scores = {key: score(target, expected, frames, regions) for key, frames in grouped.items()}
        cases[name] = emit(out, name, target, expected, old, new, audit, scores)
    for cohort in (mid_inputs, beyond_inputs):
        for name, _, target, expected, regions, _ in cohort():
            guides, old = archive(cohort, name)
            new, audit = {}, {}
            for key, frame in old.items():
                new[key], audit[key] = local_render(target, guides[key], frame)
            same, _ = local_render(target, target, old["base"])
            if not np.array_equal(same, target):
                raise ValueError("Same image changed")
            stored = checked_rgb(directory / f"{name}-exposure-13.png",
                                 exposure["cases"][name]["hashes"]["exposure-13"])
            control, audit["exposure"] = local_render(target, display(linear(target) * .65), stored)
            delta = control.astype(float) - stored
            audit["exposure"]["rgb8_chroma_change"] = float(np.abs(delta - (delta @ LUMA)[..., None]).mean())
            scores = {arm: {key: compare(target, expected, *frames.values(), area) for key, area in regions.items()}
                      for arm, frames in (("13", old), ("contrast", new))}
            cases[name] = emit(out, name, target, expected, old, new, audit, scores)
            cases[name]["exposure_sha"] = save(out / f"{name}-exposure.png", control)
    script = Path(__file__)
    json_to(out / "results.json", dict(film_parent=film_sha, exposure_parent=exposure_sha, cases=cases,
            script_sha=digest(script), protocol_sha=digest(script.with_name("local-chroma-contrast-transfer-protocol.md"))))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("film_sha")
    parser.add_argument("exposure_sha")
    run(**vars(parser.parse_args()))
