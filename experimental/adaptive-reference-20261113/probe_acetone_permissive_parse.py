"""Inspect AceTone's upstream permissive decode without re-running generation."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re

import torch

from probe_acetone_vq_film_b import PARENT_SHA, load_model, sample
from probe_arri_shared_look import quantize
from probe_gros_film_b_cross import LIGHTS, score
from probe_gros_portrait_light_transfer import masks
from probe_target_mirror_chroma import BASE, STRENGTH, checked_rgb, digest, save
from spatial_render import finish_spatial


def upstream_codes(text):
    body = text.split("<SoT>")[-1].split("<EoT>")[0]
    raw = [int(code) for code in re.findall(r"<MM(\d+)>", body)]
    if any(code >= 256 for code in raw):
        raise ValueError("Decoded code outside the published LUT vocabulary")
    return (raw + [raw[0] if raw else 0] * max(0, 64 - len(raw)))[:64], len(raw)


@torch.inference_mode()
def run(probe, codec, out, cold_only=False):
    assert upstream_codes("prefix <SoT><MM5><EoT><MM6>") == ([5] * 64, 1)
    assert upstream_codes("") == ([0] * 64, 0)
    assert upstream_codes("<SoT>" + "<MM7>" * 65) == ([7] * 64, 65)
    frozen = BASE / "gros-film-b-cross-v2"
    parent = frozen / "results.json"
    assert digest(parent) == PARENT_SHA
    report = json.loads(parent.read_bytes())
    zone = json.loads(Path(__file__).with_name("gros-portrait-light-regions.json").read_bytes())
    hashes = report["inputs"]["01-cold"]["hashes"]
    target, styled = [checked_rgb(frozen / f"01-cold-{key}.png", hashes[key]) for key in ("off", "on")]
    expected = finish_spatial(target, styled, STRENGTH)
    regions = masks(zone["01"]["cold"], zone["screen"], target.shape[:2])
    torch.set_num_threads(2)
    model = load_model(codec)
    out.mkdir()
    predictions, cases = {}, {}
    for light in LIGHTS[:1] if cold_only else LIGHTS:
        predictions[light] = {}
        for state in (("off",) if cold_only else ("off", "on")):
            label = f"{light}-{state}"
            path = probe / f"{label}.json"
            stats = json.loads(path.read_bytes())
            codes, count = upstream_codes(stats["text"])
            table = model.decode_indices(torch.tensor(codes).reshape(1, 4, 4, 4))
            array = table[0].permute(1, 2, 3, 0).numpy()[..., ::-1]
            rendered = finish_spatial(target, quantize(sample(array, target)), STRENGTH)
            predictions[light][state] = rendered
            identity_errors = {name: float(abs(rendered[area].astype("int16") - target[area]).mean())
                               for name, area in regions.items()}
            cases[label] = {"strict_valid": stats["valid"], "numeric_codes": count, "codes": codes,
                            "output_sha256": save(out / f"{label}.png", rendered),
                            "generated_transcript_sha256": digest(path),
                            "difference_from_unedited_target_mae": identity_errors}
    results = {"cases": cases}
    if not cold_only:
        results["scores"] = score(target, expected, predictions, regions)
        results["passed_appearance"] = all(row["passed"] for row in results["scores"].values())
    (out / "results.json").write_text(json.dumps(results, allow_nan=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = ArgumentParser(description=__doc__)
    for name in ("probe", "codec", "out"):
        args.add_argument(name, type=Path)
    args.add_argument("--cold-only", action="store_true")
    run(**vars(args.parse_args()))
