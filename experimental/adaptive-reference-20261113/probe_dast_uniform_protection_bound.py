"""Measure the fixed facial-chroma gain tradeoff using the archived DAST renders."""
import argparse
from pathlib import Path
import numpy as np

from probe_dast_local_edit_cross import INPUT_SHA, STATES, digest, midtone, read_ledger, verified_png, write_png
from probe_dast_radiometric_control import linear, srgb, Y
from probe_dast_reference_exposure_normalization import evaluate_group
from probe_hdm_hdr_display import json_to


def replay(original, on, mask, gain):
    src, out = [np.maximum(linear(x[mask]), 1e-8) for x in (original, on)]
    a, b = [x / x[:, 1:2] for x in (src, out)]
    ratio = a ** (1-gain) * b ** gain
    mapped = ratio * ((out @ Y) / (ratio @ Y))[:,None]
    rgb = on.copy()
    rgb[mask] = np.rint(srgb(mapped)).astype(np.uint8)
    return rgb, float((mapped.max(axis=1)>1).mean())


def run(base, out, original_sha, normalized_sha):
    src, edits, orig, norm = [base / name for name in (
        "dast-exposure-prepared-v1", "dast-local-edit-prepared-v1",
        "dast-local-edit-cross-v1", "dast-reference-exposure-normalization-v1")]
    archive = read_ledger(orig, "results.json", original_sha)
    modified = read_ledger(norm, "results.json", normalized_sha)
    assert modified["source_sha256"]==original_sha
    groups = read_ledger(src,"inputs.json",INPUT_SHA)["groups"]
    operations = read_ledger(edits,"edits.json",archive["edits_sha256"])["groups"]
    out.mkdir()
    results = {}
    for group, prior in zip(groups, archive["groups"], strict=True):
        k, dest = [str(group[field]) for field in ("group","target_group")]
        assert int(k)==prior["group"]
        row = groups[int(dest)-1]
        setting = operations[dest]["nominal"]
        original = verified_png(src/dest/"nominal.png",row["frames"]["nominal"])
        mask = verified_png(src/dest/"nominal-mask.png",row["target_mask_sha256"])>0
        outside = verified_png(edits/dest/"outside.png",setting["outside_sha256"])>0
        truth = verified_png(edits/dest/"truth.png",setting["truth_sha256"])
        results[k]={}
        for arm in ("native","normalized"):
            item = modified["cases"][k]
            current = {}
            for state in STATES:
                for tag in ("off","on"):
                    name=f"{state}-{tag}"
                    path, sha = ((orig/k/f"{name}.png",prior["output_sha256"][name]) if arm=="native" else
                                 (norm/f"{k}-{name}.png",item["outputs"][name]["sha256"]))
                    current[state,tag]=verified_png(path,sha)
            scores=evaluate_group(original,truth,current,mask,outside,operations[k])
            known=prior if arm=="native" else item["candidate"]
            assert scores=={name:known[name] for name in scores}
            exposure=max(v["chroma_norm"] for v in scores["exposure"].values())
            state=min(scores["edit"],key=lambda j:scores["edit"][j]["projection"])
            projection=scores["edit"][state]["projection"]
            assert projection>0
            cap, floor=.03/exposure,.75/projection
            area=mask & midtone(original) & midtone(current[state,"on"]) & midtone(current[state,"off"])
            image, clipped=replay(original,current[state,"on"],area,cap)
            results[k][arm]={"gain_max":cap,"gain_min":floor,"compatible":floor<cap,
                             "weakest_state":state,"projection_at_stability_limit":projection*cap,
                             "clipped":clipped,"sha256":write_png(out/f"{k}-{arm}.png",image)}
    json_to(out/"results.json",{"original_sha256":original_sha,"normalized_sha256":normalized_sha,"cases":results})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base",type=Path)
    cli.add_argument("out",type=Path)
    cli.add_argument("original_sha")
    cli.add_argument("normalized_sha")
    run(**vars(cli.parse_args()))
