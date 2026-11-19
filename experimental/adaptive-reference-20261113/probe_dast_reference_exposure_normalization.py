"""Test a fixed single-reference exposure normalization on DAST."""
import argparse
from pathlib import Path
import numpy as np
import torch
from probe_dast_local_edit_cross import (
    INPUT_SHA, STATES, digest, midtone, read_ledger, score_edit, statistics, verified_png, write_png,
)
from probe_dast_radiometric_control import linear, srgb, Y
from spatial_render import SpatialEngine, finish_spatial
from probe_hdm_hdr_display import json_to

ARCHIVE = "dast-local-edit-cross-v1"


def normalize(rgb):
    image = linear(rgb)
    gain = float(np.clip(.18 / max(float(np.median(image @ Y)), 1e-6), .25, 4))
    scaled = image * gain
    return np.rint(srgb(scaled)).astype(np.uint8), {"gain": gain, "clipped": float((scaled.max(-1)>1).mean())}


def evaluate_group(original, truth, frames, core, outside, edit):
    scores = {k: score_edit(original, truth, frames[k,"off"], frames[k,"on"], core, outside, edit[k]["source_edit"])
              for k in STATES}
    exposure = {}
    for state in STATES[1:]:
        for kind in ("off", "on"):
            a,b = frames["nominal",kind],frames[state,kind]
            exposure[f"{state}-{kind}"] = statistics(a,b,core & midtone(original) & midtone(a) & midtone(b))
    passed = all(v["passes"] for v in scores.values()) and all(v.get("chroma_norm", np.inf)<.03 for v in exposure.values())
    return {"edit": scores, "exposure": exposure, "passes": passed}


@torch.inference_mode()
def run(base,out,identity):
    torch.set_num_threads(4)
    frozen = read_ledger(base/ARCHIVE,"results.json",identity)
    source, edits = (base/k for k in ("dast-exposure-prepared-v1","dast-local-edit-prepared-v1"))
    entries = read_ledger(source,"inputs.json",INPUT_SHA)["groups"]
    operations = read_ledger(edits,"edits.json",frozen["edits_sha256"])["groups"]
    engine = SpatialEngine("multiref-v1")
    assert engine.checkpoint_sha == frozen["weight_sha256"]
    out.mkdir()
    result = {}
    for row, archived in zip(entries,frozen["groups"],strict=True):
        key, dest = (str(row[k]) for k in ("group","target_group"))
        assert int(key)==archived["group"]
        target = entries[int(dest)-1]
        nominal = operations[dest]["nominal"]
        original = verified_png(source/dest/"nominal.png",target["frames"]["nominal"])
        truth = verified_png(edits/dest/"truth.png",nominal["truth_sha256"])
        core = verified_png(source/dest/"nominal-mask.png",target["target_mask_sha256"])>0
        outside = verified_png(edits/dest/"outside.png",nominal["outside_sha256"])>0
        hist, candidate, outputs = {},{},{}
        for state in STATES:
            guide = {"off":verified_png(source/key/f"{state}.png",row["frames"][state]),
                     "on":verified_png(edits/key/f"{state}-on.png",operations[key][state]["edited_sha256"])}
            for kind,rgb in guide.items():
                tag=f"{state}-{kind}"
                hist[state,kind]=verified_png(base/ARCHIVE/key/f"{tag}.png",archived["output_sha256"][tag])
                ref,info=normalize(rgb)
                frame=finish_spatial(original,engine.prepare(original,ref).apply(original),.8)
                candidate[state,kind]=frame
                outputs[tag]=info | {"sha256":write_png(out/f"{key}-{tag}.png",frame)}
        old=evaluate_group(original,truth,hist,core,outside,operations[key])
        assert old=={k:archived[k] for k in old}
        result[key]={"baseline":old,"candidate":evaluate_group(original,truth,candidate,core,outside,operations[key]),
                     "outputs":outputs}
    count=sum(k["candidate"]["passes"] for k in result.values())
    json_to(out/"results.json",{"source_sha256":identity,"cases":result,"passing":count,"passed":count>=3})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("base",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("identity")
    run(**vars(p.parse_args()))
