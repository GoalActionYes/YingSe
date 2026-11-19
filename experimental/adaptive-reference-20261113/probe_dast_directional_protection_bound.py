"""Bound any shared linear chroma correction on DAST exposure and edit vectors."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import numpy as np

from probe_dast_local_edit_cross import read_ledger
from probe_hdm_hdr_display import json_to


def upper(row):
    nuisance = {k: np.array(v["mean_log_chroma_change"]) for k,v in row["exposure"].items()}
    effects = {}
    for state, edit in row["edit"].items():
        observed = np.array(edit["actual"]["mean_log_chroma_change"])
        desired = np.array(edit["expected"]["mean_log_chroma_change"])
        options = []
        for first, second in combinations(nuisance, 2):
            basis = np.column_stack((nuisance[first], nuisance[second]))
            if np.linalg.cond(basis) > 1e6:
                continue
            c = np.linalg.solve(basis, observed)
            np.testing.assert_allclose(basis @ c, observed, atol=1e-12)
            # Absolute projection upper bound from the triangle / Cauchy inequalities.
            options.append((float(.03*np.abs(c).sum()/np.linalg.norm(desired)), first, second, c.tolist()))
        if not options:
            raise ValueError("No usable independent exposure vectors")
        bound, first, second, coefficients = min(options)
        effects[state] = dict(upper=bound, exposure_basis=[first,second], coefficients=coefficients)
    return {"states": effects, "tightest": min(v["upper"] for v in effects.values()),
            "excluded": any(v["upper"]<.75 for v in effects.values())}


def run(base, output, original, normalized):
    a = read_ledger(base/"dast-local-edit-cross-v1","results.json",original)
    b = read_ledger(base/"dast-reference-exposure-normalization-v1","results.json",normalized)
    assert b["source_sha256"] == original and len(a["groups"])==len(b["cases"])==5
    results = {}
    for group in a["groups"]:
        k = str(group["group"])
        results[k] = {"native": upper(group), "normalized": upper(b["cases"][k]["candidate"])}
    out = {"cases": results, "original_sha256":original, "normalized_sha256":normalized}
    json_to(output,out)
    print({k: {m:round(v["tightest"],4) for m,v in row.items()} for k,row in results.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("base",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("original")
    p.add_argument("normalized")
    run(**vars(p.parse_args()))
