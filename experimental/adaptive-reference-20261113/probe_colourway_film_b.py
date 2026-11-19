"""Screen the unmodified Colourway optimizer on Film B portrait pairs."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path

import numpy as np
from PIL import Image

from probe_bg_film_b_reference_path import (
    BASE, PARENT, PARENT_SHA, SOURCE, SOURCE_SHA, STRENGTH, checked_rgb, evaluate,
    json_to, manifest, mask_for, panel, save, verified_json,
)
from spatial_render import finish_spatial

SOURCE_BLOB = "969c42db165515ff0fc9a7fa4b04d7c86263673c"
LICENSE_BLOB = "9a0d81f7170d6febb942e9ede28b2f1bb1728554"
REVISION = "1c755f5007601de139ddb8da69221cfb63be8c0e"


def blob(file):
    data = file.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def engine(folder):
    source = folder / "lookgrade.py"
    if blob(source) != SOURCE_BLOB or blob(folder / "LICENSE") != LICENSE_BLOB:
        raise ValueError("Unexpected Colourway source or license")
    spec = importlib.util.spec_from_file_location("colourway_pinned", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(module, target, ref):
    def grab(name, t, width, height):
        image = target if name == "target" else ref
        return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR),
                          dtype=np.float32) / 255

    module.grab = grab
    module.probe = lambda _: {"duration": 1.}
    module.save_params = lambda *args: None
    with contextlib.redirect_stdout(io.StringIO()):
        params = module.cmd_match("reference", "target", None, width=96, samples=1)
    frame = module.finish(module.color_grade(target.astype(np.float32) / 255, params,
                                            table=module.tone_table(params)), params, np.random.default_rng(5))
    full = np.rint(frame * 255).astype(np.uint8)
    return finish_spatial(target, full, STRENGTH), params


def run(folder, out):
    module = engine(folder)
    prior = verified_json(BASE / SOURCE / "results.json", SOURCE_SHA)
    if prior["qualified_cases"] != 4:
        raise ValueError("Film B qualification changed")
    entries = {row["pair"]: row for row in verified_json(PARENT / "results.json", PARENT_SHA)["cases"]}
    out.mkdir()
    cases = {}
    for key, row in manifest(BASE, "bg-real-editor-look")["cases"].items():
        root = BASE / "bg-real-editor-look" / key
        target = checked_rgb(root / "target.png", row["artifacts"]["target.png"])
        known = prior["qualification"][key]
        expected = checked_rgb(BASE / SOURCE / f"{key}-expected.png", known["expected_sha256"])
        fields, details, board, frozen = {}, {}, {}, {}
        old = prior["comparison"]["expanded"]["cases"][key]
        for variant in ("C1", "F1"):
            before = f"{variant}-off-reference.png"
            refs = {"off": checked_rgb(root / before, row["artifacts"][before]),
                    "on": checked_rgb(BASE / SOURCE / f"{key}-{variant}-reference.png",
                                      known["reference_diagnostics"][variant]["edited_sha256"])}
            fields[variant], details[variant], frozen[variant] = {}, {}, {}
            for state, ref in refs.items():
                tag = f"{variant}-{state}"
                result, params = render(module, target, ref)
                fields[variant][state] = result
                frozen[variant][state] = checked_rgb(BASE / SOURCE / f"expanded-{key}-{tag}.png", old["outputs"][tag])
                details[variant][state] = {"params": params, "sha256": save(out / f"{key}-{tag}.png", result)}
                board[tag] = {"target": target, "reference": ref, "expanded": frozen[variant][state],
                              "Colourway": result, "expected": expected if state == "on" else target}
        person = mask_for(entries[key], key.split("-")[0])
        truth = {"off": target, "on": expected}
        assert evaluate(target, truth, frozen, person) == old["scores"]
        cases[key] = {"candidate": evaluate(target, truth, fields, person), "expanded": old["scores"],
                      "outputs": details, "comparison_sha256": panel(out / f"{key}-comparison.png", board)}
        if len(cases) == 1:
            same, _ = render(module, target, target)
            identity = {"mae": float(np.abs(same.astype(np.int16) - target).mean()),
                        "passed": bool(np.array_equal(same, target)), "sha256": save(out / "identity.png", same)}
    n = sum(x["candidate"]["passed"] for x in cases.values())
    worst = max(x["candidate"]["max_protected_swap"] for x in cases.values())
    summary = {"passing_cases": n, "worst_protected_swap": worst, "passed": n >= 3 and worst <= 3 and identity["passed"]}
    json_to(out / "results.json", {"revision": REVISION, "source_blob": SOURCE_BLOB, "license_blob": LICENSE_BLOB,
            "film_b_source_sha256": SOURCE_SHA, "cases": cases, "identity": identity, "summary": summary})
    print(summary, identity)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("folder", type=Path)
    cli.add_argument("out", type=Path)
    run(**vars(cli.parse_args()))
