"""Check the frozen photo-development inputs and baseline outputs before a run."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = {
    "bg-real-editor-look": "11b72e0865bad7f3d20a4b3b9a8120c5e8df0de8fd2bbc78549625d67fa1d94d",
    "facd-two-image-baseline-v1": "f5c752354e97e31abde828b402e56b404a2cdb1ae7da58208fb4de8e4258a7c9",
    "arri-two-looks-v2": "e09286501decd0d41cf727bce645d9920298da166f6bdaf4bab20385d27d730e",
    "arri-film-b-crossed-v1": "fea85f0672c6da99d5734e8fb52034438122a03a71b7e493534c4eb38da07237",
    "hdm-chart-pilot-v1": "1b5ebfd558ef631ed0a95e36296a07a65ac77d313c453a3b6a424ff9aa7fb353",
}


def read(path):
    return json.loads(path.read_bytes())


def manifest(base, suite):
    raw = (base / suite / "results.json").read_bytes()
    if sha256(raw).hexdigest() != MANIFESTS[suite]:
        raise ValueError(f"Changed manifest: {suite}")
    return json.loads(raw)


def inventory(base):
    bg = "bg-real-editor-look"
    for pair, case in manifest(base, bg)["cases"].items():
        for name, sha in case["artifacts"].items():
            if name != "comparison.png":
                yield "background", f"{bg}/{pair}/{name}", sha

    facd = "facd-two-image-baseline-v1"
    baseline = manifest(base, facd)
    aligned = read(ROOT / "docs/research/2026-11-18-facd-border-registration-evidence.json")
    for scene in baseline["references"]:
        for name, sha in aligned["cases"][scene]["artifacts"].items():
            if name == "Origin.png" or name.endswith("-overlay.png"):
                yield "facd", f"facd-border-registration-v1/{scene}/{name}", sha
    for name, sha in baseline["artifacts"].items():
        yield "facd", f"{facd}/{name}", sha

    film = "arri-two-looks-v2"
    for scene, case in manifest(base, film)["preparation"]["film_b"].items():
        for role, sha in case["images"].items():
            yield "film_b", f"{film}/film_b/{scene}-{role}.png", sha
    crossed = "arri-film-b-crossed-v1"
    for case in manifest(base, crossed)["cases"].values():
        for role, sha in case["output_sha256"].items():
            yield "film_b", f"{crossed}/{case['target']}-from-{case['reference']}-{role}.png", sha

    chart = "hdm-chart-pilot-v1"
    for role, sha in manifest(base, chart)["outputs_sha256"].items():
        yield "chart", f"{chart}/{role}.png", sha


def verify(base, specification):
    verified = {}
    for suite, name, expected in inventory(base):
        path = base / name
        with path.open("rb") as stream:
            actual = file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Changed artifact: {name}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError(f"Expected RGB: {name}")
        verified.setdefault(suite, {})[name] = actual
    counts = {suite: len(data) for suite, data in verified.items()}
    if counts != specification:
        raise ValueError(f"Changed comparison coverage: {counts}")
    return counts


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    args = cli.parse_args()
    print(json.dumps(verify(args.base, {"background": 40, "facd": 70, "film_b": 18, "chart": 4}), sort_keys=True))
