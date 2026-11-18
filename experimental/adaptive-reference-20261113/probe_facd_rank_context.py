"""Hold source RGB fixed and check whether within-photo tonal rank adds information."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from audit_bg_real_skin_regions import BASE, ROOT, digest


ALIGNED_SHA = "bdbb3a3f0d331b69bf2334df9472d59c77a5fe6cafa1551ea16df45e6f53e2f2"
STYLES = ("Amaro", "Hudson", "XProII")


def load(family, name, expected):
    path = BASE/"facd-border-registration-v1"/family/name
    if digest(path) != expected[name]:
        raise ValueError("Frozen aligned photo changed")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def summarize(original, edits):
    flat = original.reshape(-1, 3).astype(np.int32)
    y = np.rint(flat @ [.2126, .7152, .0722]).astype(int)
    neutral = (np.ptp(flat, axis=1) <= 18) & (y >= 24) & (y <= 231)
    levels = np.bincount(y, minlength=256)
    ranks = (levels.cumsum() - levels/2) / len(y)
    codes = (flat[:, 0] << 16) + (flat[:, 1] << 8) + flat[:, 2]
    keys, first, inverse, count = np.unique(codes, return_index=True, return_inverse=True, return_counts=True)
    means = {style: np.column_stack([
        np.bincount(inverse, weights=img.reshape(-1, 3)[:, channel])/count for channel in range(3)
    ]) for style, img in edits.items()}
    return keys, inverse, ranks[y[first]], count, means, neutral


def pair_lookup(query, sources):
    keys, _, ranks, _, _, _ = query
    mapped, distances, shared = [], [], []
    for dk, _, dr, count, _, _ in sources:
        pos = np.minimum(np.searchsorted(dk, keys), len(dk)-1)
        valid = (dk[pos] == keys) & (count[pos] >= 4)
        mapped.append(pos)
        distances.append(np.where(valid, np.abs(dr[pos]-ranks), np.inf))
        shared.append(valid)
    mapped, distances, shared = map(np.asarray, (mapped, distances, shared))
    return mapped, shared, distances.argmin(axis=0), shared.sum(axis=0) >= 2


def blend(original, edited):
    return np.rint(.2*original + .8*edited).astype(np.uint8)


def metrics(control, approach, author, overall, neutral):
    neutral_support = overall & neutral
    def score(mask):
        if not mask.any():
            return None
        return [float(np.abs(x[mask].astype(np.int16) - author[mask]).mean()) for x in (control, approach)]
    whole, near = score(overall), score(neutral_support)
    coverage = float(overall.mean())
    count = int(neutral_support.sum())
    passed = coverage >= .10 and count >= 256 and whole[0]-whole[1] >= max(1, .2*whole[0]) and near[1] <= near[0]+.5
    return {"coverage": coverage, "pixels": int(overall.sum()), "neutral_pixels": count,
            "mae": whole, "neutral_mae": near, "passed": passed}


def panel(path, original, control, selected, truth, support):
    sheet = Image.new("RGB", (1308, 294), "#20242a")
    draw = ImageDraw.Draw(sheet)
    mask = np.where(support.reshape(original.shape[:2])[..., None], [76, 191, 137], [36, 40, 49]).astype(np.uint8)
    for i, (label, pixels) in enumerate(zip(("Source", "Scored support", "Shared RGB", "RGB + tone rank", "Authored 80%"),
                                            (original, mask, control, selected, truth), strict=True)):
        x = 8 + i*260
        draw.text((x, 8), label, fill="white")
        im = Image.fromarray(pixels)
        im.thumbnail((252, 254), Image.Resampling.LANCZOS)
        sheet.paste(im, (x, 27))
    sheet.save(path)


def run(folder):
    manifest = ROOT/"docs/research/2026-11-18-facd-border-registration-evidence.json"
    if digest(manifest) != ALIGNED_SHA:
        raise ValueError("Frozen alignment changed")
    cases = {k: v for k, v in json.loads(manifest.read_bytes())["cases"].items()
             if all(x["overlay"]["passed"] for x in v["checks"].values())}
    if len(cases) != 7:
        raise ValueError("Unexpected cohort")
    photos, references = {}, {}
    for family, row in sorted(cases.items(), key=lambda item: int(item[0])):
        checked = row["artifacts"]
        original = load(family, "Origin.png", checked)
        edits = {style: load(family, f"{style}-overlay.png", checked) for style in STYLES}
        photos[family] = (original, edits)
        references[family] = summarize(original, edits)
    folder.mkdir()
    results = {}
    for family, (image, edits) in photos.items():
        query = references[family]
        donors = [v for k, v in references.items() if k != family]
        mapped, shared, best, has_color = pair_lookup(query, donors)
        _, inverse, _, _, _, neutral = query
        target = image.reshape(-1, 3)
        support = has_color[inverse]
        results[family] = {}
        for style in STYLES:
            stack = np.stack([source[4][style][row] for source, row in zip(donors, mapped, strict=True)])
            mean = (stack * shared[..., None]).sum(axis=0)/np.maximum(shared.sum(axis=0), 1)[:, None]
            predicted = stack[best, np.arange(len(best))]
            blank, selected = target.copy(), target.copy()
            blank[support] = blend(target[support], mean[inverse[support]])
            selected[support] = blend(target[support], predicted[inverse[support]])
            expected = blend(target, edits[style].reshape(-1, 3))
            values = metrics(blank, selected, expected, support, neutral)
            path = folder/f"{family}-{style}.png"
            panel(path, image, blank.reshape(image.shape), selected.reshape(image.shape), expected.reshape(image.shape), support)
            results[family][style] = {**values, "panel_sha256": digest(path)}
    counts = {style: sum(int(results[f][style]["passed"]) for f in results) for style in STYLES}
    report = {"families": results, "passes": counts, "passed": sum(n >= 5 for n in counts.values()) >= 2,
              "alignment_sha256": ALIGNED_SHA, "source_sha256": digest(Path(__file__)),
              "protocol_sha256": digest(Path(__file__).with_name("facd-rank-context-protocol.md"))}
    (folder/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
