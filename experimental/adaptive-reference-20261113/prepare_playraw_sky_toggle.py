"""Prepare paired sidecars by toggling one active-history exposure bit."""
import argparse
import hashlib
from pathlib import Path
import re

from audit_playraw_sky_xmp import audit

CASES = {
    "A": ("sky-A", 16, False, "2c40ede1b4e183bd2c534d4b6783804ebf68a6ff7a81fb8be101475cd77d59fb"),
    "B": ("sky-B2", 28, True, "ed9da91b50e550b5faa838735a4dead00f00eec8f4742f261371cd2173a03d90"),
}


def toggle(source, step, original):
    wanted = f'darktable:num="{step}"'.encode()
    candidates = [row for row in re.finditer(rb"<rdf:li\s+[^>]*>", source) if wanted in row[0]]
    if len(candidates) != 1:
        raise ValueError(f"Expected one history row for step {step}")
    match, = candidates
    item = match[0]
    before = b'darktable:enabled="1"' if original else b'darktable:enabled="0"'
    after = b'darktable:enabled="0"' if original else b'darktable:enabled="1"'
    if item.count(before) != 1 or b'darktable:operation="exposure"' not in item:
        raise ValueError("Unexpected selected exposure step")
    return source[:match.start()] + item.replace(before, after) + source[match.end():]


def prepare(root, out):
    for case, (stem, step, original, expected) in CASES.items():
        data = (root / f"{stem}.xmp").read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Unexpected source identity for {case}")
        changed = toggle(data, step, original)
        if toggle(changed, step, not original) != data or sum(a != b for a, b in zip(data, changed, strict=True)) != 1:
            raise ValueError(f"Toggle did not round-trip for {case}")
        result = {}
        for enabled in (False, True):
            dest = out / f"{case}-{'on' if enabled else 'off'}.xmp"
            dest.write_bytes(data if enabled == original else changed)
            result[enabled] = audit(dest)
        if result[True]["active_masked_steps"] != [step] or result[False]["active_masked_steps"]:
            raise ValueError(f"Unexpected active exposure after toggle for {case}")
        print(case, result[True]["raw"], result[False]["sha256"], result[True]["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.input, args.output)
