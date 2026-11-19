"""Render the authenticated thumper RAW and Desmis graduated-filter switch."""
import argparse
import os
from pathlib import Path

from audit_rawtherapee_sky_pair import read_png
from probe_rawtherapee_local_tool_attribution import CLI_SHA
from render_rawtherapee_sky_pair import check, disabled, render

RAW_SHA = "41067ff0b376556d3c79a7fa13111c5b32ae5b14703b0e2c08d6d4ed0a9ef235"
PP3_SHA = "bbf6626f77448ae26163787bbbafcb3e812de234f58b31f7271e8cfccd10b180"


def run(base, cli, out, state):
    base, cli, out = base.resolve(strict=True), cli.resolve(strict=True), out.resolve()
    raw = base / "1Q8A5461.CR3"
    check(raw, RAW_SHA)
    check(cli, CLI_SHA)
    original = check(base / "author.pp3", PP3_SHA)
    out.mkdir()
    profile = out / f"{state}.pp3"
    profile.write_bytes(original if state == "on" else disabled(original, section=b"Gradient"))
    target = out / f"{state}.png"
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "cache"), OMP_NUM_THREADS="1")
    render(cli, raw, profile, target, env)
    read_png(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("cli", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("state", choices=("off", "on"))
    run(**vars(parser.parse_args()))
