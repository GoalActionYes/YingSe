"""Render two pinned author PP3s with their local adjustment switch on or off."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import re
import subprocess

SOURCES = {
    "P1005231": ("RW2", "241b75a13e95fa1f3be53d9eef810658374e8375021ab3f3389bb407c90b2b7a",
                "c49fb758340aef471beedaa39106cf5cc77d95f3800b57f0f5abd675277b83c0"),
    "R6_06283": ("CR3", "a737bc15d5bf22b6936bca3e6602dd14ed71c92e858066a79c420cb7e994f284",
                 "eb44da59228162de892a3cf1d7d0db38dc2d3744cf096d3bfd5b4b2b8916b899"),
}
def disabled(data, key=b"Enabled", section=b"Locallab"):
    pattern = re.compile(rb"(?ms)^\[" + re.escape(section) + rb"\]\r?\n(.*?)(?=^\[|\Z)")
    local = list(pattern.finditer(data))
    if len(local) != 1:
        raise ValueError("Expected one processing-profile section")
    pattern = re.compile(rb"(?m)^" + re.escape(key) + rb"=(true)(?=\r?$)")
    field = list(pattern.finditer(local[0][1]))
    if len(field) != 1:
        raise ValueError("Expected one enabled local-adjustment switch")
    start = local[0].start(1) + field[0].start(1)
    end = start + 4
    return data[:start] + b"false" + data[end:]


def check(path, expected):
    value = path.read_bytes()
    if sha256(value).hexdigest() != expected:
        raise ValueError(f"Unexpected bytes: {path.name}")
    return value


def render(cli, original, profile, output, env, depth=8):
    process = subprocess.run(
        [str(cli), "-o", str(output), "-q", "-p", str(profile), "-n", f"-b{depth}", "-c", str(original)],
        cwd=cli.parent, env=env, capture_output=True, timeout=120,
    )
    output.with_suffix(".log").write_bytes(process.stdout + b"\n" + process.stderr)
    if process.returncode or not output.is_file():
        raise RuntimeError(f"CLI exited {process.returncode}: {original.name} / {profile.name}")


def run(source, cli, out, state, threads):
    source, cli, out = source.resolve(strict=True), cli.resolve(strict=True), out.resolve()
    out.mkdir(exist_ok=False)
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "cache"),
               OMP_NUM_THREADS=str(threads))
    for name, (extension, raw_hash, profile_hash) in SOURCES.items():
        original = source / f"{name}.{extension}"
        prior = check(source / f"{name}-author.pp3", profile_hash)
        check(original, raw_hash)
        profile = out / f"{name}-{state}.pp3"
        profile.write_bytes(prior if state == "on" else disabled(prior))
        output = out / f"{name}-{state}.png"
        render(cli, original, profile, output, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("cli", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("state", choices=("off", "on"))
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    run(**vars(parser.parse_args()))
