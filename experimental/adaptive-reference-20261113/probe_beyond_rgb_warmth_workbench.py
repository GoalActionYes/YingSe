"""Run both workbenches with the second physically matched warmth reference."""

from argparse import ArgumentParser
from pathlib import Path

import torch

from probe_gros_local_editor_channel import request, serving
from probe_matched_wall_warmth_workbench import SOURCE, STABLE_SHA, TARGETS, compare_target
from probe_target_mirror_chroma import CHECKPOINT, STRENGTH, checked_json, digest, json_to
from reference_color_app.pipeline import algorithm_fingerprint
from reference_color_app.server import make_server
from spatial_workbench import make_spatial_workbench
from probe_playraw_flash_room import read


PROTOCOL = "c2afa8839a62fa937b08cba9709cebdb6312bd3569413c9ec06600f6736e54d1"
DONOR_RECEIPT = "4811df036d89f4457e7f8d5779c0b2335a7dbdda151b2b882fb7fe8c8c1814ef"
DONOR = "beyond-rgb-near-white-v2/warm-verified-first"
REFERENCE = {"base": ("light-b", "off"), "captured": ("light-a", "off"), "edited": ("light-b", "on")}
DIRECTION = "light-b-to-light-a"


def run(base, out):
    torch.set_num_threads(4)
    script = Path(__file__)
    if digest(script.with_name("beyond-rgb-warmth-workbench-protocol.md")) != PROTOCOL:
        raise ValueError("Changed workbench comparison protocol")
    if algorithm_fingerprint() != STABLE_SHA or STRENGTH != 0.8:
        raise ValueError("Changed engine or strength")

    target_dir = base / "shared-local-warmth-v2"
    scenes = checked_json(target_dir / "results.json", SOURCE)["results"]
    for family, group in scenes.items():
        for name, scene in group.items():
            scene["folder"] = target_dir / family / name

    donor_dir = base / DONOR
    donor = checked_json(donor_dir / "results.json", DONOR_RECEIPT)
    if DIRECTION not in donor["matching_directions"]:
        raise ValueError("Fixed direction did not qualify")
    matched = donor["directions"][DIRECTION]
    paths, previews = {}, {}
    for role, (light, state) in REFERENCE.items():
        path = donor_dir / "render" / f"{light}-{state}.png"
        image, _, identity = read(path, False)
        frozen = donor["frames"][light]["files"][state]
        if any(identity[key] != frozen[key] for key in ("pixels_sha256", "icc_sha256")):
            raise ValueError(f"Changed reference: {path}")
        paths[role], previews[role] = path, image

    out.mkdir()
    result = {}
    for kind, factory, options in (
            ("stable", make_server, {}),
            ("spatial13", make_spatial_workbench, {"enhanced": True})):
        root = out / kind
        with serving(factory(root / "session", **options)) as server:
            if kind == "spatial13" and server.store.engine.checkpoint_sha != CHECKPOINT:
                raise ValueError("Workstation loaded the wrong checkpoint")
            ids = {role: request(server, "/api/media", frame.read_bytes(), f"white-blue-{frame.name}")["id"]
                   for role, frame in paths.items()}
            reports = {}
            for family, name, level in TARGETS:
                target = f"{family}-{name}"
                reports[target] = compare_target(server, scenes, ids, previews, family, name, level, root / target)
                print(kind, target, reports[target]["passed"], flush=True)
            result[kind] = reports

    json_to(out / "results.json", dict(targets_source=SOURCE, donor_receipt=DONOR_RECEIPT, donor=DONOR,
                                       direction=matched, reference=REFERENCE, script=digest(script),
                                       protocol=PROTOCOL, strength=STRENGTH, stable=STABLE_SHA,
                                       checkpoint=CHECKPOINT, workbenches=result))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
