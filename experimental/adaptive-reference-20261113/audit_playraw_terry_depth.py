"""Compare fixed RGB8 and RGB16 exports and trace the old endpoint pixels."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_playraw_terry_opencl import REGIONS, STATES, STEP, load_render
from probe_playraw_flash_room import RAW, XMP, darktable_off
from probe_playraw_terry_opencl import DT50, read_16
from probe_target_mirror_chroma import digest, json_to, panel


REFERENCE_SHA = "1ae45310c965b226a0080f10c19520fc9340f88ae6bf94e2a7687054b68dba9f"
ENDPOINT_LIMIT = .02


def quantize(values):
    return ((values.astype(np.uint32) + 128) // 257).astype(np.uint8)


def endpoint_channels(off, on):
    high = np.iinfo(on.dtype).max
    return ((on == 0) & (off > 0)) | ((on == high) & (off < high))


def trace_endpoints(old_off, old_on, high_off, high_on):
    off8, on8 = quantize(high_off), quantize(high_on)
    old = endpoint_channels(old_off, old_on)
    native = endpoint_channels(high_off, high_on)
    matched = (old_off == off8) & (old_on == on8)
    any_old = old.any(axis=2)
    persistent = (old & native & matched).any(axis=2)
    rounding_only = any_old & ~persistent & (~old | matched).all(axis=2)
    unresolved = any_old & ~(persistent | rounding_only)
    return dict(old=any_old, persistent=persistent, rounding_only=rounding_only, unresolved=unresolved,
                native=native.any(axis=2), matched_old_channels=old & matched,
                unmatched_old_channels=old & ~matched)


def self_check():
    off = np.array([[[10, 30, 50], [10, 30, 50], [10, 30, 50], [10, 30, 50]]], np.uint8)
    on = off.copy()
    on[0, :3, 0] = 0
    hi_off = off.astype(np.uint16) * 257
    hi_on = hi_off.copy()
    hi_on[0, :3, 0] = (0, 30, 180)
    hi_off[0, 3, 0] = 65435
    hi_on[0, 3, 0] = 65535
    actual = trace_endpoints(off, on, hi_off, hi_on)
    assert actual["persistent"].tolist() == [[True, False, False, False]]
    assert actual["rounding_only"].tolist() == [[False, True, False, False]]
    assert actual["unresolved"].tolist() == [[False, False, True, False]]
    assert actual["native"].tolist() == [[True, False, False, True]]


def stages(log):
    content = log.read_text(encoding="utf-8", errors="replace")
    sequence = STEP.findall(content)
    native_step = [line for line in content.splitlines() if "demosaic inplace" in line]
    if (len(native_step) != 1 or "CL0 [export]" not in native_step[0]
            or native_step[0].count("6240x4156 scale=1.0000") != 2
            or "demosaic approx zoom" in content or "NVIDIA GeForce RTX 4060 Laptop GPU" not in content):
        raise ValueError(f"Changed OpenCL capture pipeline: {log.parent.name}")
    return sequence


def high_frame(root, state, author):
    enabled = state.startswith("on")
    folder = root / state
    path, log = (folder / name for name in ("image.tif", "darktable.log"))
    expected = author if enabled else darktable_off(author)
    receipt = json.loads((folder / "render.json").read_bytes())
    if (not receipt["render_completed"] or receipt["exit_code"] != "0x00000000"
            or (receipt["raw_sha256"], receipt["author_xmp_sha256"], receipt["renderer_sha256"]) != (RAW, XMP, DT50)
            or (folder / "settings.xmp").read_bytes() != expected
            or receipt["settings_sha256"] != sha256(expected).hexdigest()
            or digest(path) != receipt["output"]["tiff_sha256"] or digest(log) != receipt["log_sha256"]):
        raise ValueError(f"Unverified RGB16 source: {state}")
    rgb, mask, identity = read_16(path, enabled)
    with Image.open(path) as tiff:
        profile = tiff.info["icc_profile"]
    normalized_profile = sha256(profile[:24] + profile[36:]).hexdigest()
    if (rgb.shape != (682, 1024, 3) or identity["pixels_sha256"] != receipt["output"]["pixels_sha256"]
            or identity["mask_sha256"] != receipt["output"]["mask_sha256"]):
        raise ValueError(f"Changed RGB16 pixels: {state}")
    ledger = dict(identity, tiff_sha256=digest(path), log_sha256=digest(log), normalized_profile=normalized_profile)
    return rgb, mask, ledger


def comparison_layout(out, old, high, traces):
    base = old["off"][0] // 3
    show = lambda keys: {label: np.where(traces[key][:, :, None], color, base)
                         for label, key, color in keys}
    markers = [("still endpoint", "persistent", [255, 45, 110]),
               ("rounding only", "rounding_only", [45, 255, 138]),
               ("unresolved", "unresolved", [255, 190, 40])]
    return panel(out / "comparison.png", {
        "8 bit": {state: old[state][0] for state in ("off", "on")},
        "16 to 8 bit": {state: quantize(high[state][0]) for state in ("off", "on")},
        "old endpoint attribution": {key: value.astype(np.uint8) for key, value in show(markers).items()},
    })


def run(base, high_root, out):
    self_check()
    old_root = base / "playraw-terry-opencl-v1"
    baseline = base / "playraw-terry-opencl-audit-v1" / "results.json"
    if digest(baseline) != REFERENCE_SHA:
        raise ValueError("Changed original endpoint audit")
    prior = json.loads(baseline.read_bytes())
    source = base / "playraw-flash-room-intake-v1"
    author = (source / "terry.xmp").read_bytes()
    if (digest(source / "terry.xmp"), digest(source / "DSCF8379.raf")) != (XMP, RAW):
        raise ValueError("Changed frozen source")
    old = {state: load_render(old_root / state, state, author) for state in STATES}
    high = {state: high_frame(high_root, state, author) for state in STATES}
    logs = {state: (stages(old_root/state/"darktable.log"), stages(high_root/state/"darktable.log"))
            for state in STATES}
    module_match = all(a == b for a, b in logs.values())
    dither = any("dither" in module for pair in logs.values() for engine in pair for _, module in engine)
    profile_match = all(old[state][2]["profile_transform"] == high[state][2]["normalized_profile"] for state in STATES)
    repeated = {state: bool(np.array_equal(high[state][0], high[state+"-repeat"][0]))
                for state in ("off", "on")}
    repeated_mask = bool(np.array_equal(high["on"][1], high["on-repeat"][1]))
    same_path = module_match and profile_match and not dither and all(repeated.values()) and repeated_mask

    frames = {state: dict(id=high[state][2]) for state in STATES}
    for state in STATES:
        diff = np.abs(quantize(high[state][0]).astype(np.int16) - old[state][0])
        frames[state].update(max_roundtrip_rgb8=int(diff.max()), differing_samples=int(np.count_nonzero(diff)),
                             differing_pixels=int(diff.any(axis=2).sum()), over_one_samples=int((diff > 1).sum()))
    traces = trace_endpoints(old["off"][0], old["on"][0], high["off"][0], high["on"][0])
    counts = {key: int(value.sum()) for key, value in traces.items()}
    if counts["old"] != prior["endpoints"]["pixels"] or sum(
            counts[x] for x in ("persistent", "rounding_only", "unresolved")) != counts["old"]:
        raise ValueError("Endpoint attribution does not partition the frozen count")
    counts["native_only"] = int((traces["native"] & ~traces["old"]).sum())
    total = int(traces["old"].size)
    normalized = (high["on"][0].astype(np.float64) - high["off"][0]) / 257
    warmth = .5 * (normalized[:, :, 0] - normalized[:, :, 2])
    registry = Path(__file__).with_name("playraw-terry-opencl-regions.json")
    if digest(registry) != REGIONS or old["off"][2]["pixel_sha256"] != prior["identities"]["off"]["pixel_sha256"]:
        raise ValueError("Changed region registration")
    boxes = json.loads(registry.read_bytes())["rectangles"]
    regions = {}
    for name, (left, top, right, bottom) in boxes.items():
        zone = np.s_[top:bottom, left:right]
        regions[name] = dict(warmth_median_rgb8_equivalent=float(np.median(warmth[zone])),
                             warmth_abs_p95_rgb8_equivalent=float(np.percentile(np.abs(warmth[zone]), 95)),
                             pixel_identical=not bool(normalized[zone].any()))
    old_mask = old["on"][1]
    mask_difference = np.abs(quantize(high["on"][1]).astype(np.int16) - old_mask)
    out.mkdir()
    code = Path(__file__)
    board = comparison_layout(out, old, high, traces)
    json_to(out / "results.json", dict(reference_sha256=REFERENCE_SHA,
        script=digest(code), protocol=digest(code.with_name("playraw-terry-depth-protocol.md")),
        render_script=digest(code.with_name("probe_playraw_terry_opencl.py")),
        module_match=module_match, profile_match=profile_match, dither_executed=dither,
        strict_comparison=same_path, frame_shape=list(high["off"][0].shape), repeated=repeated,
        repeated_mask=repeated_mask, mask_roundtrip=dict(max_difference=int(mask_difference.max()),
            changed_pixels=int(np.count_nonzero(mask_difference))), frames=frames, regions=regions,
        endpoint_counts=counts, pixel_count=total, original_fraction=counts["old"]/total,
        native_fraction=counts["native"]/total, persistent_lower_fraction=counts["persistent"]/total,
        maximum_eligible_count=total//50, endpoint_limit=ENDPOINT_LIMIT,
        still_over_limit=same_path and counts["native"] > total//50, comparison_sha256=board))
    print(same_path, counts, counts["native"]/total, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("base", "high_root", "out"):
        parser.add_argument(key, type=Path)
    run(**vars(parser.parse_args()))
