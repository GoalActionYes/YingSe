"""Measure a local warm-light pilot and prepare its four study exports."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image, ImageOps

from audit_capture_edit_pairs import audit
from probe_chromamill_blue_white_warmth import aligned
from probe_hdm_hdr_display import digest, json_to
from probe_inretouch_registration import RADIUS, UnmeasurableRegistration, gradient, shifts
from probe_playraw_flash_room import PP3, RT_CLI
from probe_shared_local_warmth import WARMTH_FIELDS, prepare
from render_rawtherapee_sky_pair import check
from triplet_color import to_srgb


RESEARCH_ROOT = Path(__file__).absolute().parents[2] / "outputs/adaptive-reference-20261113"
RT_RUNTIME = Path("rawtherapee-sky-pair-v1/runtime/RawTherapee_5.13_win64_x86_64_release")
RT_PROFILE = "17aebbdf8a88c39b07eb881dcd824eb1cf9828914d5e74a7324d7a31045e8871"


def read_inputs(first, second):
    frames, sources = {}, {}
    for light, path in (("light-a", first), ("light-b", second)):
        content = path.read_bytes()
        with Image.open(BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG"} or image.mode != "RGB":
                raise ValueError("Use RGB SDR JPEG or PNG captures")
            rgb, color = to_srgb(ImageOps.exif_transpose(image))
            frames[light] = np.array(rgb)
        sources[light] = {"sha256": sha256(content).hexdigest(), "color": color}
    if frames["light-a"].shape != frames["light-b"].shape:
        raise ValueError("A and B must have the same dimensions after orientation")
    if np.array_equal(frames["light-a"], frames["light-b"]):
        raise ValueError("A and B have identical RGB pixels")
    return frames, sources


def region_bounds(shape, edit, guards):
    height, width = shape[:2]
    if not guards:
        raise ValueError("Declare at least one protection region")
    areas = {"wall_left": tuple(edit)}
    areas.update({f"guard_{i}": tuple(guard) for i, guard in enumerate(guards, 1)})
    for name, (left, top, right, bottom) in areas.items():
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"{name} must be a nonempty rectangle inside the image")
        if name != "wall_left":
            x0, y0, x1, y1 = edit
            if max(left, x0) < min(right, x1) and max(top, y0) < min(bottom, y1):
                raise ValueError(f"{name} must not overlap the edit measurement region")
    return areas


def registration(first, second):
    if min(first.shape[:2]) <= 2 * RADIUS + 3:
        return {"passed": False, "reason": "Frame is too small for the registration window"}
    try:
        score = shifts(gradient(first), gradient(second))[0]
    except UnmeasurableRegistration:
        return {"passed": False, "reason": "No measurable structure for gradient registration"}
    return {"passed": aligned(score), **score}


def run(first, second, capture_record, out, edit, guards, research_root):
    note = capture_record.read_bytes()
    if not note.decode("utf-8-sig").strip():
        raise ValueError("Capture record must describe the setup and light change")
    frames, sources = read_inputs(first, second)
    areas = region_bounds(frames["light-a"].shape, edit, guards)
    alignment = registration(frames["light-a"], frames["light-b"])
    if not alignment["passed"]:
        raise ValueError(f"Registration is unconfirmed: {alignment}")

    out = out.absolute()
    research_root = research_root.absolute()
    cli = research_root / RT_RUNTIME / "rawtherapee-cli.exe"
    if not cli.as_posix().isascii() or not out.as_posix().isascii():
        raise ValueError("Map the RawTherapee runtime and output directory to ASCII paths")
    check(cli, RT_CLI)
    template = check(research_root / "playraw-flash-room-intake-v1/martin.pp3", PP3)
    icc = check(cli.parent / "iccprofiles/output/RTv4_sRGB.icc", RT_PROFILE)
    out.mkdir()
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "rt-cache"), OMP_NUM_THREADS="1")
    rendered = out / "render"
    scored = prepare(cli, template, icc, env, rendered, frames, areas, {light: True for light in frames})

    material_root = out / "materials"
    pair = material_root / "first-pair"
    pair.mkdir(parents=True)
    (pair / "capture.md").write_bytes(note)
    (pair / "edit.md").write_text(
        f"RawTherapee 5.13 fixed local warmth. Template SHA-256: {PP3}\n"
        "The same edit is switched off/on for both lights. See ../../render/on.pp3 and ../../render/off.pp3.\n",
        encoding="utf-8",
    )
    for light in frames:
        copyfile(rendered / f"{light}-off.png", pair / f"{light}-before.png")
        copyfile(rendered / f"{light}-on.png", pair / f"{light}-after.png")
    inventory = audit(material_root)
    numerical_matches = [name for name, direction in scored["directions"].items() if direction["passed"]]
    report = {
        "capture_provenance_requires_review": True,
        "local_material_and_landmark_registration_requires_review": True,
        "capture_record_sha256": sha256(note).hexdigest(),
        "script_sha256": digest(Path(__file__)),
        "sources": sources,
        "alignment": alignment,
        "editor": RT_CLI,
        "template": PP3,
        "profile": RT_PROFILE,
        "edit_fields": WARMTH_FIELDS,
        "inventory": inventory,
        "matching_directions": numerical_matches if inventory["inventory_complete"] else [],
        **scored,
    }
    json_to(out / "results.json", report)
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="Fixed-render sRGB SDR export under light A")
    parser.add_argument("second", type=Path, help="Fixed-render sRGB SDR export under light B")
    parser.add_argument("capture_record", type=Path, help="UTF-8 local capture notes")
    parser.add_argument("out", type=Path, help="Fresh local output directory")
    coords = ("LEFT", "TOP", "RIGHT", "BOTTOM")
    parser.add_argument("--edit", required=True, type=int, nargs=4, metavar=coords)
    parser.add_argument("--guard", required=True, type=int, nargs=4, metavar=coords, action="append",
                        dest="guards")
    parser.add_argument("--research-root", type=Path, default=RESEARCH_ROOT)
    args = parser.parse_args()
    try:
        report = run(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    for direction, row in report["directions"].items():
        failed = ", ".join(key for key, passed in row["checks"].items() if not passed)
        print(f"{direction}: cosine={row['cosine']}, ratio={row['ratio']}, failed={failed or 'none'}")
    if not report["inventory"]["inventory_complete"]:
        for name, group in report["inventory"]["groups"].items():
            issues = ", ".join(group["issues"])
            print(f"{name}: {issues}")
    print(f"Numerical matches with complete inventory: {len(report['matching_directions'])}")
    print(f"Full report: {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
