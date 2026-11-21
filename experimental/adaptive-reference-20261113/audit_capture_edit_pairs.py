"""Check the local inventory of crossed capture and edit photo pairs."""

from argparse import ArgumentParser
from hashlib import file_digest, sha256
import json
from pathlib import Path

from PIL import Image, ImageOps


ROLES = ("light-a-before", "light-a-after", "light-b-before", "light-b-after")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
NOTES = ("capture.md", "edit.md")


def check_group(group, root):
    issue = []
    frames = {}
    if not group.resolve().is_relative_to(root):
        return {"issues": ["group_outside_intake_root"], "frames": frames}
    paths = list(group.iterdir())
    for role in ROLES:
        candidates = [p for p in paths if p.stem.lower() == role and p.suffix.lower() in IMAGE_SUFFIXES]
        if len(candidates) != 1:
            issue.append(f"{role}:expected_one_image_found_{len(candidates)}")
            continue
        path = candidates[0]
        if not path.resolve().is_relative_to(group.resolve()) or not path.is_file():
            issue.append(f"{role}:expected_local_file")
            continue
        try:
            with path.open("rb") as stream:
                identity = file_digest(stream, "sha256").hexdigest()
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                width, height = rgb.size
                pixels = sha256(f"{width}x{height}:RGB:".encode("ascii") + rgb.tobytes()).hexdigest()
        except (OSError, ValueError, Image.DecompressionBombError):
            issue.append(f"{role}:unreadable_image")
            continue
        frames[role] = {"sha256": identity, "pixel_sha256": pixels, "size": [width, height]}
    if len({tuple(frame["size"]) for frame in frames.values()}) > 1:
        issue.append("output_dimensions_differ")
    for name in NOTES:
        note = group / name
        try:
            present = (note.resolve().is_relative_to(group.resolve()) and note.is_file()
                       and bool(note.read_text(encoding="utf-8-sig").strip()))
        except (OSError, UnicodeError):
            present = False
        if not present:
            issue.append(f"{name}:missing_or_empty")
    return {"issues": issue, "frames": frames}


def audit(root):
    root = root.resolve(strict=True)
    groups = {group.name: check_group(group, root)
              for group in sorted(root.iterdir()) if group.is_dir()}
    seen = {}
    for name, group in groups.items():
        for role, frame in group["frames"].items():
            seen.setdefault(frame["pixel_sha256"], []).append((name, role))
    duplicate_pixels = [matches for matches in seen.values() if len(matches) > 1]
    for matches in duplicate_pixels:
        for name, role in matches:
            groups[name]["issues"].append(f"{role}:duplicate_pixels")
    return {"inventory_complete": bool(groups) and all(not group["issues"] for group in groups.values()),
            "requires_manual_capture_and_edit_review": True, "groups": groups,
            "duplicate_pixels": duplicate_pixels}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Dedicated local folder containing only sample group directories")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("Intake root must be a directory")
    report = audit(args.root)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    raise SystemExit(0 if report["inventory_complete"] else 1)
