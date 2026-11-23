"""Acquire and check fixed-ratio lamp brightness controls from the Gros release."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile
from zlib import crc32

import numpy as np
from PIL import Image, ImageDraw

from audit_gros_portrait_light import MIB, PublicRange, exposure, write
from audit_gros_remaining_archives import CAPTURE_FIELDS, RELEASES
from triplet_color import to_srgb


BASELINE = {
    2: ("gros-remaining-archives-v1", "47ace6b9d6dfac8139b1ae82753573350179eac2cadf0e31f0eb8c17edaad3b0"),
    4: ("gros-part-four-recovery-20261122", "f887cebc94626dd13e76202516c4e408bb98b4544923532539915a2b09dd8afd"),
}
REGIONS_HASH = "0f8508e2b6d195d5c42a5e145590435d3c9f4247883170cf52bf3679ccdad88a"
LABELS = ("warm", "cold", "low", "medium", "high")
EXIF_FIELDS = (*CAPTURE_FIELDS, "exposure_time", "exposure_mode")
LUMA = np.array([0.2126, 0.7152, 0.0722])


def checked(path, fingerprint):
    data = path.read_bytes()
    if sha256(data).hexdigest() != fingerprint:
        raise ValueError(f"Input identity changed: {path}")
    return data


def selection(archive, photos):
    warm, cold = (PurePosixPath(photos[label]["member"]) for label in LABELS[:2])
    members = [item for item in archive.infolist() if not item.is_dir()
               and PurePosixPath(item.filename).parent == warm.parent]
    by_number = {}
    for member in members:
        name = PurePosixPath(member.filename).name
        match = re.fullmatch(r"IMG_(\d{4})\.JPG", name, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Unexpected sequence member: {member.filename}")
        by_number[int(match[1])] = member
    numbers = sorted(by_number)
    if len(members) != 5 or len(numbers) != 5 or numbers != list(range(numbers[0], numbers[0] + 5)):
        raise ValueError("Expected five consecutive camera frames")
    ordered = [by_number[n] for n in numbers]
    if [item.filename for item in ordered[:2]] != [str(warm), str(cold)]:
        raise ValueError("Frozen warm and cold identities not at start of sequence")
    return ordered


def pixels_and_record(payload, member):
    if len(payload) != member.file_size or crc32(payload) != member.CRC:
        raise ValueError(f"JPEG size or CRC mismatch: {member.filename}")
    settings = exposure(payload)
    with Image.open(BytesIO(payload)) as image:
        image.load()
        profile = image.info.get("icc_profile")
        rgb, color = to_srgb(image)
        pixels = rgb.resize((384, 256), Image.Resampling.LANCZOS)
    return pixels, dict(member=member.filename, bytes=len(payload), crc32=f"{member.CRC:08x}",
                        sha256=sha256(payload).hexdigest(), settings=settings,
                        color=dict(policy=color["policy"], icc_sha256=sha256(profile).hexdigest() if profile else None))


def summarize(pixels, regions):
    result = {}
    for name, rect in regions.items():
        rgb = np.asarray(pixels.crop(rect), dtype=np.float64).reshape(-1, 3)
        srgb = rgb / 255
        linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
        total = linear.sum(axis=1)
        fractions = linear[total > 0] / total[total > 0, None]
        result[name] = dict(y_median=float(np.median(rgb @ LUMA)),
                            endpoint_fraction=float(((rgb == 0) | (rgb == 255)).any(axis=1).mean()),
                            linear_channel_fraction_median=np.median(fractions, axis=0).tolist() if fractions.size else None)
    return result


def admit(records):
    baseline = records["warm"]
    matches = {field: baseline["settings"][field] is not None and all(
        row["settings"][field] == baseline["settings"][field] for row in records.values()) for field in EXIF_FIELDS}
    matches["color_policy"] = all(row["color"] == baseline["color"] for row in records.values())
    checks = {}
    for region in ("cheek_left", "cheek_right"):
        rows = [records[label]["regions"][region] for label in LABELS[2:]]
        lo, mid, hi = widths = [row["y_median"] for row in rows]
        checks[region] = dict(y_median=widths, high_minus_low=hi-lo, passed=lo < mid < hi and hi-lo >= 3
                              and max(row["endpoint_fraction"] for row in rows) < .02)
    return dict(fields_match=matches, cheeks=checks,
                numeric_qualified=all(matches.values()) and all(row["passed"] for row in checks.values()))


def run(base, output, proxy):
    coords_file = Path(__file__).with_name("gros-local-editor-cross-regions.json")
    coords = json.loads(checked(coords_file, REGIONS_HASH))
    output.mkdir()
    board = Image.new("RGB", (384 * len(LABELS), 280 * len(BASELINE)), "#20242c")
    draw = ImageDraw.Draw(board)
    report, transferred = {}, 0
    for row, (part, dataset, file, size, tag) in enumerate(release for release in RELEASES if release[0] in BASELINE):
        path, prior_hash = BASELINE[part]
        prior = base / path
        receipt = json.loads(checked(prior / f"part-{part}.json", prior_hash))["selected"]["photos"]
        folder = output / f"part-{part}"
        folder.mkdir()
        by_region = {**coords["screen"], **{key: coords[str(part)][key] for key in ("cheek_left", "cheek_right")}}
        url = f"https://data.mendeley.com/public-files/datasets/{dataset}/files/{file}/file_downloaded"
        reader = PublicRange(40*MIB-transferred, proxy, url=url, archive_size=size, archive_tag=f'"{tag}"')
        with ZipFile(reader) as archive:
            members = selection(archive, receipt)
            if sum(member.file_size for member in members[2:]) > reader.limit - reader.transferred:
                raise ValueError("JPEG budget would be exceeded")
            records = {}
            for col, (label, member) in enumerate(zip(LABELS, members, strict=True)):
                if label in receipt:
                    payload = checked(prior / f"part-{part}" / f"{label}.jpg", receipt[label]["sha256"])
                else:
                    payload = archive.read(member)
                pixels, record = pixels_and_record(payload, member)
                if label not in receipt:
                    (folder / f"{label}.jpg").write_bytes(payload)
                record["regions"] = summarize(pixels, by_region)
                records[label] = record
                board.paste(pixels, (384 * col, 280 * row + 24))
                draw.text((384 * col + 8, 280 * row + 6), f"{part} / {label}", fill="white")
                for x0,y0,x1,y1 in by_region.values():
                    draw.rectangle((384*col+x0,280*row+24+y0,384*col+x1-1,280*row+24+y1-1), outline="#5affcc")
                print(f"part {part}: {label} verified", flush=True)
        transferred += reader.transferred
        report[str(part)] = dict(archive=dataset, archive_bytes=size, etag=tag, baseline_receipt_sha256=prior_hash,
                                 transferred_bytes=reader.transferred, photos=records, admission=admit(records))
        write(folder / "results.json", report[str(part)])
    board.save(output / "level-comparison.png")
    write(output / "results.json", dict(parts=report, total_transfer_bytes=transferred, regions_sha256=REGIONS_HASH,
                                         board_sha256=sha256((output/"level-comparison.png").read_bytes()).hexdigest()))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--proxy")
    run(**vars(parser.parse_args()))
