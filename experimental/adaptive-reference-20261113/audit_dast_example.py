"""Qualify public DAST example links without exposing participant records or pixels."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from PIL import Image

REVISION = "839482cd09c487a1a0f9c342b0fd76626e04dc3e"
BLOB = "213db1a76bdf8a07545fbd90c7545764c3d0d836"
SIZE = 8043401
SHA256 = "96cfc50fe4d0c48f937235f2c4cea0f2a8e0e4868e514a0ad673575458b3526e"
EXPOSURES = {"optimal", "oe1", "oe2", "ue1", "ue2"}


def capture_key(path):
    name = PurePosixPath(path).stem
    match = re.fullmatch(r"(\d+-I\d+)(?:-\d+)?", name)
    if match is None:
        raise ValueError("Unexpected published capture name")
    return match[1]


def measurement_sites(data):
    rows = [row for row in csv.reader(io.StringIO(data.decode("utf-8-sig")), delimiter="\t")
            if any(field.strip() for field in row)]
    header = [name.strip() for name in rows[1]]
    columns = [header.index(key) for key in ("CIE-L*", "CIE-a*", "CIE-b*", "X", "Y", "Z")]
    facial_names = {"rw", "s", "lw", "measure 2", "measure 3", "measure 4"}
    facial, fields = 0, 0
    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        for index in columns:
            value = float(row[index].replace(",", "."))
            if not (-200 <= value <= 200):
                raise ValueError("Invalid instrument record")
        fields += 1
        facial += row[0].strip().lower() in facial_names
    if fields != 5 or facial != 3:
        raise ValueError("Incomplete instrument records")
    return fields, facial


def inspect(file):
    data = file.read_bytes()
    if (len(data) != SIZE or hashlib.sha256(data).hexdigest() != SHA256
            or hashlib.sha1(f"blob {SIZE}\0".encode() + data).hexdigest() != BLOB):
        raise ValueError("Unexpected DAST sample identity")
    with ZipFile(io.BytesIO(data)) as archive:
        members = [x for x in archive.infolist() if not x.is_dir() and x.filename.startswith("example-data/")]
        if sum(x.file_size for x in members) > 16 * 1024**2 or archive.testzip() is not None:
            raise ValueError("DAST sample archive failed integrity checks")
        pictures = {}
        for entry in members:
            if PurePosixPath(entry.filename).suffix.lower() != ".jpg":
                continue
            key = capture_key(entry.filename)
            if key in pictures:
                raise ValueError("Ambiguous photograph name")
            pictures[key] = entry.filename
        rows = list(csv.reader(io.StringIO(archive.read("example-data/example_overview.csv").decode("utf-8-sig")),
                               delimiter=";"))
        if len(rows) != len(pictures) or any(len(row) != 17 for row in rows):
            raise ValueError("Unexpected DAST manifest dimensions")
        joined = {}
        attributes, labeled, groups = Counter(), Counter(), defaultdict(Counter)
        direct = 0
        for row in rows:
            key = capture_key(row[3])
            path = pictures[key]
            subject = int(PurePosixPath(path).parts[1])
            if key in joined or subject != int(row[0]) or row[1] != key.split("-")[-1]:
                raise ValueError("Capture metadata is not uniquely linked")
            direct += PurePosixPath(row[3]).name == PurePosixPath(path).name
            joined[key] = subject
            with Image.open(io.BytesIO(archive.read(path))) as image:
                attributes[(image.format, image.mode, image.width, image.height,
                            "icc_profile" in image.info, bool(image.getexif()))] += 1
            camera, exposure = (row[index].strip().lower() for index in (8, 9))
            if not camera and not exposure:
                continue
            if camera not in ("sony", "canon") or exposure not in EXPOSURES:
                raise ValueError("Unexpected labeled camera or exposure")
            groups[subject, camera][exposure] += 1
            labeled[exposure] += 1
        measurements = {int(PurePosixPath(x.filename).parts[1]): measurement_sites(archive.read(x))
                        for x in members if PurePosixPath(x.filename).suffix == ".cmf"}
        participants = set(joined.values())
        if participants != set(measurements):
            raise ValueError("Missing measurement-to-capture link")
        usable = [(key, count) for key, count in groups.items() if count["optimal"]
                  and sum(count.values()) > count["optimal"]]
    return {
        "revision": REVISION, "git_blob": BLOB, "archive_size": SIZE, "archive_sha256": SHA256,
        "participants": len(participants), "example_photos": len(pictures),
        "links_exact_filename": direct, "links_unique_after_suffix_normalization": len(joined) - direct,
        "measurement_rows": sum(r[0] for r in measurements.values()),
        "facial_measurement_rows": sum(r[1] for r in measurements.values()),
        "exposure_counts": dict(sorted(labeled.items())), "without_camera_and_exposure": len(pictures) - sum(labeled.values()),
        "same_camera_usable_groups": len(usable), "groups_with_all_five_exposures": sum(EXPOSURES <= set(x) for _, x in usable),
        "participants_with_same_camera_exposure_pairs": len({key[0] for key, _ in usable}),
        "jpeg_headers": [{"format": form, "mode": mode, "width": w, "height": h, "icc": icc, "exif": exif, "count": n}
                         for (form, mode, w, h, icc, exif), n in sorted(attributes.items())],
        "qualified_for_exposure_development": len({key[0] for key, _ in usable}) >= 2,
        "photographic_pixels_decoded": 0, "models_run": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inspect(args.archive), indent=2) + "\n", encoding="utf-8", newline="\n")
