"""Read fixed Gros release directories and acquire a bounded new capture sample."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from PIL import Image

from audit_gros_portrait_light import MIB, PublicRange, exposure, write


RELEASES = (
    (2, "ycjd7mdsbs", "1fe86929-918c-4aa0-88f9-28569d23a5d5", 8253791949, "0fce7915b032582dea2ded957c0a5828-1575"),
    (3, "55wmmr8j3g", "31db0cfd-ebe7-4ad1-a459-4d991ca29b6b", 8440524018, "06c355c3db343fe629681beee5569f74-1610"),
    (4, "zsv3bgp7t4", "8c484765-2e68-40aa-a2ac-4a0f221d3d2f", 5738181865, "afe18729eea0ce82119caad983251260-1095"),
)
PART_ONE_CATALOG = "f56abe78071c493298c359db34c784858663129fafbafe14e8fbb1e4d893e264"
CAPTURE_FIELDS = ("size", "orientation", "camera_model", "f_number", "iso", "white_balance")


def catalog(archive, part, old_groups):
    all_files = [item for item in archive.infolist() if not item.is_dir()]
    if len({item.filename for item in all_files}) != len(all_files):
        raise ValueError(f"Duplicate ZIP entries in part {part}")
    groups, options, extensions, layouts = Counter(), defaultdict(list), Counter(), Counter()
    for item in all_files:
        path = PurePosixPath(item.filename)
        extensions[path.suffix.lower()] += 1
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        fields = path.parts
        if len(fields) != 6 or fields[0] != f"part_{part}_of_4" or not fields[1].isdigit():
            raise ValueError(f"Unknown JPEG grouping in part {part}: {path}")
        group = int(fields[1])
        groups[group] += 1
        layouts["/".join(fields[2:5])] += 1
        if fields[2:5] == ("vanilla", "pos_2", "angle_4") and group not in old_groups:
            match = re.fullmatch(r"IMG_(\d{4})\.JPG", path.name, flags=re.IGNORECASE)
            if not match:
                raise ValueError(f"Unknown capture number in part {part}: {path}")
            options[group].append((int(match[1]), item))
    pairs = {}
    for group, records in sorted(options.items()):
        sequence = sorted(records, key=lambda row: row[0])
        if len(sequence) == 5 and [number for number, _ in sequence] == list(range(sequence[0][0], sequence[0][0] + 5)):
            pairs[group] = [member for _, member in sequence[:2]]
    if not groups:
        raise ValueError(f"No JPEGs in part {part}")
    return dict(groups=dict(sorted(groups.items())), extensions=dict(sorted(extensions.items())),
                layouts=dict(sorted(layouts.items())), complete_new_groups=list(pairs)), pairs


def fetch_pair(archive, items, destination):
    photos = {}
    for light, item in zip(("warm", "cold"), items, strict=True):
        payload = archive.read(item)
        settings = exposure(payload)
        with Image.open(BytesIO(payload)) as image:
            image.load()
        (destination / f"{light}.jpg").write_bytes(payload)
        photos[light] = dict(member=item.filename, bytes=len(payload), crc32=f"{item.CRC:08x}",
                             sha256=sha256(payload).hexdigest(), settings=settings)
    unchanged = {field: photos["warm"]["settings"][field] is not None
                 and photos["warm"]["settings"][field] == photos["cold"]["settings"][field]
                 for field in CAPTURE_FIELDS}
    return dict(photos=photos, matched_capture_fields=unchanged, qualified_capture=all(unchanged.values()))


def run(part_one, output, proxy):
    previous = part_one.read_bytes()
    if sha256(previous).hexdigest() != PART_ONE_CATALOG:
        raise ValueError("Part 1 catalog differs")
    previous_groups = {int(PurePosixPath(row["name"]).parts[1]) for row in json.loads(previous)}
    output.mkdir()
    audit, selected = {}, set()
    image_bytes = range_bytes = 0
    for part, dataset, file, size, tag in RELEASES:
        url = f"https://data.mendeley.com/public-files/datasets/{dataset}/files/{file}/file_downloaded"
        reader = PublicRange(min(2 * MIB, 38 * MIB - range_bytes), proxy, url=url, archive_size=size, archive_tag=f'"{tag}"')
        with ZipFile(reader) as archive:
            record, pairs = catalog(archive, part, previous_groups)
            record.update(dataset=dataset, size=size, etag=tag, catalog_range_bytes=reader.transferred)
            eligible = sorted(pairs.keys() - selected)
            if eligible:
                group = eligible[0]
                items = pairs[group]
                image_bytes += sum(item.file_size for item in items)
                if image_bytes > 32 * MIB:
                    raise ValueError("Combined source JPEG budget exceeded")
                reader.limit = 38 * MIB - range_bytes
                folder = output / f"part-{part}"
                folder.mkdir()
                record["selected"] = dict(group=group, **fetch_pair(archive, items, folder))
                selected.add(group)
            range_bytes += reader.transferred
            record["range_bytes"] = reader.transferred
            audit[str(part)] = record
            write(output / f"part-{part}.json", record)
    report = dict(part_one_catalog_sha256=PART_ONE_CATALOG, part_one_groups=sorted(previous_groups),
                  parts=audit, source_jpeg_bytes=image_bytes, range_bytes=range_bytes)
    write(output / "report.json", report)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("part_one", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--proxy")
    run(**vars(parser.parse_args()))
