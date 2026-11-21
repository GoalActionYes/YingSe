"""Read the six fixed side-lit Gros portraits from the original archive."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import re
from zipfile import ZipFile

from audit_gros_portrait_light import ARCHIVE_SIZE, ARCHIVE_TAG, MIB, PublicRange, exposure, write


ANGLES = (2, 6)


def run(out, proxy):
    reader = PublicRange(limit=34 * MIB, proxy=proxy)
    out.mkdir()
    with ZipFile(reader) as archive:
        entries = [item for item in archive.infolist() if item.filename.lower().endswith(".jpg")]
        assert len(entries) == 1576
        photos = {}
        for subject in (1, 2, 3):
            for angle in ANGLES:
                prefix = f"part_1_of_4/{subject}/vanilla/pos_2/angle_{angle}/"
                members = sorted((entry for entry in entries if entry.filename.startswith(prefix)),
                                 key=lambda entry: entry.filename)
                numbers = [int(re.fullmatch(r"IMG_(\d{4})\.JPG", Path(member.filename).name)[1])
                           for member in members]
                if len(numbers) != 5 or numbers != list(range(numbers[0], numbers[0] + 5)):
                    raise ValueError(f"Unexpected warm-first sequence: {prefix}")
                member = members[0]
                name = f"{subject:02d}-angle-{angle}"
                raw = archive.read(member)
                metadata = exposure(raw)
                if metadata["size"] != (5184, 3456) or metadata["iso"] != 800 or metadata["f_number"] != 8:
                    raise ValueError(f"Unexpected camera settings: {member.filename}")
                photos[name] = {"member": member.filename, "sha256": sha256(raw).hexdigest(),
                                "bytes": len(raw), "crc32": f"{member.CRC:08x}", "exposure": metadata}
                (out / f"{name}.jpg").write_bytes(raw)
                print(name, len(raw), flush=True)
    write(out / "intake.json", {"photos": photos, "transfer_bytes": reader.transferred,
                                "archive_bytes": ARCHIVE_SIZE, "archive_etag": ARCHIVE_TAG})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--proxy")
    run(**vars(parser.parse_args()))
