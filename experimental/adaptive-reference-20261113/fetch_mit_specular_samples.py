"""Fetch the fixed MIT Intrinsic Images gallery subset for local research."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from shutil import disk_usage
from urllib.request import Request, urlopen

GALLERY = "https://www.cs.toronto.edu/~rgrosse/intrinsic/images"
SCENES = ("frog1", "frog2", "cup1", "paper1")
ROLES = ("original", "diffuse", "specular", "mask")
MAX_BYTES = 8 * 1024**2


def fetch(output):
    records = {}
    for scene in SCENES:
        row = {}
        for role in ROLES:
            url = f"{GALLERY}/{scene}/{role}.png"
            with urlopen(Request(url, method="HEAD"), timeout=15) as response:
                size = int(response.headers["Content-Length"])
                if (response.url != url or not 0 < size <= 2 * 1024**2
                        or response.headers.get_content_type() != "image/png"):
                    raise ValueError(f"Unexpected gallery metadata: {scene}/{role}")
                row[role] = {"url": url, "bytes": size, "file": f"{scene}-{role}.png"}
        records[scene] = row
    total = sum(item["bytes"] for row in records.values() for item in row.values())
    if total > MAX_BYTES or disk_usage(output.parent).free < 2 * total:
        raise ValueError("Gallery exceeds local download or disk budget")
    output.mkdir()
    for row in records.values():
        for item in row.values():
            with urlopen(item["url"], timeout=20) as response:
                data = response.read(item["bytes"] + 1)
                if (response.url != item["url"] or len(data) != item["bytes"]
                        or not data.startswith(b"\x89PNG\r\n\x1a\n")):
                    raise ValueError(f"Unexpected gallery download: {item['file']}")
                item["sha256"] = sha256(data).hexdigest()
                (output / item["file"]).write_bytes(data)
    manifest = {"bytes": total, "cases": records}
    (output / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"Stored {len(records)} published scenes and {total} PNG bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    fetch(parser.parse_args().output)
