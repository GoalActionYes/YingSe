"""Retrieve one preselected official ARRI LogC3 creative transform by ZIP range."""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import zipfile

import requests

URL = ("https://arri.canto.de/direct/other/27l7vgn15p6kj0or4labq57f5g/"
       "nPgi_Ik_Qd6Z-rpv0-hJ4jopbSA/original?content-type=application%2Fzip"
       "&name=ARRI+Look+Library+LogC3+%28log-to-log%29+3D+LUTs_data.zip")
SIZE = 31286603
ETAG = '"0d5eeec1f6c9858da9a1058b24a34250"'
MEMBER = "ARRI Look Library LogC3 (log-to-log)/3110 Film A.cube"


class RangeReader(io.RawIOBase):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.position = 0
        self.requests = []
        self.total = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, SIZE)[whence] + offset
        if not 0 <= self.position <= SIZE:
            raise ValueError("ZIP position outside archive")
        return self.position

    def read(self, count=-1):
        count = SIZE - self.position if count < 0 else min(count, SIZE - self.position)
        if not count:
            return b""
        if self.total + count > 4 * 1024**2 or len(self.requests) >= 12:
            raise ValueError("ZIP range budget exceeded")
        start, end = self.position, self.position + count - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        with self.session.get(URL, headers=headers, stream=True, timeout=(8, 18)) as response:
            if (response.status_code != 206 or response.headers.get("ETag") != ETAG
                    or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}"):
                raise ValueError("Unverified ARRI archive range")
            data = response.raw.read(count + 1)
        if len(data) != count:
            raise ValueError("Truncated ARRI archive range")
        self.position += count
        self.total += count
        self.requests.append({"start": start, "end": end, "sha256": sha256(data).hexdigest()})
        return data


def fetch(output, proxy):
    with requests.Session() as session:
        if proxy:
            session.proxies.update({"http": proxy, "https": proxy})
        remote = RangeReader(session)
        with zipfile.ZipFile(remote) as archive:
            film = sorted(name for name in archive.namelist()
                          if re.fullmatch(r"\d+ Film.*\.cube", Path(name).name, re.IGNORECASE))
            if not film or film[0] != MEMBER or archive.namelist().count(MEMBER) != 1:
                raise ValueError("Fixed official look unavailable")
            info = archive.getinfo(MEMBER)
            if info.file_size > 3 * 1024**2 or info.flag_bits & 1:
                raise ValueError("Invalid official look member")
            index_bytes = remote.total
            data = archive.read(info)
    output.mkdir()
    (output / "film-a.cube").write_bytes(data)
    receipt = {
        "archive_size": SIZE, "archive_etag": ETAG, "film_members": len(film), "index_bytes": index_bytes,
        "member": MEMBER, "member_bytes": len(data), "member_crc32": f"{info.CRC:08x}",
        "member_sha256": sha256(data).hexdigest(), "ranges_total_bytes": remote.total, "ranges": remote.requests,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: receipt[key] for key in ("member", "member_sha256", "ranges_total_bytes")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--proxy")
    args = parser.parse_args()
    fetch(args.output, args.proxy)
