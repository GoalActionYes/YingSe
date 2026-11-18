"""Fetch four ordinal-matched HdM HDR stills for local academic comparison."""
import argparse
from ftplib import FTP
from hashlib import sha256
import html
import json
from pathlib import Path, PurePosixPath
import re
from shutil import disk_usage
from urllib.request import ProxyHandler, build_opener

PROJECT = "https://hdm-stuttgart.de/vmlab/hdm-hdr-2014"
SCENES = ("showgirl_01", "poker_fullshot")
FOLDERS = {
    "original": "HdM-HDR-2014_Original-HDR-Camera-Footage",
    "graded": "HdM-HDR-2014_Color-Graded-for-HDR",
}


def remote_credentials(proxy):
    opener = build_opener(ProxyHandler({"https": proxy} if proxy else {}))
    with opener.open(PROJECT, timeout=18) as response:
        raw = response.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ValueError("Author page exceeds metadata budget")
    page = raw.decode("utf-8")
    fields = [html.unescape(re.search(rf"{key}:\s*([^<\r\n]+)", page)[1]).strip()
              for key in ("Server", "User", "Password")]
    if fields[0] != "hdr-2014.hdm-stuttgart.de":
        raise ValueError("Unexpected FTP host published by author")
    return fields


def entries(ftp, directory, room):
    ftp.cwd(directory)
    selected = {}
    with ftp.transfercmd("NLST") as socket:
        with socket.makefile("rb") as stream:
            while raw := stream.readline(8192):
                room[0] -= len(raw)
                if room[0] < 0:
                    raise ValueError("FTP directory metadata budget exceeded")
                filename = PurePosixPath(raw.decode("utf-8").rstrip()).name
                match = re.search(r"(\d+)\.(?:dpx|tiff?|exr)$", filename, re.I)
                if match:
                    number = int(match[1])
                    if number in selected:
                        raise ValueError(f"Multiple files share a frame in {directory}")
                    selected[number] = filename
    ftp.voidresp()
    return selected


def run(output, proxy, list_only, showgirl_light=False):
    host, username, password = remote_credentials(proxy)
    output.mkdir()
    with FTP(host, timeout=16) as ftp:
        ftp.login(username, password)
        ftp.voidcmd("TYPE I")
        license_buffer = bytearray()

        def retain_license(data):
            if len(license_buffer) + len(data) > 8192:
                raise ValueError("Unexpected release license size")
            license_buffer.extend(data)

        ftp.retrbinary("RETR /License.txt", retain_license)
        if b"Academic use for all three datasets is free" not in license_buffer:
            raise ValueError("Release license changed")
        (output/"license.txt").write_bytes(license_buffer)

        rows = {}
        total = 0
        metadata_budget = [256*1024]
        samples = {"showgirl_02": (68, 272)} if showgirl_light else dict.fromkeys(SCENES, (None,))
        for scene, ordinals in samples.items():
            paths = {role: f"/{directory}/{scene}" for role, directory in FOLDERS.items()}
            by_role = {role: entries(ftp, path, metadata_budget) for role, path in paths.items()}
            numbers = {role: sorted(items) for role, items in by_role.items()}
            if len({len(seq) for seq in numbers.values()}) != 1:
                raise ValueError(f"Frame counts differ: {scene}")
            for seq in numbers.values():
                if not seq or seq != list(range(seq[0], seq[-1] + 1)):
                    raise ValueError(f"Frame numbers are not contiguous: {scene}")
            if showgirl_light and len(numbers["original"]) != 341:
                raise ValueError("Unexpected Showgirl 2 frame count")
            for sample in ordinals:
                ordinal = len(numbers["original"]) // 2 if sample is None else sample
                label = scene if sample is None else f"{scene}-{ordinal:03d}"
                files = {}
                for role, path in paths.items():
                    frame = numbers[role][ordinal]
                    remote = path + "/" + by_role[role][frame]
                    size = ftp.size(remote)
                    if size is None or size < 1:
                        raise ValueError("File size unavailable")
                    files[role] = {"official_path": remote, "bytes": size, "frame": frame,
                                   "number_range": [numbers[role][0], numbers[role][-1]]}
                    total += size
                rows[label] = {"counts": {role: len(items) for role, items in by_role.items()},
                               "selected_ordinal_zero_based": ordinal, "files": files}

        if total > 64*1024**2:
            raise ValueError(f"Selected source would exceed the download budget: {total}")
        if not list_only and disk_usage(output).free < 2 * total:
            raise ValueError("Insufficient local room for source and analysis")
        result = {"project": PROJECT, "license_sha256": sha256(license_buffer).hexdigest(),
                  "cases": rows, "selected_bytes": total, "fetched": not list_only}
        if not list_only:
            for scene, row in rows.items():
                for role, record in row["files"].items():
                    source = record["official_path"]
                    target = output/(scene + "-" + role + PurePosixPath(source).suffix.lower())
                    checksum = sha256()
                    with target.open("xb") as handle:
                        def write(data):
                            if handle.tell() + len(data) > record["bytes"]:
                                raise ValueError("Source exceeded published size")
                            handle.write(data)
                            checksum.update(data)
                        ftp.retrbinary("RETR " + source, write)
                    if target.stat().st_size != record["bytes"]:
                        raise ValueError("Source read was truncated")
                    record.update(sha256=checksum.hexdigest(), file=target.name)
    # Keep the frozen Showgirl annotation's Windows manifest bytes reproducible.
    newline = "\r\n" if showgirl_light else None
    (output/"manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline=newline)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--proxy")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--showgirl-light", action="store_true", help="Fetch fixed Showgirl 2 ordinals")
    args = parser.parse_args()
    run(args.output, args.proxy, args.list_only, args.showgirl_light)
