"""Audit the official ARRI workflow guide and public share metadata."""

import argparse
from collections import defaultdict
from hashlib import sha256
import html
import json
from pathlib import Path
import re

from pypdf import PdfReader

SHOT = re.compile(r"\b(\d{2})_([A-Z]\d{3}C\d{3})_(\d{6})_([A-Z0-9]{4})\b")
PAGE_NAMES = ("proxies.html", "tiff.html", "clip06.html", "clip08.html")
PROXY_GUIDE_LINK = "https://f.io/Jft16WLA"


def matches_in_guide(path):
    clips = {}
    for page_index, page in enumerate(PdfReader(path).pages, 1):
        text = page.extract_text()
        matches = list(SHOT.finditer(text))
        if not matches:
            continue
        links = {}
        for item in page.get("/Annots", []):
            entry = item.get_object()
            action = entry.get("/A")
            uri = str(action.get("/URI", "")) if action else ""
            left, bottom, right, top = map(float, entry["/Rect"])
            if uri.startswith("https://f.io/") and uri != PROXY_GUIDE_LINK and right > left and top > bottom:
                links[uri] = bottom
        if len(links) != len(matches):
            raise ValueError(f"Ambiguous shot link match on page {page_index}")
        order = sorted(links, key=links.__getitem__, reverse=True)
        for index, (match, link) in enumerate(zip(matches, order, strict=True)):
            number, name, date, camera = match.groups()
            stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            description = " ".join(text[match.end():stop].lower().split())
            tags = [tag for tag in ("skin tone", "grey buildings", "bright windows", "dark interior",
                                    "camp fire", "water reflections", "salt flats") if tag in description]
            if number in clips:
                raise ValueError(f"Duplicate guide ordinal {number}")
            clips[number] = {"original": f"{name}_{date}_{camera}", "date_code": date, "page": page_index,
                             "tags": tags, "guide_link_sha256": sha256(link.encode()).hexdigest()}
    if list(clips) != [f"{n:02d}" for n in range(1, 18)]:
        raise ValueError("Unexpected published shot numbering")
    return clips


def webpage(path):
    content = path.read_bytes()
    rendered = html.unescape(content.decode("utf8")).replace(r'\"', '"')
    title, = re.findall(r"<title>([^<]+)</title>", rendered)
    flag, = re.findall(r'"isDownloadingEnabled"\s*:\s*(true|false)', rendered)
    ids = sorted({match.group(0) for match in SHOT.finditer(rendered)})
    return {"sha256": sha256(content).hexdigest(), "bytes": len(content), "title": title,
            "download_ui_enabled": flag == "true", "visible_shot_identifiers": ids}


def audit(folder):
    guide = folder / "guideline.pdf"
    clips = matches_in_guide(guide)
    shots, dates = defaultdict(list), defaultdict(list)
    for number, row in clips.items():
        shots[row["original"]].append(number)
        dates[row["date_code"]].append(number)
    pages = {name: webpage(folder / name) for name in PAGE_NAMES}
    for n in ("06", "08"):
        picture = pages[f"clip{n}.html"]
        if picture["visible_shot_identifiers"] != [f"{n}_{clips[n]['original']}"]:
            raise ValueError("Individual share does not match guide")
    source_bytes = guide.stat().st_size + sum(data["bytes"] for data in pages.values())
    if source_bytes > 3 * 1024**2:
        raise ValueError("Response budget exceeded")
    return {"guide_sha256": sha256(guide.read_bytes()).hexdigest(), "guide_bytes": guide.stat().st_size,
            "source_document_bytes": source_bytes, "clip_count": len(clips), "unique_original_count": len(shots),
            "duplicate_original_ordinals": [v for v in shots.values() if len(v) > 1],
            "clips": clips, "groups_by_date_code": dict(sorted(dates.items())), "share_pages": pages}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    result = audit(args.source)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write("\n")
