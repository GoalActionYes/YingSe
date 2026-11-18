"""Select a small reproducible COCO TV/person intake before visual adjudication."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw

from independent_coco import ANNOTATION_SHA, BUCKET, ROOT, digest, fetch


def overlap(person, screen):
    a, b, w, h = person
    x, y, width, height = screen
    return (max(0, min(a+w, x+width)-max(a, x)) * max(0, min(b+h, y+height)-max(b, y))) / (width*height)


def run(output):
    archive = ROOT / ".downloads/coco/annotations_trainval2017.zip"
    if digest(archive) != ANNOTATION_SHA:
        raise ValueError("COCO annotation file changed")
    with ZipFile(archive) as z, z.open("annotations/instances_val2017.json") as stream:
        dataset = json.load(stream)
    labels = {c["name"]: c["id"] for c in dataset["categories"]}
    categories = {labels["person"], labels["tv"]}
    boxes = defaultdict(lambda: defaultdict(list))
    for record in dataset["annotations"]:
        if record["category_id"] in categories and not record["iscrowd"]:
            boxes[record["image_id"]][record["category_id"]].append(record["bbox"])
    licenses = {c["id"]: c for c in dataset["licenses"]}
    eligible = []
    for image in dataset["images"]:
        ident, dimensions = image["id"], (image["width"], image["height"])
        if image["license"] not in {4, 5, 7, 8} or min(dimensions) < 240:
            continue
        records = boxes[ident]
        minimum = .03 * dimensions[0] * dimensions[1]
        people, tvs = ([b for b in records[labels[category]] if b[2] * b[3] >= minimum]
                       for category in ("person", "tv"))
        if people and tvs:
            score = max(overlap(p, t) for p in people for t in tvs)
            eligible.append((score, ident, image))
    overlaps = sorted((v for v in eligible if v[0] > 0), key=lambda v: (-v[0], v[1]))[:8]
    controls = sorted((v for v in eligible if v[0] == 0), key=lambda v: v[1])[:2]
    output.mkdir(exist_ok=True)
    sheet = Image.new("RGB", (1100, 474), "#20242a")
    draw = ImageDraw.Draw(sheet)
    rows = []
    for i, (score, ident, image) in enumerate(overlaps + controls):
        name = image["file_name"]
        if Path(name).name != name:
            raise ValueError("Invalid filename")
        path = output / name
        url = f"{BUCKET}/val2017/{name}"
        fetch(url, path, maximum=3_000_000)
        with Image.open(path) as original:
            if original.size != (image["width"], image["height"]):
                raise ValueError("COCO metadata does not match the local image")
            thumb = original.convert("RGB")
            thumb.thumbnail((210, 202), Image.Resampling.LANCZOS)
        x, y = 5 + i % 5 * 220, 5 + i // 5 * 235
        draw.text((x, y), f"{i+1}. {ident}  box overlap {score:.0%}", fill="white")
        sheet.paste(thumb, (x, y+23))
        rows.append({"id": ident, "overlap": score, "source": url, "flickr": image["flickr_url"],
                     "license": licenses[image["license"]], "sha256": digest(path), "bytes": path.stat().st_size})
    sheet.save(output / "review.jpg", quality=92)
    report = {"eligible": len(eligible), "eligible_overlap": sum(v[0] > 0 for v in eligible), "photos": rows,
              "annotation_sha256": ANNOTATION_SHA, "source_sha256": digest(Path(__file__)),
              "protocol_sha256": digest(Path(__file__).with_name("screen-occlusion-intake-protocol.md"))}
    (output / "index.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
