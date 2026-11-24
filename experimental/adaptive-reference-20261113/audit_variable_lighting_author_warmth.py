"""Compare the released P1020943 XMP histories without opening JPEG pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree

from audit_playraw_sky_xmp import DT, RDF, payload

STANDARD_XMP = b"http://ns.adobe.com/xap/1.0/\0"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_range(directory, filename, headers, start, end, total):
    receipt = (directory / headers).read_text(encoding="ascii")
    data = (directory / filename).read_bytes()
    if not re.search(r"^HTTP/\S+ 206\b", receipt, re.M):
        raise ValueError(f"Expected HTTP 206 for {filename}")
    if not re.search(rf"^content-range: bytes {start}-{end}/{total}\s*$", receipt, re.I | re.M):
        raise ValueError(f"Wrong byte range for {filename}")
    if len(data) != end - start + 1:
        raise ValueError(f"Wrong size for {filename}")
    return data


def standard_xmp(data):
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("Expected JPEG start marker")
    offset = 2
    while offset + 4 <= len(data):
        tag = data[offset:offset + 2]
        length = int.from_bytes(data[offset + 2:offset + 4], "big")
        end = offset + 2 + length
        if not 0xE0 <= tag[1] <= 0xEF or tag[0] != 0xFF or length < 2 or end > len(data):
            break
        body = data[offset + 4:end]
        if tag == b"\xff\xe1" and body.startswith(STANDARD_XMP):
            return body[len(STANDARD_XMP):], offset, end
        offset = end
    raise ValueError("Complete standard XMP was not found in the bounded JPEG prefix")


def latest_history(data):
    root = ElementTree.fromstring(data.rstrip(b"\0"))
    if any("HasExtendedXMP" in key for node in root.iter() for key in (node.tag, *node.attrib)):
        raise ValueError("XMP declares an extension outside this audit")
    description, = (node for node in root.iter(RDF + "Description") if DT + "history_end" in node.attrib)
    end = int(description.attrib[DT + "history_end"])
    masks = [dict(sorted(node.attrib.items()))
             for node in description.findall(f"./{DT}masks_history/{RDF}Seq/{RDF}li")]
    effective = {}
    for node in description.findall(f"./{DT}history/{RDF}Seq/{RDF}li"):
        a = {key.removeprefix(DT): value for key, value in node.attrib.items()}
        step = int(a["num"])
        if step >= end:
            continue
        blend = payload(a["blendop_params"])
        if a["blendop_version"] != "14" or len(blend) != 420:
            raise ValueError(f"Unverified blend format at step {step}")
        key = f'{a["operation"]}/{a["multi_priority"]}'
        if key in effective and effective[key]["step"] >= step:
            raise ValueError(f"Out of order history at step {step}")
        effective[key] = dict(step=step, enabled=a["enabled"] == "1", version=int(a["modversion"]),
                              parameter_sha256=sha(payload(a["params"])), blend_sha256=sha(blend),
                              mask_mode=int.from_bytes(blend[:4], "little"))
    colors = {key: value for key, value in sorted(effective.items())
              if key.split("/")[0] in {"agx", "channelmixerrgb", "colorbalancergb", "toneequal"}}
    summary = dict(xmp_bytes=len(data), xmp_sha256=sha(data), history_end=end,
                   instances=len(effective), mask_records=len(masks), color_and_tone=colors)
    return summary, effective, masks


def compare(first, second):
    _, a, am = first
    _, b, bm = second
    fields = {}
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b:
            fields[key] = ["added" if key in b else "removed"]
            continue
        changes = sorted(field for field in a[key] if field != "step" and a[key][field] != b[key][field])
        if changes:
            fields[key] = changes
    return dict(changed_instances=fields, mask_history_equal=am == bm)


def audit(directory):
    upper = read_range(directory, "upper.head", "upper.headers", 0, 65535, 912602)
    tail = read_range(directory, "upper.tail", "upper-tail.headers", 65536, 73863, 912602)
    lower = read_range(directory, "lower.head", "lower.headers", 0, 65535, 877347)
    frames = {}
    for name, prefix in (("upper", upper + tail), ("lower", lower)):
        xmp, start, end = standard_xmp(prefix)
        summary, effective, masks = latest_history(xmp)
        summary.update(prefix_bytes=len(prefix), prefix_sha256=sha(prefix), xmp_app1_start=start,
                       xmp_app1_end_exclusive=end)
        frames[name] = summary, effective, masks
    sidecars = {name: (directory / f"{name}.xmp").read_bytes() for name in ("todd", "kofa")}
    if sum(map(len, sidecars.values())) > 80 * 1024:
        raise ValueError("Sidecars exceed the registered byte limit")
    frames.update({name: latest_history(data) for name, data in sidecars.items()})
    return dict(received_bytes=len(upper) + len(tail) + len(lower) + sum(map(len, sidecars.values())),
                upper_ranges_sha256=[sha(upper), sha(tail)],
                histories={name: frame[0] for name, frame in frames.items()},
                lower_to_upper=compare(frames["lower"], frames["upper"]),
                upper_to_todd_sidecar=compare(frames["upper"], frames["todd"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.input), indent=2, allow_nan=False) + "\n", encoding="utf-8")
