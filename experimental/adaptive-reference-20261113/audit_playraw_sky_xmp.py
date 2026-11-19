"""Read bounded public darktable sidecars without collecting any image data."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
from xml.etree import ElementTree
import zlib

DT = "{http://darktable.sf.net/}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
DERIVED = "{http://ns.adobe.com/xap/1.0/mm/}DerivedFrom"
NAMES = ("sky-A", "sky-B1", "sky-B2", "sky-B3")


def payload(encoded):
    if not encoded.startswith("gz"):
        return bytes.fromhex(encoded)
    decoder = zlib.decompressobj()
    value = decoder.decompress(base64.b64decode(encoded[4:], validate=True), 4096)
    if not decoder.eof or decoder.unused_data:
        raise ValueError("Unexpected compressed module size")
    return value


def audit(path):
    data = path.read_bytes()
    if len(data) > 65536:
        raise ValueError(f"Oversize sidecar {path.name}")
    node = ElementTree.fromstring(data).find(f".//{RDF}Description")
    end = int(node.attrib[DT + "history_end"])
    rows, last = [], {}
    for step in node.findall(f"./{DT}history/{RDF}Seq/{RDF}li"):
        a = {k.removeprefix(DT): v for k, v in step.attrib.items()}
        index = int(a["num"])
        if a["operation"] != "exposure" or index >= end:
            continue
        instance = int(a["multi_priority"])
        version, blend_version = int(a["modversion"]), int(a["blendop_version"])
        params, blend = payload(a["params"]), payload(a["blendop_params"])
        if version not in (6, 7) or len(params) != 4 * version or blend_version not in (13, 14) or len(blend) != 420:
            raise ValueError(f"Unverified exposure schema in {path.name} step {index}")
        mode, black, ev = struct.unpack_from("<Iff", params)
        mask, = struct.unpack_from("<I", blend)
        enabled = a["enabled"] == "1"
        row = {"step": index, "instance": instance, "enabled": enabled, "version": version,
               "mode": mode, "black": black, "ev": ev, "blend_version": blend_version, "mask": mask,
               "masked_darken": enabled and mode == 0 and ev < 0 and (mask & 1) != 0 and (mask & 6) != 0,
               "params_sha256": hashlib.sha256(params).hexdigest(), "blend_sha256": hashlib.sha256(blend).hexdigest()}
        rows.append(row)
        last[instance] = row
    return {"file": path.stem, "raw": node.attrib[DERIVED], "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "history_end": end, "exposure_history": rows,
            "active_masked_steps": [row["step"] for row in last.values() if row["masked_darken"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    print(json.dumps([audit(args.input / f"{name}.xmp") for name in NAMES], indent=2, allow_nan=False))
