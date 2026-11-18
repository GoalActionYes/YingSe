"""Audit the fixed public GIER task index without reading linked images."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

SOURCE_SHA = "56c99fbcbb7f0439000e95bab7cc9d1a625fb8da1309b9440937b77ed1da6216"
TONE = {"black&white", "brightness", "contrast", "exposure", "hue", "lightness", "saturation", "tint"}


def audit(rows):
    occurrences, local, isolated = Counter(), Counter(), Counter()
    urls, sources = [], Counter()
    fields = set()
    for row in rows:
        operations = row["operator"]
        occurrences.update(operations.keys() & TONE)
        for name, data in operations.items():
            if name not in TONE:
                continue
            fields.update(data)
            if data["local"] is not True or not data["ids"]:
                continue
            local[name] += 1
            if len(operations) == 1:
                isolated[name] += 1
                urls.append(row["input_url"].strip())
                sources[row["dataset"]] += 1
    return {"rows": len(rows), "operator_fields": sorted(fields),
            "operations": {name: {"all": occurrences[name], "grounded": local[name], "isolated": isolated[name]}
                           for name in sorted(TONE)},
            "isolated_rows": len(urls), "isolated_domains": dict(sorted(sources.items())),
            "isolated_distinct_nonempty_input_urls": len(set(urls) - {""}),
            "isolated_empty_input_urls": urls.count("")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    contents = args.index.read_bytes()
    identity = sha256(contents).hexdigest()
    if identity != SOURCE_SHA:
        raise ValueError("Unexpected published GIER index")
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump({"index_sha256": identity, **audit(json.loads(contents))}, output, indent=2)
        output.write("\n")
