"""Reserve fixed BG-REAL source groups using release and consumption metadata only."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import parse_qs, urlsplit

from audit_bg_real_controls import CC_BY, REVISION, verified_csv

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/adaptive-reference-20261113"
INTAKE_SHA = "79580687384ed0791faa8e1ef9b4926e29c3e8cf8a1b470f11fcd84ee3663a8c"
FIT_SHA = "58a4cd10c1d7e3d43aa5cfe5da311459aea5664022dcf012282e551645ee1703"
DEV_SHA = "83be16a13691b50cd3449375df656b0cedabdbf3b31e63639b43d6e42f3c68d9"
SALT = "yingse-bg-reserve-v1|"
LABELS = ("Woman", "Man")
ROLES = ("C1", "F1", "mask")
FURTHER_LEDGER = {
    "independent-coco": "bbd61c0b5efe4bdfa5c4d1b36dcabe8af474ea80c24bc36b44bac34e5607c7e2",
    "supervised-style": "db488ce279e888f3b1a6c27b359c47b8640a2f2ebb55c66fec620b673468fc4d",
}


def pinned(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Changed evidence: {path.name}")
    return json.loads(data)


def flickr(url):
    uri = urlsplit(url or "")
    host = uri.hostname or ""
    path = PurePosixPath(uri.path)
    if host.endswith(".staticflickr.com") or host == "staticflickr.com":
        match = re.match(r"^(\d+)_[0-9a-f]+", path.name, re.I)
        return match[1] if match else None
    if host in ("flickr.com", "www.flickr.com"):
        parts = path.parts
        if len(parts) >= 4 and parts[1] == "photos" and parts[3].isdigit():
            return parts[3]
        keys = parse_qs(uri.query).get("id", [])
        if path.name == "photo.gne" and len(keys) == 1 and keys[0].isdigit():
            return keys[0]
    return None


def sources(rows):
    identities = {}
    conflicting = set()
    for row in rows:
        key = row["bgreal_id"]
        a, b = (flickr(row[field]) for field in ("OriginalURL", "OriginalLandingURL"))
        if not row["ImageID"] or row["License"] != CC_BY or (a and b and a != b) or not (a or b):
            conflicting.add(key)
            continue
        identities[key] = (("open_images", row["ImageID"]), ("flickr", a or b))
    return identities, conflicting


class Components:
    def __init__(self):
        self.parents = {}

    def root(self, value):
        self.parents.setdefault(value, value)
        head = value
        while self.parents[head] != head:
            head = self.parents[head]
        while self.parents[value] != value:
            parent = self.parents[value]
            self.parents[value] = head
            value = parent
        return head

    def union(self, first, second):
        a, b = sorted((self.root(first), self.root(second)))
        self.parents[b] = a


def match_families(catalog_dir, eligible):
    manifest, _ = verified_csv(catalog_dir, "manifest_v2.csv")
    attribution, _ = verified_csv(catalog_dir, "attribution.csv")
    recipients, _ = verified_csv(catalog_dir, "recipients.csv")
    identities, conflicting = sources(attribution)
    graph = Components()
    subjects = {row["source_image_id"] for row in recipients
                if row["subject_group"] == "person" and row["split"] == "test_id"}
    for key, nodes in identities.items():
        for node in nodes:
            graph.union(("release", key), node)

    observed = set()
    for row in manifest:
        subject, donor = row["source_image_id"], row["donor_id"]
        if subject not in subjects or row["split"] != "test_id" or row["robustness"] != "none":
            continue
        if row["family"] not in ("C1", "F1"):
            continue
        if re.fullmatch(r"[a-f0-9]{64}", row["sha256"]):
            graph.union(("release", subject), ("asset", row["sha256"]))
        else:
            conflicting.add(subject)
        if row["family"] == "F1" and donor:
            graph.union(("release", subject), ("release", donor))
            observed.add((subject, donor))
    if any((key, item["donor"]) not in observed for key, item in eligible.items()):
        raise ValueError("A pinned pair is absent from its original release manifest")
    unresolved = {key for key, row in eligible.items()
                  if {key, row["donor"]} & conflicting or not {key, row["donor"]} <= identities.keys()}
    excluded_roots = {graph.root(("release", key)) for key in conflicting}
    return graph, unresolved, excluded_roots, len(observed)


def used_images(intake):
    train = BASE / "bg-real-color-directions-inputs"
    develop = BASE / "bg-real-pixel-inputs"
    training = pinned(train / "intake.json", FIT_SHA)
    frozen = pinned(ROOT / "docs/research/2026-11-18-bg-real-pixel-pilot-evidence.json", DEV_SHA)["cases"]
    development = {key: row["input"] for key, row in frozen.items()}
    if len(training) != 12 or len(development) != 4 or training.keys() & development.keys():
        raise ValueError("Unexpected consumed group ledger")
    verified = 0
    for folder, ledger in ((train, training), (develop, development)):
        expected = set()
        for key, row in ledger.items():
            for role in ROLES:
                item = row[role]
                path = folder / item["name"]
                if path.parent != folder or sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError("Changed consumed media")
                if role != "mask" and item["sha256"] != intake[key]["arms"][role]["sha256"]:
                    raise ValueError("The consumed reference differs from its release row")
                expected.add(path.name)
                verified += 1
        actual = {p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() != ".json"}
        if actual != expected:
            raise ValueError("An extra or missing source asset prevents safe reservation")
    return training.keys() | development.keys(), verified


def freeze(records, graph, used, isolated, unresolved):
    roots = {key: graph.root(("release", key)) for key in records}
    seen = set(isolated) | {roots[key] for key in used}
    eligible = [key for key in records if key not in unresolved and roots[key] not in seen]
    label_counts = Counter(records[key]["subject"] for key in eligible)
    selected = []
    for label in LABELS:
        remaining = sorted((key for key in eligible if records[key]["subject"] == label),
                           key=lambda key: sha256((SALT + key).encode()).digest())
        count = 0
        for key in remaining:
            if roots[key] in seen:
                continue
            selected.append(key)
            seen.add(roots[key])
            count += 1
            if count == 6:
                break
    counts = dict(available_after_exclusions=dict(sorted(label_counts.items())),
                  selected_by_label=dict(sorted(Counter(records[key]["subject"] for key in selected).items())),
                  usable_sources=len(eligible), source_components=len(set(roots.values())),
                  consumed_pair_components=len({roots[key] for key in used}),
                  total_rows_excluded=len(records) - len(eligible))
    if len(selected) != 12:
        return counts, None
    assigned = selected[1:] + selected[:1]
    return counts, [
        {"reference": ref, "target": target, "reference_items": records[ref], "target_items": records[target]}
        for ref, target in zip(selected, assigned, strict=True)
    ]


def save(path, data):
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def cross_reference(graph, queue):
    seen = set()
    parts = {}
    for name, fingerprint in FURTHER_LEDGER.items():
        images = pinned(BASE / name / "manifest.json", fingerprint)["images"]
        asset_numbers = [flickr(row["flickr_url"]) for row in images]
        known = set(x for x in asset_numbers if x is not None)
        parts[name] = {"manifest_sha256": fingerprint, "records": len(images),
                       "resolved_flickr_ids": len(known), "unresolved": asset_numbers.count(None)}
        seen.update(known)
    shared = {graph.root(("flickr", key)) for key in seen}
    quarantine = [
        i for i, row in enumerate(queue or [], 1)
        if {graph.root(("release", row[field])) for field in ("reference", "target")} & shared
    ]
    return {"ledgers": parts, "unique_flickr_ids": len(seen), "quarantined_pair_ordinals": quarantine}


def run(output):
    record = pinned(ROOT / "docs/research/2026-11-18-bg-real-control-intake-evidence.json", INTAKE_SHA)
    pairs = record["pairs"]
    if record["revision"] != REVISION or len(pairs) != 87:
        raise ValueError("Unexpected release cohort")
    used, verified = used_images(pairs)
    graph, unresolved, conflicts, matched = match_families(BASE / "bg-real-control-intake", pairs)
    counts, queue = freeze(pairs, graph, used, conflicts, unresolved)
    output.mkdir()
    private = output / "reserved-manifest.json"
    if queue is not None:
        save(private, {"revision": REVISION, "pairs": queue})
    save(output / "results.json", {
        "release_revision": REVISION, "intake_sha256": INTAKE_SHA,
        "fitting_manifest_sha256": FIT_SHA, "development_manifest_sha256": DEV_SHA,
        "used_reference_groups": len(used), "verified_consumed_files": verified,
        "release_graph_edges": matched, "unresolved_rows": len(unresolved), **counts,
        "reserved_pairs": len(queue) if queue is not None else 0,
        "manifest_sha256": sha256(private.read_bytes()).hexdigest() if queue is not None else None,
        "downloaded_reserved_images": 0, "used_for_scoring": False,
        "cross_reference": cross_reference(graph, queue),
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    run(cli.parse_args().output)
