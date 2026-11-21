"""Test a geometric face-surround cue for a known local edit on real exposures."""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from probe_dast_exposure_response import digest, midtone, save_report, statistics, verified_png
from probe_dast_local_edit_cross import INPUT_SHA, STATES, WARM, apply_edit, read_ledger, score_edit, write_png
from probe_dast_paired_local_replay import EDIT_SHA, locate
from segment import Segmenter


PARENT = "b16bd8e38e07f9eb7ea70762079cb6f2780526735c6a3957e7be749fde0c677c"
CASCADE = "0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0"
ARMS = ("face", "face_minus_surround")


def detector(path):
    if digest(path) != CASCADE:
        raise ValueError("Changed OpenCV face detector data")
    storage = cv2.FileStorage(path.read_text(encoding="utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    cascade = cv2.CascadeClassifier()
    if not cascade.read(storage.getFirstTopLevelNode()) or cascade.empty():
        raise ValueError("Cannot load OpenCV face detector data")
    return cascade


def cues(rgb, cascade):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))
    if len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda box: (box[2] * box[3], -box[0], -box[1]))
    yy, xx = np.ogrid[:rgb.shape[0], :rgb.shape[1]]
    radius = np.hypot((xx - (x + w / 2)) / (w / 2), (yy - (y + h / 2)) / (h / 2))
    mid = midtone(rgb)
    inside = mid & (radius <= .65)
    surround = mid & (radius >= .95) & (radius <= 1.30)
    warm = rgb @ (WARM / np.linalg.norm(WARM))
    count_a = int(inside.sum())
    count_b = int(surround.sum())
    face = float(warm[inside].mean()) if count_a >= 128 else None
    relative = face - float(warm[surround].mean()) if face is not None and count_b >= 128 else None
    return dict(face=face, face_minus_surround=relative, pixels=[count_a, count_b],
                box=[int(n) for n in (x, y, w, h)])


def fit(known, arm):
    result = {}
    for label in ("off", "on"):
        values = [row[arm] for row, role in known if role == label and row is not None and row[arm] is not None]
        result[label] = float(np.mean(values)) if values else None
        result[f"{label}_samples"] = len(values)
    off, on = result["off"], result["on"]
    result["threshold"] = (off + on) / 2 if off is not None and on is not None and off < on else None
    return result


def is_on(measured, arm, model):
    threshold = model["threshold"]
    return bool(measured is not None and measured[arm] is not None and threshold is not None
                and measured[arm] > threshold)


def run(root, cascade_path, out):
    from PIL import Image, ImageOps

    cv2.setNumThreads(1)
    cascade = detector(cascade_path)
    source = root / "dast-exposure-prepared-v1"
    prepared = root / "dast-local-edit-prepared-v1"
    groups = read_ledger(source, "inputs.json", INPUT_SHA)["groups"]
    edits = read_ledger(prepared, "edits.json", EDIT_SHA)["groups"]
    reference = read_ledger(root / "dast-local-edit-cross-v1", "results.json", PARENT)
    persons = {str(row["group"]): row for row in groups}
    known = {}
    for person, row in persons.items():
        known[person] = {}
        for state in STATES:
            before = verified_png(source / person / f"{state}.png", row["frames"][state])
            after = verified_png(prepared / person / f"{state}-on.png", edits[person][state]["edited_sha256"])
            for role, image in (("off", before), ("on", after)):
                known[person][state, role] = cues(image, cascade)

    out.mkdir()
    reports = {}
    locator = Segmenter()
    try:
        for donor, row in persons.items():
            person = str(row["target_group"])
            target_row = persons[person]
            entry = edits[person]["nominal"]
            target = verified_png(source / person / "nominal.png", target_row["frames"]["nominal"])
            positive = apply_edit(target, locate(target, locator), .8)
            core = verified_png(source / person / "nominal-mask.png", target_row["target_mask_sha256"]) > 0
            truth = verified_png(prepared / person / "truth.png", entry["truth_sha256"])
            outside = verified_png(prepared / person / "outside.png", entry["outside_sha256"]) > 0
            train = [(feature, role) for key, profiles in known.items() if key not in (donor, person)
                     for (_, role), feature in profiles.items()]
            generated, summaries, hashes = {}, {}, {}
            oracle = score_edit(target, truth, target, positive, core, outside, edits[donor]["nominal"]["source_edit"])
            hashes["positive"] = write_png(out / f"{donor}-positive.png", positive)
            for arm in ARMS:
                model = fit(train, arm)
                frames, cases, decisions = {}, {}, {}
                for state in STATES:
                    for role in ("off", "on"):
                        key = f"{state}-{role}"
                        selected = is_on(known[donor][state, role], arm, model)
                        decisions[key] = selected
                        frames[state, role] = positive if selected else target
                        hashes[f"{arm}-{key}"] = write_png(out / f"{donor}-{arm}-{key}.png", frames[state, role])
                    cases[state] = score_edit(target, truth, frames[state, "off"], frames[state, "on"],
                                              core, outside, edits[donor][state]["source_edit"])
                exposure = {}
                for state in STATES[1:]:
                    for role in ("off", "on"):
                        a, b = frames["nominal", role], frames[state, role]
                        exposure[f"{state}-{role}"] = statistics(a, b, core & midtone(target) & midtone(a) & midtone(b))
                passes = (all(item["passes"] for item in cases.values())
                          and all(item.get("chroma_norm", np.inf) < .03 for item in exposure.values())
                          and not any(decisions[f"{state}-off"] for state in STATES))
                summaries[arm] = dict(decisions=decisions, score=cases, exposure=exposure, passed=passes, fit=model)
                generated[arm] = frames
            images = [target, truth, positive] + [generated[arm][state, "on"] for arm in ARMS for state in STATES]
            board = Image.new("RGB", (900, 390), "#292c34")
            for index, pixels in enumerate(images):
                tile = ImageOps.contain(Image.fromarray(pixels), (294, 124))
                board.paste(tile, (300 * (index % 3), 130 * (index // 3)))
            board.save(out / f"{donor}-comparison.png")
            reports[donor] = dict(target=person, cues={f"{state}-{role}": v for (state, role), v in known[donor].items()},
                                  results=summaries, positive=oracle, hashes=hashes,
                                  comparison=digest(out / f"{donor}-comparison.png"))
    finally:
        locator.close()
    protocol = Path(__file__).with_name("dast-local-face-surround-protocol.md")
    save_report(out / "results.json", dict(parent=PARENT, inputs=INPUT_SHA, edits=EDIT_SHA,
                cascade=CASCADE, opencv=cv2.__version__, protocol=digest(protocol),
                baseline_passed_groups=reference["groups_passing"], groups=reports))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("cascade_path", type=Path)
    parser.add_argument("out", type=Path)
    run(**vars(parser.parse_args()))
