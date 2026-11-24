"""Evaluate the frozen local warmth edit on actual MLS illumination settings."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path

import numpy as np
from PIL import Image

from audit_mls_material_identity import checked, table
from fetch_mls_warmth_sequences import ANCHOR, LIGHTS, SCENES, entries, validated
from probe_playraw_flash_room import LUMA, PP3, RT_CLI
from probe_shared_local_warmth import PROTOCOL as WARMTH_PROTOCOL, prepare, warm_profile
from probe_target_mirror_chroma import digest, json_to
from render_rawtherapee_sky_pair import check
from triplet_color import to_srgb


PROTOCOL = "9aba1aa485a613a3ff22ec588eed6ae431fd28e8ecc294e734b7a63dd875da98"
MASK_SIZE = (578, 475)
PHOTO_SIZE = (600, 493)
PATCH_SIZE = (28, 24)


def white_mask(image, row):
    with Image.open(BytesIO(image)) as mask:
        if mask.mode != "RGB" or mask.size != MASK_SIZE:
            raise ValueError("Unexpected official mask geometry")
        rgb = np.array(mask.resize(PHOTO_SIZE, Image.Resampling.NEAREST))
    color = np.array(row["mask_colour_rgb"].split(), dtype=np.uint8)
    return (rgb == color).all(axis=-1)


def select_rectangles(mask, template):
    width, height = PATCH_SIZE
    summed = np.pad(mask.astype(np.int32).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    region_size = summed[height:, width:] - summed[:-height, width:] - summed[height:, :-width] + summed[:-height, :-width]
    top, left = np.nonzero(region_size == width * height)
    if not len(top):
        raise ValueError("No complete white background patch")
    extent = None
    for near_x, near_y in ((.25, .25), (.75, .25), (.25, .75), (.75, .75)):
        squared = (left + width / 2 - near_x * mask.shape[1])**2 + (top + height / 2 - near_y * mask.shape[0])**2
        index = int(np.argmin(squared))
        x, y = int(left[index]), int(top[index])
        rect = (x, y, x + width, y + height)
        if extent is None:
            edit = rect
            _, _, extent = warm_profile(template, edit, mask.shape)
        elif x >= extent[2] or y >= extent[3] or rect[2] <= extent[0] or rect[3] <= extent[1]:
            return {"wall_left": edit, "protect": rect}
    raise ValueError("No protected white patch outside the edit")


def capture_frames(source, archive_entries, scene, area):
    decoded, identity = {}, {}
    for light in LIGHTS:
        filename = f"{scene}_{light}.jpg"
        photo = (source / filename).read_bytes()
        fingerprint = validated(archive_entries[scene, light], photo)
        with Image.open(BytesIO(photo)) as image:
            if image.format != "JPEG" or image.size != PHOTO_SIZE or image.mode != "RGB":
                raise ValueError(f"Unexpected preview decoding: {filename}")
            icc = image.info.get("icc_profile")
            exif = image.getexif().get_ifd(0x8769)
            camera = tuple(float(exif[tag]) for tag in (33434, 33437, 34855, 41987))
            if not icc or camera != (1., 16., 320., 1.):
                raise ValueError(f"Unexpected ICC or exposure metadata: {filename}")
            srgb, conversion = to_srgb(image)
            if "Linear Rec2020 RGB" not in conversion["source_profile"]:
                raise ValueError(f"Changed source color space: {filename}")
            decoded[light] = np.array(srgb)
            identity[light] = dict(file_sha256=fingerprint, bytes=len(photo), icc_sha256=sha256(icc).hexdigest(),
                                   camera=camera, conversion=conversion["source_profile"])
    x0, y0, x1, y1 = area
    gradients = {key: np.diff(frame[y0:y1, x0:x1] @ LUMA, axis=1).ravel()
                 for key, frame in decoded.items()}
    baseline = gradients[ANCHOR]
    for name, gradient in gradients.items():
        score = float(np.corrcoef(baseline, gradient)[0, 1]) if np.ptp(baseline) and np.ptp(gradient) else None
        identity[name].update(gradient=score, registered=bool(score is not None and score >= .70))
    return decoded, identity


def run(base, source, cli, out):
    code = Path(__file__)
    if (digest(code.with_name("mls-local-warmth-protocol.md")) != PROTOCOL
            or digest(code.with_name("shared-local-warmth-protocol.md")) != WARMTH_PROTOCOL):
        raise ValueError("Changed frozen test protocol")
    template = check(base / "playraw-flash-room-intake-v1/martin.pp3", PP3)
    check(cli, RT_CLI)
    profile = (cli.parent / "iccprofiles/output/RTv4_sRGB.icc").read_bytes()
    index = (base / "mls-warmth-index-audit-v1/preview-central-directory.bin").read_bytes()
    archive_entries = entries(index)
    material = base / "mls-material-identity"
    out.mkdir()
    with checked(material, "surfaces.zip") as surface:
        rows = table(surface, "surfaces/mask_to_surface.csv")
    env = dict(os.environ, RT_SETTINGS=str(out / "settings"), RT_CACHE=str(out / "rt-cache"), OMP_NUM_THREADS="1")
    reports = {}
    with checked(material, "masks_preview.zip") as masks:
        for scene in SCENES:
            row, = [value for value in rows if int(value["scene"]) == int(scene) and value["surface"] == "background_white"]
            mask = white_mask(masks.read(f"masks_preview/{scene}.png"), row)
            boxes = select_rectangles(mask, template)
            frames, captures = capture_frames(source, archive_entries, scene, boxes["wall_left"])
            results = prepare(cli, template, profile, env, out / scene, frames, boxes,
                              {name: value["registered"] for name, value in captures.items()})
            reports[scene] = dict(material=row, captures=captures, **results)
    json_to(out / "results.json", dict(scenes=reports, source_index=sha256(index).hexdigest(),
            protocol=PROTOCOL, script=digest(code), template=PP3, cli=RT_CLI,
            warmth_script=digest(code.with_name("probe_shared_local_warmth.py")),
            warmth_protocol=WARMTH_PROTOCOL))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("base", "source", "cli", "out"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
