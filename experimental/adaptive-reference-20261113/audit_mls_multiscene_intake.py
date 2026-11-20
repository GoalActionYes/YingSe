"""Check native MLS masks and exposure for the fixed multi-scene material panel."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageFilter

from audit_mls_material_identity import checked, table
from fetch_mls_linear_pilot import INDEX_SHA, LIGHTS
from independent_coco import digest
from probe_fau_two_light_response import encode
from probe_mls_light_response import registration
from probe_mls_linear_pilot import read, valid_pixels


SCENES = {
    "02": "paper_prism_blue",
    "03": "paper_prism_blue",
    "04": "plastic_stick_green",
    "05": "paper_prism_blue",
    "06": "plastic_stick_green",
    "09": "plastic_stick_green",
}


def palette(archive, member, size):
    with Image.open(BytesIO(archive.read(member))) as image:
        assert image.mode == "RGB"
        if size is not None:
            image = image.resize(size, Image.Resampling.NEAREST)
        return np.asarray(image)


def run(original, added, material):
    mask_path = added / "masks_16bit_color.zip"
    assert mask_path.stat().st_size < 1024**2
    with checked(material, "surfaces.zip") as spectra:
        rows = table(spectra, "surfaces/mask_to_surface.csv")
    sources = {
        source: json.loads((source / "intake.json").read_bytes())
        for source in (original, added)
    }
    assert all(receipt["index_sha256"] == INDEX_SHA for receipt in sources.values())
    results = {}
    with checked(material, "masks_preview.zip") as previews, ZipFile(mask_path) as masks:
        assert masks.testzip() is None
        mask_names = {Path(name).name: name for name in masks.namelist() if name.endswith(".png")}
        for scene, material_name in SCENES.items():
            source = original if scene in ("02", "03") else added
            receipt = sources[source]["entries"]
            names = [f"{scene}_{light}.png" for light in LIGHTS]
            raw = [read(source / name, receipt[name]) for name in names]
            assert all(frame.shape == (950, 1157, 3) for frame in raw)
            scaled = [frame.astype(np.uint32) * 4 for frame in raw]
            frames = [encode(frame.clip(0, 65535)) for frame in scaled]
            color, = [row["mask_colour_rgb"] for row in rows
                      if int(row["scene"]) == int(scene) and row["surface"] == material_name]
            rgb = np.fromstring(color, dtype=int, sep=" ")
            native = palette(masks, mask_names[f"{scene}.png"], None)
            assert native.shape == raw[0].shape
            preview = palette(previews, f"masks_preview/{scene}.png", (1157, 950))
            native_region = (native == rgb).all(axis=-1)
            preview_region = (preview == rgb).all(axis=-1)
            overlap = float((native_region & preview_region).sum() / (native_region | preview_region).sum())
            interior = np.asarray(Image.fromarray(native_region.astype(np.uint8) * 255)
                                  .filter(ImageFilter.MinFilter(5))) > 0
            count = int((interior & valid_pixels(*frames)).sum())
            interior_count = int(interior.sum())
            fraction = count / interior_count if interior_count else 0.0
            alignment = registration(*frames)
            results[scene] = {
                "material": material_name,
                "input_sha256": [receipt[name]["sha256"] for name in names],
                "dimensions": [1157, 950],
                "paired_edge_correlation": alignment,
                "mask_iou": overlap,
                "interior_pixels": interior_count,
                "common_valid_pixels": count,
                "common_valid_fraction": fraction,
                "native_saturated_pixels": [int((frame == 65535).any(axis=-1).sum()) for frame in raw],
                "gain_clipped_pixels": [int((frame > 65535).any(axis=-1).sum()) for frame in scaled],
                "qualified": alignment >= .5 and overlap >= .90 and count >= 256 and fraction >= .5,
            }
    return {
        "image_index_sha256": INDEX_SHA,
        "native_masks_sha256": digest(mask_path),
        "native_masks_bytes": mask_path.stat().st_size,
        "added_source": sources[added],
        "scenes": results,
        "qualified": all(row["qualified"] for row in results.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("original", "added", "material", "result"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    result = run(args.original, args.added, args.material)
    with args.result.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
