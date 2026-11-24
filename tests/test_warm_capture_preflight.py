from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


EXPERIMENT = Path(__file__).resolve().parents[1] / "experimental/adaptive-reference-20261113"
sys.path.insert(0, str(EXPERIMENT))
from warm_capture_preflight import read_inputs, region_bounds, registration


class WarmCapturePreflightTest(unittest.TestCase):
    def test_rejects_out_of_bounds_and_overlapping_protection(self):
        shape = (64, 64, 3)
        edit = (8, 9, 20, 21)
        self.assertEqual(region_bounds(shape, edit, [(30, 30, 45, 45)]),
                         {"wall_left": edit, "guard_1": (30, 30, 45, 45)})
        with self.assertRaisesRegex(ValueError, "inside"):
            region_bounds(shape, edit, [(60, 30, 65, 40)])
        with self.assertRaisesRegex(ValueError, "overlap"):
            region_bounds(shape, edit, [(11, 10, 30, 37)])

    def test_duplicate_or_different_geometry_stops_before_render(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            first, second = base / "a.png", base / "b.png"
            Image.new("RGB", (48, 48), (100, 110, 120)).save(first)
            Image.new("RGB", (48, 48), (100, 110, 120)).save(second)
            with self.assertRaisesRegex(ValueError, "identical"):
                read_inputs(first, second)
            Image.new("RGB", (48, 49), (100, 110, 120)).save(second)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                read_inputs(first, second)

    def test_registration_rejects_real_pixel_shift(self):
        image = np.random.default_rng(24).integers(30, 200, (64, 64, 3), dtype=np.uint8)
        stationary = image + 7
        shifted = np.roll(stationary, 2, axis=1)
        self.assertTrue(registration(image, stationary)["passed"])
        self.assertFalse(registration(image, shifted)["passed"])

    def test_untextured_registration_remains_unconfirmed(self):
        image = np.full((64, 64, 3), 64, dtype=np.uint8)
        self.assertFalse(registration(image, image + 2)["passed"])

    def test_small_local_shift_can_pass_global_registration(self):
        source = np.random.default_rng(8).integers(30, 200, (96, 96, 3), dtype=np.uint8)
        moved = source + 3
        patch = moved[31:41, 31:41].copy()
        moved[31:41, 31:41] = np.roll(patch, 2, axis=1)
        self.assertFalse(np.array_equal(moved[31:41, 31:41], patch))
        self.assertTrue(registration(source, moved)["passed"])
