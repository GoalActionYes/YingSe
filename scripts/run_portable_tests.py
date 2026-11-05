"""Source-only checks, with external evidence exclusions stated explicitly."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests/reference_color')]


def main():
    names = ['test_aligned_reference', 'test_calibration', 'test_pipeline',
             'test_parameters', 'test_matrices', 'test_http', 'test_jobs',
             'test_lut.LUTTests']
    print('Source checks exclude test_evidence, test_features and test_lut.FrozenGPUChecks: external native/GPU fixtures.', flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
