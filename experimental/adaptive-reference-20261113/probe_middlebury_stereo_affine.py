"""Compare full RGB affine grading with the diagonal stereo diagnostic."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from probe_middlebury_stereo_grade import run, STRENGTH, finish_spatial


def fit_affine(original, source, destination):
    design = np.column_stack((source / 255, np.ones(len(source))))
    result, _, rank, singular = np.linalg.lstsq(design, destination / 255, rcond=None)
    if rank != design.shape[1]:
        raise ValueError('Calibration colors do not span the full affine model')
    rgb = original.astype(np.float64) @ result[:3] + result[3] * 255
    graded = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
    return finish_spatial(original, graded, STRENGTH), {
        'matrix': result[:3].tolist(),
        'intercept_rgb8': (result[3] * 255).tolist(),
        'condition': float(singular[0] / singular[-1]),
    }


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()), fit_grade=fit_affine)
