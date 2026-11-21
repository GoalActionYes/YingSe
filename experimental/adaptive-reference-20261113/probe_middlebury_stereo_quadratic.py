"""Add existing quadratic RGB terms to the calibrated stereo diagnostic."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from capacity_audit import polynomial_features
from probe_middlebury_stereo_grade import STRENGTH, finish_spatial, run


def fit_quadratic(original, source, destination):
    samples = source.astype(np.uint8).reshape(-1, 1, 3)
    design = polynomial_features(samples, 2)
    coefficients, _, rank, singular = np.linalg.lstsq(design, destination / 255, rcond=None)
    if rank != design.shape[1]:
        raise ValueError('Calibration colors do not span the quadratic model')
    predicted = (polynomial_features(original, 2) @ coefficients).reshape(original.shape)
    graded = np.rint(np.clip(predicted, 0, 1) * 255).astype(np.uint8)
    return finish_spatial(original, graded, STRENGTH), {
        'coefficients': coefficients.tolist(),
        'condition': float(singular[0] / singular[-1]),
    }


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    run(**vars(parser.parse_args()), fit_grade=fit_quadratic)
