"""Train-input output-sensitivity basis; no GT labels are consumed here."""
import numpy as np


def sensitivity_basis(gradients):
    g = np.asarray(gradients, dtype=np.float64)
    if g.ndim != 2 or min(g.shape) < 1 or not np.isfinite(g).all():
        raise ValueError('Expected nonempty finite gradient matrix')
    covariance = g.T @ g / len(g)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if values[0] <= 0:
        raise ValueError('No output sensitivity')
    for col in range(vectors.shape[1]):
        if vectors[np.argmax(np.abs(vectors[:, col])), col] < 0:
            vectors[:, col] *= -1
    return values, vectors
