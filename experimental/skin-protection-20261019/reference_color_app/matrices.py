"""Independent static port of recovered color-matching matrix routines.

Evidence: core-pass12 fn-35b00c0; core-pass4 fn-35af240/35aeed0/
35b1d50; core-pass11 fn-35b5f30; parameter-rules-audit. Static reconstruction,
not yet validated against native runtime matrix output. Row-major arrays;
column-vector affine convention. No np.linalg whitening approximation.
"""
from __future__ import annotations

import math
import numpy as np

f = np.float32


def matmul(left, right):
    """Recovered ordered float32 multiply/add (no BLAS, no fused FMA)."""
    left, right = np.asarray(left, np.float32), np.asarray(right, np.float32)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError('matrix dimensions differ from recovered valid branch')
    out = np.zeros((left.shape[0], right.shape[1]), np.float32)
    for i in range(left.shape[0]):
        for j in range(right.shape[1]):
            value = f(0)
            for k in range(left.shape[1]):
                value = f(value + f(left[i, k] * right[k, j]))
            out[i, j] = value
    return out


def eigen2(matrix):
    """35af240 and inlined first stage in 35b00c0, exact branch/order.

    Eigenvalues are plus-root then minus-root; diagonal input deliberately
    keeps identity vectors even when the descending values change order.
    This preserves the observed code, rather than fixing its special case.
    """
    m = np.asarray(matrix, np.float32)
    if m.shape != (2, 2) or not np.isfinite(m).all():
        raise ValueError('expected finite 2x2 matrix')
    a, b, c, d = m.ravel()
    trace = f(d + a)
    discriminant = f(f(f(trace * trace) * f(.25)) - f(f(d * a) - f(c * b)))
    if discriminant < 0:
        raise ArithmeticError('negative native eigen discriminant; CRT NaN path unimplemented')
    root = math.sqrt(float(discriminant))
    half = f(trace * f(.5))
    e = np.array([f(float(half) + root), f(float(half) - root)], np.float32)
    v = np.eye(2, dtype=np.float32)
    if c != 0:
        x0, x1 = f(e[0] - d), f(e[1] - d)
        n0 = f(math.sqrt(float(f(f(x0*x0) + f(c*c)))))
        n1 = f(math.sqrt(float(f(f(x1*x1) + f(c*c)))))
        if n0 == 0 or n1 == 0:
            raise ArithmeticError('zero native eigenvector norm')
        v = np.array([[f(x0/n0), f(x1/n1)], [f(c/n0), f(c/n1)]], np.float32)
    elif b != 0:
        y0, y1 = f(e[0] - a), f(e[1] - a)
        n0 = f(math.sqrt(float(f(f(y0*y0) + f(b*b)))))
        n1 = f(math.sqrt(float(f(f(y1*y1) + f(b*b)))))
        if n0 == 0 or n1 == 0:
            raise ArithmeticError('zero native eigenvector norm')
        v = np.array([[f(b/n0), f(b/n1)], [f(y0/n0), f(y1/n1)]], np.float32)
    return e, v


def solve_covariance(source_cov, reference_cov):
    """35b00c0: native ordered covariance transport, returns affine A, e_s.

    X = sqrt(Ds) Vs.T Cr Vs sqrt(Ds)
    A2 = Vs invsqrt(Ds) Vx sqrt(Dx) Vx.T invsqrt(Ds) Vs.T
    Each multiplication rounds to float32. Inverses add double 1e-6 to
    already float32 square roots. Source eigenvalues < double 1e-7 => I3.
    """
    e, v = eigen2(source_cov)
    out = np.eye(3, dtype=np.float32)
    if float(e[0]) < 1e-7 or float(e[1]) < 1e-7:
        return out, e
    es = np.array([f(max(1e-6, float(x))) for x in e], np.float32)
    ds = np.diag(np.array([f(math.sqrt(abs(float(x)))) for x in es], np.float32))
    x = matmul(ds, v.T)
    x = matmul(x, reference_cov)
    x = matmul(x, v)
    x = matmul(x, ds)
    ex, vx = eigen2(x)
    dx = np.diag(np.array([f(math.sqrt(max(0., float(z)))) for z in ex], np.float32))
    inv = np.diag(np.array([f(1. / (float(ds[k, k]) + 1e-6)) for k in range(2)], np.float32))
    a = matmul(v, inv)
    a = matmul(a, vx)
    a = matmul(a, dx)
    a = matmul(a, vx.T)
    a = matmul(a, inv)
    a = matmul(a, v.T)
    out[:2, :2] = a
    return out, e


def negligible(translation, covariance):
    """35b4be0: strict thresholds, abs means < f32 .0039, cov < 1e-5."""
    return bool(np.all(np.abs(translation) < f(.0039)) and
                np.all(np.abs(covariance).astype(np.float64) < 1e-5))


def solve_matrices(source63, reference57):
    """Decode caches and port 35b5f30's three groups.

    Returns T/R/A/F: (3,3,3), C: (3,2,2), flags: (3,) int32,
    eigenvalues: source eigensolver outputs per group (NaN when not run),
    max_excess: f32 O+154, diagnostics: explicit unobserved-state notes.

    T is a copied source affine block with its two translation values negated;
    R is a copied reference affine block. C copies source cached inverse cov.
    Flags start from reference[48:51], preserving native float->int truncation.
    Fresh-object F starts zero; reference-negligible groups preserve zero.
    Reused native object history is outside this stateless interface.
    """
    source = np.array(source63, np.float32, copy=True)
    reference = np.array(reference57, np.float32, copy=True)
    if source.shape != (63,) or reference.shape != (57,):
        raise ValueError('expected source63 and reference57 feature packets')
    if not np.isfinite(source).all() or not np.isfinite(reference).all():
        raise ValueError('nonfinite packets are outside validated native branches')
    sc = source[:12].reshape(3, 2, 2)
    rc = reference[:12].reshape(3, 2, 2)
    t = source[12:39].reshape(3, 3, 3).copy()
    r = reference[12:39].reshape(3, 3, 3).copy()
    c = source[39:51].reshape(3, 2, 2).copy()
    flags = reference[48:51].astype(np.int32)
    a = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
    result_f = np.zeros((3, 3, 3), np.float32)
    eigenvalues = np.full((3, 2), np.nan, np.float32)
    last_eigen = None
    max_excess = f(0)
    diagnostics = []
    for i in range(3):
        if not negligible(t[i, :2, 2], sc[i]):
            a[i], last_eigen = solve_covariance(sc[i], rc[i])
            eigenvalues[i] = last_eigen
        # Multiplication by -1 rounds/negates the original copied values.
        t[i, :2, 2] = np.negative(t[i, :2, 2])
        if negligible(r[i, :2, 2], rc[i]):
            flags[i] = 1
            diagnostics.append(f'group {i}: fresh F remains zero; native skips write')
            continue
        if i == 1:
            if last_eigen is None:
                # A is already I: either unknown ratio branch leaves it I.
                diagnostics.append('group 1: unobserved eigen scratch; A=I in both branches; FP status not modeled')
            else:
                ratio = f(float(max(last_eigen)) / (float(min(last_eigen)) + 1e-5))
                if ratio > f(70):
                    a[i] = np.eye(3, dtype=np.float32)
        else:
            maximum = f(np.max(np.abs(a[i])))
            if maximum > f(1.75):
                excess = f(float(maximum) - 1.75)
                clipped = f(min(max(float(excess), 0.), .4))
                max_excess = f(max(float(max_excess), float(clipped)))
                k = f(.6 if float(np.max(np.abs(t[i, :2, 2]))) > .15 else .3)
                divisor = f(max(float(f(k * maximum)), 1.))
                a[i] = np.asarray(a[i] / divisor, np.float32)
                if float(np.max(np.abs(a[i]))) > 3.2:
                    a[i] = np.eye(3, dtype=np.float32)  # verified 35b1ce0
        result_f[i] = matmul(r[i], matmul(a[i], t[i]))
    return dict(T=t, R=r, A=a, F=result_f, C=c, flags=flags,
                max_excess=max_excess, eigenvalues=eigenvalues,
                diagnostics=diagnostics)
