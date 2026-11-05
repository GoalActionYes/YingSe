"""Independent static reconstruction of native color-match parameter packing.

Evidence: cccreator RVA 35b4ab0 (pass6), 35b6a50/35b6e10 (pass8),
35b5d70 (pass2), feature-layout-audit and parameter-rules-audit.
No target DLL is loaded. This is not a claim of native output equivalence.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

F = np.float32
TABLE_PATH = Path(__file__).with_name("curve_tables.json")
TABLE_SHA256 = "21e1ef57d474a2d177fb0bf83d6e8bc1118f59dd679a0e59505ae3192c481e55"


def _clamp(x, lo, hi):
    # Original comparison ordering, including NaN selecting lower bound.
    if lo <= x:
        return F(x if x <= hi else hi)
    return F(lo)


def _tables():
    raw = TABLE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TABLE_SHA256:
        raise ValueError("Packaged curve table integrity check failed")
    payload = json.loads(raw)
    result = {}
    for address, item in payload.items():
        data = bytes(item["values"])
        if len(data) != 256 or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Extracted scalar curve table failed integrity check")
        result[int(address, 16)] = np.frombuffer(data, dtype=np.uint8)
    return result


def _base_scalar(L, Lr, U, Ur, flag):
    tables = _tables()
    # cvttss2si: truncation toward zero, with valid finite input required below.
    l, lr, u, ur = map(int, (L, Lr, U, Ur))
    if not 0 <= l <= 255:
        raise ValueError("Native curve table index outside 0..255")
    addr = 0x185141de0 if l <= lr else (0x185141ce0 if flag else 0x185141be0)
    sden = abs(int(tables[addr][l]) - l)
    base = F(.71 if ur < u else 1.4)
    exponent = F(F(u) / F(255))
    # NumPy float32 pow models float inputs/output, but not a proven UCRT powf.
    power = F(np.power(base, exponent))
    hden = abs(int(F(power * F(u))) - u)
    with np.errstate(divide="ignore", invalid="ignore"):
        sraw = F(np.divide(F(lr-l), F(sden)))
        hraw = F(np.divide(F(ur-u), F(hden)))
    return _clamp(sraw, F(-1), F(1)), _clamp(hraw, F(-1), F(1)), {
        "table_va": hex(addr), "truncated_inputs": [l, lr, u, ur],
        "shadow_denominator": sden, "highlight_denominator": hden,
        "pow_base": float(base), "pow_exponent": float(exponent),
        "pow_value": float(power), "unclamped": [float(sraw), float(hraw)],
    }


def scalar_rules(source63, reference57):
    """Return fresh-process scalar state; does not reuse native mutable state."""
    L, M, U = source63[60:63]
    Lr, Mr, Ur = reference57[54:57]
    flag = bool(F(M-Mr) > F(10))
    S, H, info = _base_scalar(L, Lr, U, Ur, flag)
    black, white = int(reference57[51]), int(reference57[52])
    enable, contrast = False, F(1)
    dL, dU = F(L-Lr), F(U-Ur)
    if F(dL*dU) > 0:
        if L < Lr and U < Ur:
            if M <= F(70):
                S, H = _clamp(S, 0., .8), _clamp(H, 0., .3)
            else:
                S, H = _clamp(S, 0., .5), _clamp(H, 0., .15)
            if F(Lr-L) > F(70) and L > F(50):
                black = min(black, 10)
            contrast, enable = F(1.15), True
            branch = "both_source_darker"
        else:
            if float(H) < -.05:
                contrast, enable = F(1.1), True
                S = _clamp(S, -.8, 0.)
            if L < F(90):
                S = _clamp(S, -.45, 0.)
            H = _clamp(H, -.3 if flag else -.6, 0.)
            if 0 < dU < F(10):
                H = F(.08 if flag else 0.)
                if flag:
                    white = 255
            if dL > F(40):
                S = _clamp(S, -.6, 0.)
            black = 0
            branch = "both_source_brighter"
    else:
        S = _clamp(S, -.5 if flag else -.6, 1.)
        black = 0
        if L < Lr:
            if M < Mr:
                S = F(float(S)*1.5)
            enable, contrast = True, F(1.15)
            H = F(max(float(H), -.1))
        branch = "opposite_or_zero_deltas"
    info.update(branch=branch, flag=flag, S=float(S), H=float(H),
                black=black, white=white, enable=enable, contrast=float(contrast))
    return S, H, black, white, enable, contrast, info


def _mask_q(a, b):
    if a < 0 and b <= 0:
        return [1, F(1.2), 0, 0, 1, 0, 0]
    if a < 0 and b > 0:
        return [1, 1, 0, 0, 0, 1, 0]
    if a > 0 and b < 0:
        return [0, 0, 1, F(.4), 1, 0, 0]
    if a >= 0 and b > 0:
        return [0, 0, 1, F(.4), 0, 1, 2 if b < F(.15) else F(.2)]
    return [0, 0, 1, 0, 0, 1, 0]


def _matmul_native(left, right):
    # ADDSS/MULSS ordering in native 3x3 loop: term1+term0, then term2.
    out = np.empty((3, 3), np.float32)
    for i in range(3):
        for j in range(3):
            out[i, j] = F(F(F(left[i, 1]*right[1, j]) +
                              F(left[i, 0]*right[0, j])) + F(left[i, 2]*right[2, j]))
    return out


def _limited_translation(matrix):
    out = np.eye(3, dtype=np.float32)
    uv = np.asarray(matrix, np.float32)[:2, 2].copy()
    maximum = F(max(abs(uv[0]), abs(uv[1])))
    if maximum > F(.08):
        uv = np.asarray(uv * F(F(.08)/maximum), np.float32)
    out[:2, 2] = uv
    return out


def solve_parameters(source63, reference57, strength=1.0):
    """Return (37x4 float32 parameters, diagnostics).

    strength is native object+4b0 (default constructor value 1), NOT UI
    strength. Native LUT layers use .8/1; final UI intensity belongs elsewhere.
    Inputs are independent serialized feature arrays; no cache files are read.
    """
    source = np.asarray(source63, dtype=np.float32)
    reference = np.asarray(reference57, dtype=np.float32)
    if source.shape != (63,) or reference.shape != (57,):
        raise ValueError("Expected exactly 63 and 57 float32 values")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(reference)):
        raise ValueError("Nonfinite feature payload is outside this probe's supported domain")
    t = F(strength)
    if not np.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("Supported native internal strength domain is [0,1]")
    from .matrices import solve_matrices
    matrices = solve_matrices(source, reference)
    S, H, black, white, enable, contrast, info = scalar_rules(source, reference)
    limited = min(float(t), .8)
    a, b = F(float(S)/.8*limited), F(float(H)/.8*limited)
    contrast = _clamp(F(float(F(contrast-F(1)))/.8*float(t)+1), 0., 1.15)
    black_value = F(float(black)/.8*float(t))
    white_value = F(255.-float(255-white)/.8*float(t))
    q = _mask_q(a, b)
    # NaN marks native unwritten fields; ignored by shipped shader.
    p = np.full((37, 4), np.nan, np.float32)
    p[0] = [a, b, q[0], q[1]]
    p[1] = [q[2], q[3], q[4], q[4]]
    p[2] = [q[5], q[6], 2 if enable else 0, contrast]
    p[3, :2] = [F(black_value/F(255)), F(white_value/F(255))]
    flags = np.asarray(matrices["flags"])
    without_uv = bool(np.all(flags == 1))
    if not without_uv:
        T, R, C, fm = (np.asarray(matrices[k], np.float32) for k in ("T", "R", "C", "F"))
        # Returned T must already be mean-negated, as after native 35b5f30.
        p[3, 2:] = -T[0, :2, 2]
        p[4] = [C[0,0,0], C[0,1,0], C[0,0,1], C[0,1,1]]
        p[5] = [-T[1,0,2], -T[1,1,2], C[1,0,0], C[1,1,0]]
        p[6] = [C[1,0,1], C[1,1,1], -T[2,0,2], -T[2,1,2]]
        p[7] = [C[2,0,0], C[2,1,0], C[2,0,1], C[2,1,1]]
        p[8] = np.asarray([int(x) for x in source[55:59]], np.float32)/F(255)
        for i in range(3):
            p[9+i*3:12+i*3, :3] = fm[i]
            p[18+i*3:21+i*3, :3] = _matmul_native(R[2-i], T[2-i])
        B = [_limited_translation(x) for x in R]
        for row in range(3):
            for group in range(3):
                p[27+row*3+group, :3] = B[group][row]
        ratio = np.asarray(reference[39:42], np.float64)/(np.asarray(source[51:54], np.float64)+.0001)
        p[36, :3] = [_clamp(x, .7, 1.) for x in ratio]
    consumed = [(row, col) for row in range(37) for col in range(4)
                if (row < 3 or (row == 3 and col < 2) or
                    (not without_uv and (row < 9 or col < 3)))]
    diagnostics = {
        "consumed_indices": [list(x) for x in consumed],
        "consumed_values_finite": bool(all(np.isfinite(p[x]) for x in consumed)),
        "shader_layer_strengths": [.8, 1.],
        "parameter_buffer_shared_by_layers": True,
        "scalar": info, "internal_strength": float(t), "without_uv": without_uv,
        "matrix_diagnostics": {k: v for k,v in matrices.items() if k not in ("T","R","A","F","C")},
        "reference_4ac_loaded": float(reference[53]),
        "reference_4ac_usage": "Not read by audited scalar/packing functions; retained for upper GPU dispatch investigation",
        "unwritten_indices": np.argwhere(np.isnan(p)).tolist(),
        "limitations": ["NumPy pow float32 is not yet UCRT powf bit-equivalence validated",
                        "No comparison to captured native 37vec4 buffer performed",
                        "NaN output slots represent unwritten native fields, not submitted shader values"],
    }
    return p, diagnostics
