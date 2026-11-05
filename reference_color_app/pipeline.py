"""Reviewed feature/solver/LUT composition with portable diagnostic provenance."""
import hashlib
import math
from pathlib import Path
import numpy as np
from .contracts import Mapping
from .features import extract_features
from .lut import build_mapping
from .calibration import calibrate
from .matrices import solve_matrices
from .parameters import solve_parameters
from .media import decode_image, decode_frame, probe_media


def json_safe(value):
    """Return JSON-compatible copies; nonfinite diagnostic scalars become null.

    Array positions are retained. Solver unwritten_indices and original solver
    diagnostics supply provenance; no layer data are sanitized by this helper.
    """
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def file_sha256(path):
    """Hash only a specifically supplied input file, with bounded read memory."""
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def algorithm_fingerprint():
    """Stable SHA256 of named algorithm sources and curve data, never paths."""
    digest=hashlib.sha256();base=Path(__file__).parent
    for name in ('contracts.py','features.py','matrices.py','parameters.py','lut.py',
                 'media.py','pipeline.py','calibration.py','curve_tables.json'):
        raw=(base/name).read_bytes()
        digest.update(name.encode('ascii')+b'\0'+len(raw).to_bytes(8,'big')+raw)
    return digest.hexdigest()


def layer_sha256(layers):
    """Hash C-order little-endian float32 two-layer bytes (axes RGB)."""
    return hashlib.sha256(np.asarray(layers,dtype='<f4').tobytes(order='C')).hexdigest()


def analyze(target_path,reference_path,time_seconds=0.0) -> Mapping:
    reference_meta=probe_media(reference_path)
    if reference_meta['kind']!='image':raise ValueError('The reference must be an image')
    target_meta=probe_media(target_path)
    if target_meta['kind']=='image':source_rgb=decode_image(target_path)
    else:source_rgb,target_meta=decode_frame(target_path,time_seconds)
    reference_rgb=decode_image(reference_path)
    source=extract_features(source_rgb,0);reference=extract_features(reference_rgb,1)
    mapping=build_mapping(source,reference)
    if np.array_equal(source_rgb,reference_rgb):
        raw_hash=layer_sha256(mapping.layers)
        mapping=calibrate(mapping,mapping)
        mapping.diagnostics['neutral_calibration'].update(raw_layer_sha256=raw_hash,
            neutral_layer_sha256=raw_hash,scope='Exact decoded RGB equality only; unequal images use recovered path unchanged')
    matrices=solve_matrices(source.values,reference.values)
    parameters,solver=solve_parameters(source.values,reference.values)
    diagnostics=dict(mapping.diagnostics)
    diagnostics.update(source_packet=dict(mode=source.mode,values=source.values,diagnostics=source.diagnostics),
        reference_packet=dict(mode=reference.mode,values=reference.values,diagnostics=reference.diagnostics),
        matrices=matrices,parameter_buffer=parameters,parameter_solver=solver,
        nonfinite_diagnostic_policy='null preserves index; parameter_solver.unwritten_indices identifies native unwritten slots; scalar unclamped nonfinite results retain solver provenance',
        inputs=dict(target=dict(sha256=file_sha256(target_path),metadata=target_meta,
                                original_dimensions=[source_rgb.shape[1],source_rgb.shape[0]]),
                    reference=dict(sha256=file_sha256(reference_path),metadata=reference_meta,
                                   original_dimensions=[reference_rgb.shape[1],reference_rgb.shape[0]])),
        algorithm_fingerprint=algorithm_fingerprint(),layer_sha256=layer_sha256(mapping.layers),
        layer_hash_format='C-order little-endian float32; shape 2x64x64x64x3; RGB axes')
    return Mapping(mapping.layers,json_safe(diagnostics))
