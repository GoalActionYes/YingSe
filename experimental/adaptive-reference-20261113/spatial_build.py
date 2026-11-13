"""Snapshot identity used by the owned launcher and the running UI service."""
import hashlib
from pathlib import Path


def build_tag(enhanced=False):
    base=Path(__file__).parent;root=base.parents[1];digest=hashlib.sha256()
    files=[base/name for name in ['spatial_build.py','spatial_workbench.py','spatial_store.py','spatial_jobs.py','spatial_render.py','triplet_color.py','training_checks.py']]
    files += [root/'reference_color_app'/name for name in ['adjustments.py','web/index.html','web/app.js','web/styles.css','web/adjustments.js','web/history.js']]
    if enhanced:
        files.extend([base/'temporal_grid.py',base/'spatial_strength_preview.js',base/'spatial_manual_worker.js',base/'spatial_manual_preview.js']);digest.update(b'multiref-temporal-v1')
    for path in files:digest.update(path.name.encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()
