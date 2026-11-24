"""Isolated image-replay research workbench using stock media and export routes."""
from argparse import ArgumentParser
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from threading import RLock
import os
import webbrowser

import cv2
import numpy as np
from PIL import Image

from canonical_model import ROOT
from launch_workbench import startup_lock
from probe_playraw_mixed_light_geometric_replay import align, texture_similarity
from probe_same_capture_detail_replay import shared_detail
from reference_color_app.storage import _json
from spatial_jobs import SpatialJobs
from spatial_render import finish_spatial
from spatial_store import SpatialStore
from spatial_workbench import make_spatial_workbench


LIMIT_PIXELS = 8_000_000


def integer_basis(target, reference):
    placed, support, trace = align(cv2.SIFT_create(), target, reference)
    if placed is None:
        return None, {"accepted": False, "reason": "geometry or texture unsupported", "alignment": trace}
    matrix = np.asarray(trace[-1]["matrix"])
    integer = np.array_equal(matrix[:, :2], np.eye(2)) and np.array_equal(matrix[:, 2], np.rint(matrix[:, 2]))
    if not integer:
        return None, {"accepted": False, "reason": "not an integer placement", "alignment": trace}
    detail = shared_detail(target, placed, support)
    decision = dict(accepted=detail["accepted"], alignment=trace, detail=detail)
    if not detail["accepted"]:
        return None, decision
    basis = target.copy()
    basis[support] = placed[support]
    return basis, decision


def reduced_basis(target, reference):
    height, width = target.shape[:2]
    small_height, small_width = reference.shape[:2]
    if not (height // 2 <= small_height <= 3 * height // 4
            and width // 2 <= small_width <= 3 * width // 4
            and abs(width * small_height - height * small_width) <= max(width, height)):
        return None, None

    size = (small_width, small_height)
    support = np.ones(reference.shape[:2], dtype=bool)
    resized = {
        "area": cv2.resize(target, size, interpolation=cv2.INTER_AREA),
        "lanczos": np.asarray(Image.fromarray(target).resize(size, Image.Resampling.LANCZOS)),
    }
    attempts = {}
    reductions = []
    for kernel, small in resized.items():
        detail = shared_detail(small, reference, support)
        similarity = texture_similarity(small, reference, support) if detail["accepted"] else None
        attempts[kernel] = dict(detail=detail, gradient_correlation=similarity)
        if similarity is not None and similarity >= .98:
            reductions.append(small)
    decision = {"accepted": False, "resized": attempts}
    if not reductions:
        decision["reason"] = "insufficient shared detail after downsampling"
        return None, decision
    if len(reductions) == 2 and not np.array_equal(*reductions):
        decision["reason"] = "ambiguous resampling residual"
        return None, decision

    delta = reference.astype(np.float32) - reductions[0]
    basis = cv2.resize(delta, (width, height), interpolation=cv2.INTER_LINEAR)
    basis += target
    np.clip(basis, 0, 255, out=basis)
    np.rint(basis, out=basis)
    decision["accepted"] = True
    return basis.astype(np.uint8), decision


def certified_basis(target, reference):
    if max(target.shape[0] * target.shape[1], reference.shape[0] * reference.shape[1]) > LIMIT_PIXELS:
        return None, {"accepted": False, "reason": "image exceeds research resource limit"}
    basis, decision = integer_basis(target, reference)
    if basis is None:
        basis, reduced = reduced_basis(target, reference)
        if reduced is not None:
            decision = dict(reduced, alignment=decision["alignment"])
    if basis is not None:
        basis.setflags(write=False)
    return basis, decision


class ReplayStore(SpatialStore):
    def __init__(self, root, engine, identity_tag):
        super().__init__(root, engine, identity_tag)
        self.replay_lock = RLock()
        self.replay_cache = OrderedDict()

    def _basis(self, record, target=None):
        key = record["cache_key"]
        with self.replay_lock:
            if key in self.replay_cache:
                self.replay_cache.move_to_end(key)
                return self.replay_cache[key]
            if target is None:
                target = self.get_media(record["target_id"])
            reference = self.get_media(record["reference_id"])
            left, _ = self.decode(target)
            right, _ = self.decode(reference)
            cv2.setRNGSeed(0)
            try:
                result = certified_basis(left, right)
            except cv2.error:
                result = None, {"accepted": False, "reason": "feature matching unavailable"}
            self.replay_cache[key] = result
            while len(self.replay_cache) > 2:
                self.replay_cache.popitem(last=False)
            return result

    def analyze(self, target_id, reference_id, time_seconds=0.):
        record = super().analyze(target_id, reference_id, time_seconds)
        target = self.get_media(target_id)
        if target["kind"] != "image" or record["diagnostics"]["exact_identity"]:
            return record
        _, decision = self._basis(record, target)
        record["diagnostics"]["same_capture_replay"] = decision
        if decision["accepted"]:
            record["diagnostics"]["mapping_scope"] = "Verified same-capture image replay; spatial overlap, PNG only"
        _json(self._record("mappings", record["id"]), record)
        return record

    def render(self, mapping_id, rgb, strength, adjustments=None, preview=False, verified_target=None):
        record = self.get_mapping_record(mapping_id)
        decision = record["diagnostics"].get("same_capture_replay", {})
        if strength == 0 or not decision.get("accepted"):
            return super().render(mapping_id, rgb, strength, adjustments, preview, verified_target)
        self._validated_grid(mapping_id, verified_target)
        basis, _ = self._basis(record, verified_target)
        if basis is None:
            raise ValueError("同原片核验结果已变化，请重新分析")
        if basis.shape != rgb.shape:
            if not preview:
                raise ValueError("同原片成片尺寸与目标不符")
            picture = Image.fromarray(basis)
            picture.thumbnail((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR)
            basis = np.array(picture)
            if basis.shape != rgb.shape:
                raise ValueError("同原片预览尺寸与目标不符")
        return finish_spatial(rgb, basis, strength, adjustments)


def make_candidate(root, port=0):
    cv2.setNumThreads(1)
    server = make_spatial_workbench(root, port, enhanced=True)
    directory = Path(__file__).parent
    tracked = (
        Path(__file__), directory / "probe_playraw_mixed_light_geometric_replay.py",
        directory / "probe_same_capture_detail_replay.py", directory / "probe_synclight_cross_view.py",
    )
    building = sha256((server.build_tag + server.store.engine_tag).encode())
    for path in tracked:
        building.update(path.read_bytes())
    server.build_tag = building.hexdigest()
    server.store = ReplayStore(root, server.store.engine, server.build_tag)
    server.manager = SpatialJobs(server.store, temporal=True)
    server.variant = "same-capture-image-research"
    body, content_type = server.assets["/"]
    current = "自适应参考调色研究版".encode()
    candidate = "同原片局部调色研究版".encode()
    server.assets["/"] = (body.replace(current, candidate), content_type)
    return server


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dev-data" / "same-capture-image-research")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    with startup_lock(args.data_dir, timeout=1), make_candidate(args.data_dir, args.port) as server:
        address = f"http://127.0.0.1:{server.server_port}"
        _json(server.store.root / "session.json", dict(url=address, pid=os.getpid(), token=server.token, app=server.variant))
        print(address, flush=True)
        if args.open:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
