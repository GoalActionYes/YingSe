"""Test hard background-direction removal on the existing crossed-edit development set."""
import argparse
from pathlib import Path

import numpy as np

from probe_crossed_edit_pair_supervision import EDITS, ROLES, feature, run, solve, training_levels


def solve_orthogonal(fitting, local):
    baseline, details = solve(fitting, local)
    center, scale, intercept, base_weights = baseline["per-image"]
    levels = training_levels(fitting, local)
    swaps = np.stack([levels[i, "F1", "none"] - levels[i, "C1", "none"] for i in range(len(fitting))]) / scale
    _, singular, right = np.linalg.svd(swaps, full_matrices=True)
    tolerance = max(swaps.shape) * np.finfo(swaps.dtype).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if not 0 < rank < swaps.shape[1]:
        raise ValueError("No usable separation of fitted background directions")
    remaining = right[rank:].T
    labels = np.asarray([EDITS[name] for _, _, name in levels])
    x = ((np.stack(list(levels.values())) - center) / scale) @ remaining
    system = x.T @ x + np.eye(x.shape[1])
    restricted = np.linalg.solve(system, x.T @ (labels - intercept))
    weights = remaining @ restricted
    residual = float(np.max(np.abs(swaps @ weights)))
    if residual > 1e-10:
        raise ValueError("Background constraint failed")

    def audit_features(observed):
        def retained(first, second):
            delta = second - first
            normalized = feature(np.zeros_like(delta), delta) / scale
            norm = np.linalg.norm(normalized)
            return float(np.linalg.norm(normalized @ remaining) / norm) if norm else None

        return {
            "background_retained_fraction": {
                name: retained(observed["C1", name], observed["F1", name]) for name in EDITS
            },
            "edit_retained_fraction": {
                f"{role}-{name}": retained(observed[role, "none"], observed[role, name])
                for role in ROLES for name in EDITS if name != "none"
            },
        }

    models = {"per-image": baseline["per-image"], "orthogonal": (center, scale, intercept, weights)}
    audit = {
        "level_rows": details["level_rows"], "features": details["features"], "background_rows": len(swaps),
        "background_rank": rank, "singular_values": singular.tolist(), "rank_tolerance": float(tolerance),
        "training_background_coefficient_change": {
            "per-image": float(np.abs(swaps @ base_weights).mean()),
            "orthogonal": float(np.abs(swaps @ weights).mean()),
        },
        "max_constraint_residual": residual,
        "max_projected_normal_residual": float(np.max(np.abs(system @ restricted - x.T @ (labels - intercept)))),
    }
    return models, audit, audit_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.output, fit_models=solve_orthogonal, candidate="orthogonal")
