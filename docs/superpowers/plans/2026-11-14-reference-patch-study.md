# Reference Feature Token Study Implementation Plan

> For agentic workers: execute inline with `superpowers:executing-plans`; use one read-only review before publication. User authorizes autonomous implementation and testing.

**Goal:** Test whether 16 reference feature tokens improve real reference color/tone/saturation transfer over collapsed-mean tokens and external C10, without changing the usable app.

**Architecture:** Follow `experimental/adaptive-reference-20261113/reference-patch-protocol.md` exactly. Current13 base plus one zero-output attention residual; same parameter/state/budget controls. Existing assets remain private and model/app defaults stay unchanged.

**Tech Stack:** Existing Python/PyTorch CUDA environment, NumPy/Pillow, unittest, native Git. No installs.

## Task 1: module and invariants

- [x] Add `test_reference_patch.py`; observe missing-module failure.
- [x] Implement `reference_patch.py`: ReferencePatchContext, pooled/repeated token helper, enable/init, grid/forward wrappers, finite input/output checks.
- [x] Prove 65,664 params, zero residual, output-projection gradients, later q/key gradients, float64 pooled query-invariance and permutation-invariance, distinct-token query dependence, validation.
- [x] Run synthetic/native64 initial bridge in training preflight versus current13 before any optimizer update.

## Task 2: two new fixed training runs

- [x] Implement `train_reference_patch.py --tokens` (candidate) or default (mean control), frozen source/config and asset checks. Save every output/weight/order and loss/gradient statistics.
- [x] Initialize both from current13 plus seed20261126 new head; reseed20261124 before batches. Copy/check complete initial state across arms, no old candidate warm start.
- [x] Reproduce external C10 selected64 output/metrics before using benchmark. Retain its complete snapshot; never treat it as matched control.
- [x] Train mean control to terminal, then token candidate to terminal. Same1024 rows/8epochs2048updates/minRGB0..8 earliest selection. Re-poll live handles, never restart on wait timeout.
- [x] Require both full unchanged gates against mean control and C10; no new independent pixels if either fails.

## Task 3: audit and inspection

- [x] Add `audit_reference_patch.py`, freeze actual metric definition source. Verify both initial/8trained weights, all prepared assets, source hashes, target orders, all1280 baseline/epoch PNG scores and minRGB selection.
- [x] Native reproduce current/control/candidate selected192 outputs and full RGB/delta/tone scores; recheck external C10. Save all64 contact sheets and inspect fixedfirst2/allsevere/max gain/max regression.
- [x] Run full research unittest suite, retain red-green record and any runtime failures.

## Task 4: handoff and publication

- [x] Create study report and sanitized numeric evidence under1MiB each, update MILESTONES/RESEARCH_PROGRESS/experimental README, distinguish research from feature upgrade.
- [ ] Read-only review, UTF8/links/source/evidence/secret/staged-scope checks, commit. Publish coherent batch including prior769d3f6 via atomic push, verify remote branch/tag exact.
- [ ] If both full gates pass, stop short of claiming final objective: freeze fresh independent paired evaluation, then original-size/video/stability/latency/UI acceptance before default changes. If failed, retain results and do not tune this protocol.
