# Reference token execution ledger

Binding protocol: `reference-patch-protocol.md`; implementation plan: `docs/superpowers/plans/2026-11-14-reference-patch-study.md`.

Ruling: continue in the existing research checkout/branch, which already contains pinned private environments/assets and is used by prior reproducible studies. Do not create a second worktree or move these assets. This is branch isolation, not a linked worktree; stable main and running app defaults stay untouched.

Ruling: implement inline; subagents only provide read-only review. Existing autonomous development/testing/publication authorization applies. Publication remains a complete research batch, never a tiny UI/feature release claim.

Task 1 started: current state769d3f6 verified, only unrelated Codex/讨论 untracked files present. Protocol read unchanged. No new independent-image allocation.

Task 1 complete: six module/gate tests observed red then green; full232-test suite passed14.825s. Read-only preflight review found no important implementation issue. Initial64 raw/grid/RGB bridges (enabled/disabled/opposite mode) and externalC10 native/full metrics passed in the mean run.

Task 2: mean-control session81300 terminated exit0 after8epochs2048updates. MinimumRGB selects epoch7, mean11.565264860788982; control is not an eligible candidate. All observed control q/key gradients zero, value/output gradients nonzero. Candidate session99232 started from full matching initial state; poll the same handle until terminal, no parameter/protocol changes.

Task 3 prepared: audit_reference_patch.py will recompute all20 score tables and native predictions, plus fixed-selected-state collapsed/disabled counterfactuals. Read-only audit review found no important coverage issue; counterfactuals are not new trained controls. Audit not yet executed.

Task 2 complete: candidate99232 terminated exit0,8epochs2048updates, exact full initialization and all target-order SHAs matched. MinimumRGB selects epoch7 mean11.621979316075642, worse than mean-control11.565264860788982 and externalC10 10.942884524663288. Both full gates failed. No seed/head-size/epoch/loss tuning, no new independent-image allocation, no app/model-default change. Token q/key gradients nonzero; this proves active mechanism, not useful color correspondence.

Task 3 started: audit job launched after both training jobs terminal. Keep original frozen training source untouched.

Task 3 complete: audit session50304 exit0. All2445prepared assets/18weights/fullinitial/order hashes checked. All1280PNG and20 full RGB/paired/tone tables reproduced;192current/control/candidate native plus64externalC10 outputs exact.128fixed-weight counterfactual PNG retained. Contact sheets inspected:22927,88391,3026,27824,59779,9839,26768,64217. Both full gates reproduced false.

Task 4 underway: report, sanitized evidence463979bytes and three indices written. Historical diagnostic links now point to completed study. No implementation/source edits after runs began. Await final read-only review and publication checks; no fresh evaluation or app integration permitted by these failed gates.

Task 4 prepublication verification complete: read-only final review found no blocking/important issue; report/evidence hashes and metrics matched, all source and selected-state hashes unchanged, UTF8/relative links/secret/size/asset-scope checks passed.232-test result remains valid; no tested training implementation changed after that run. Commit/tag/remote verification occurs after this checkpoint; the final Git reference is the publication receipt, not an App version upgrade.
