# Expanded training byte preflight

Consume only frozen cohort-training-coverage-v1 cohort SHA256d001cec5f2f3386e6237627524b88ea42da70feb7bb8d49a60281e1df434aeb5. This metadata-preflight tool does not download or decode image bytes. No new independent set is allocated.

Exactly9162 asset paths from existing64 validation and expanded4096 training pairs are allowed. Retired validation/reserved families must remain excluded from training. Reuse remote metadata only from hash-verified original cohort-v1/v2/multiref download identities at the same pinned dataset revision. Request remaining paths from the pinned Hugging Face paths-info endpoint,100 paths per request,at most4 concurrent requests,60second per-request timeout. Retry only transient timeout/429/5xx at most3 attempts. Do not restart completed metadata parts on an observation timeout.

Each record must have one safe relative POSIX path, positive integer size≤20MiB, valid Git SHA1 identity and,when present,valid SHA256 LFS identity. No traversal, backslash, drive-qualified, absolute or duplicate paths. Exact requested path set required. Total frozen byte budget8GiB (based on≈6.83GB metadata-derived estimate,not a quality filter); disk free space must exceed3× total plus2GiB before future original/prepared copies. If size exceeds limits,stop and report; do not silently drop groups.

Persist each successful metadata response with its requested set and dataset/cohort/source identity. Resume only matching records. Final download-metadata.json/download-identity.json must be compatible with the existing verified byte downloader. Validate complete identity and bounds again before downloading. Preserve original bytes and make separately ICC-managed RGB8 files; no product or source-resolution fidelity claim from training preparation.

Subsequent image decoding and exact/perceptual duplicate study need a separate fixed protocol before reading new images. Near-duplicate collisions pause training without dropping/replacing rows to chase scores. The metadata report contains hashes/counts/bytes/reuse estimates only; private path records, image bytes and weights stay outside Git.
