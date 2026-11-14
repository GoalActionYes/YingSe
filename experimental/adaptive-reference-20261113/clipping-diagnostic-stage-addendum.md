# Clarify the exact measured output stage

The completed coverage-clipping-diagnostic-v1 measurement calls appearance_forward. Its variable named raw is AFTER the author's Grid.apply_coeffs post_guide and BEFORE the training loop's outer clamp(0,1). It is not the affine transform output before internal clipping.

The source order is pre_guide → affine matrix/bias → clamp(-1,1) → post_guide → appearance_forward return → outer loss clamp(0,1). post_guide uses bounded monotone tables. Thus the observed zero outside fraction at the measured stage is expected from the output structure; it does not establish that internal clipping is absent or harmless, nor that previous training gradients were preserved.

The synthetic statistic tests remain valid for supplied unconstrained arrays, but do not change the stage of the native probe. A separate internal-stage probe must first reconstruct or instrument the affine stage with exact complete-output/native equivalence at fixed weights. Do not change gradients,rendering or training based on the completed outer-stage observation alone. Preserve the existing result and frozen source as recorded rather than rewriting them to claim deeper coverage.
