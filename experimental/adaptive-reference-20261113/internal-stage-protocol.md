# Frozen internal affine observation

Measure expanded-v2 selectedepoch4 on all64consumeddev. Reconstruct exactly the author's Grid.apply_coeffs affine stage after coefficient/alpha normalization and pre_guide, before clamp(-1,1)/post_guide. Keep weights/source/data fixed,eval and no gradients/BN updates.

Synthetic tests cover known diagonal+bias/alpha normalization,batch/channel order,finite gradients and malformedcoefficients. Native reconstruction must reproduce original apply_coeffs full float output with maximumdifference0 for every64case and the saved selected PNG bytes. Preserve ordinary inference; no modified-gradient or unclamped-output product path.

Report strict affine<-1or>1 fractions,below/above counts and excess magnitude,all64 and previous32response strata. Exact boundary values are not outside. Do not evaluate affine values asRGB or useGT to choose an operation. Clipping frequency in an eval snapshot is not training-gradient history,semantic/cause evidence or proof that removing clipping improves anything. If frequency is small/zero,do not assume a large clipped-gradient explanation from this snapshot. No extra training/new independent allocation or gate changes.
