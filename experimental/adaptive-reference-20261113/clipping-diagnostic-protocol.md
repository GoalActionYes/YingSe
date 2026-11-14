# Selected-weight eval clipping check

Prior direct pair-delta training already failed on the narrow cohort; do not repeat it as if untried. Before considering another loss recipe, measure strict raw output<0or>1 at frozen expanded-v2 epoch4,eval mode/all64consumeddev. No new training,weights/BN updates,independent images or loss tuning.

Report channel-level outside fraction and fraction of post-clamp RGB L1 error located on outside channels; raw L1 for context. Exact0/1 boundaries are not included because PyTorch clamp has nonzero boundary derivative. Synthetic tests cover error denominators/invalid inputs. A selected eval output frequency is not a training-trajectory measurement and cannot establish cause of weak reference response. Even zero outside error would not prove every earlier training gradient was preserved.

First reproduce all64selectedPNG hashes from native raw→clamp/round output. Hash GT/prepared content/reference inputs and selected checkpoint. Summarize all64 and prior32response strata. No GT-assisted inference,promotion,threshold search or modifying rendering. High outside counts only justify designing a separate matched loss study; low counts rule out assuming a large loss contribution from this snapshot alone.
