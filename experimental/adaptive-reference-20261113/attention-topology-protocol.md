# Current13 attention conditioning topology probe

Inspect the actual frozen current13 epoch6 and pinned author source. No training/data-image access/app changes. Use one deterministic synthetic256RGB ramp content=(u,v,(u+v)/2), reference=(v,u,1-(u+v)/2), with u/v linspace0..1. Compute actual BT709 marginal128 histograms, hist_projection context, img_projection of64px target, and attention output. Snapshot current13 SHA and program/author source SHA.

Record context/query/output shapes and attention variation across spatial positions. Detach query/context from encoders, require query gradients, then backprop sum(attention_output * fixed linspace channelweights .5..1.5). Measure query-input, q-weight, kv-key-half and kv-value-half gradient maxima. Compare output when query input is negated with the same context. No checkpoint modification/saving or score/quality test. Verify original checkpoint/program hashes afterward.

If context has one token, softmax over a singleton must be1: this attention output cannot depend on query or select among multiple reference tokens. This says nothing by itself about final model expressivity or quality: target img features still enter fusion and target-dependent spatial grid/guide remain. The global vector may encode relationships. Do not call this a recovered Jianying algorithm, a software bug, proof that all tone mapping is global, or a proven cause of observed errors. It motivates a separately controlled multi-token reference experiment only, not product integration or automatic success.

Keep no user photos/model weights in Git. Publish code/numerical shape-gradient observations and constraints only; academic noncommercial context.
