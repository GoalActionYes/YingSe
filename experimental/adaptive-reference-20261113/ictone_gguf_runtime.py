"""Research-only compatibility helpers; do not modify installed vendor files."""
import types


def adapt_t5_gated_gguf(model):
    """Keep T5's gated FFN computation, selecting GGUF compute dtype for wo.

    Stock T5 casts activations to wo.weight.dtype, which is uint8 for GGUF
    packed storage. Casting to uint8 loses values before the linear layer.
    Only gated T5 modules with GGUF wo are changed; non-GGUF layers stay stock.
    """
    from diffusers.quantizers.gguf.utils import GGUFLinear
    from transformers.models.t5.modeling_t5 import T5DenseGatedActDense

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = self.dropout(hidden_gelu * hidden_linear)
        if hidden_states.dtype != self.wo.compute_dtype:
            hidden_states = hidden_states.to(self.wo.compute_dtype)
        return self.wo(hidden_states)

    changed = []
    for name, layer in model.named_modules():
        if isinstance(layer, T5DenseGatedActDense) and isinstance(layer.wo, GGUFLinear):
            layer.forward = types.MethodType(forward, layer)
            changed.append(name)
    return changed


def release_disk_group_storage(model):
    """For frozen, already-saved stock disk groups, release CPU placeholders.

    Keep each Parameter object and GGUF attributes alive. Stock onload assigns
    its saved packed bytes back to the same object. No weights may be updated
    after this switch: the files are the frozen source for every forward.
    This helper uses private Diffusers 0.36 internals and must be revalidated
    on dependency changes. It does not implement training/checkpoint saving.
    """
    import torch

    groups = []
    seen = set()
    for layer in model.modules():
        registry = getattr(layer, '_diffusers_hook', None)
        hook = registry.get_hook('group_offloading') if registry is not None else None
        if hook is None or id(hook.group) in seen:
            continue
        group = hook.group
        seen.add(id(group))
        assert group.stream is None, 'only synchronous disk groups are verified'
        assert group.offload_to_disk_path is not None and group._is_offloaded_to_disk
        from pathlib import Path
        assert Path(group.safetensors_file_path).is_file()
        assert all(not p.requires_grad for p in group.tensor_to_key), 'frozen inference only'

        def offload(self):
            assert self._is_offloaded_to_disk
            for tensor in self.tensor_to_key:
                tensor.data = torch.empty(0, dtype=tensor.dtype, device=self.offload_device)

        group._offload_to_disk = types.MethodType(offload, group)
        group._offload_to_disk()
        groups.append(group)
    assert groups, 'no initialized disk groups found'
    return groups
