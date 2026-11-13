"""Fixed initialization for the one predeclared seed24 replication."""
import torch
from pairwise_context import enable_pairwise_context
from appearance_pairwise import enable_appearance_context


def initialize_replication_model(model,base_state):
    if hasattr(model,'pairwise_adapter') or hasattr(model,'reference_adapter'):
        raise ValueError('Replication initialization requires an unextended base model')
    model.load_state_dict(base_state,strict=True)
    torch.manual_seed(20261124)
    enable_pairwise_context(model)
    torch.manual_seed(20261125)
    enable_appearance_context(model)
    # Caller reseeds20261124 immediately before the training stream.
